# import libraries
import sqlite3
from pathlib import Path
import os
import pandas as pd
import numpy as np
import gc
import pickle
import torch
from torch.nn import BCEWithLogitsLoss, BCELoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from tqdm import tqdm, trange


# Implement a machine learning pipeline that is responsible for:
# - Loads data from the SQLite database
# - Splits the dataset into training and test sets
# - Builds a text processing and machine learning pipeline
# - Trains and tunes a model using GridSearchCV
# - Outputs results on the test set
# - Exports the final model as a pickle file

# - Feel free to use any libraries
# - You should be able to run this file as a script and save the best classifier as a pickle file.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

categories = []
max_length = 300
batch_size = 32 # batch size
epochs = 10 # Number of training epochs (authors recommend between 2 and 4)

def load_data_and_build_labels():
    global categories
    conn = sqlite3.connect('../data/db.sqlite')
    data = pd.read_sql('select id, message, categories from disaster', conn)
    data.set_index('id')
    categ = data.values[0][2].split(';')
    categories = list(map(lambda el:el.split('-')[0],categ))
    def extract_label(s):
        l = s.split(';')
        # there are some error in collecting the data where some labels are given 2 instead of 0/1,
        # I will assume that 2 means 1 and fix it here when constructing one hot encoding
        one_hot = list(map(
            lambda el: int(el.split('-')[-1]) if int(el.split('-')[-1]) == 0 or int(el.split('-')[-1]) == 1 else 1,l)
        )
        return one_hot
    # create one hot encoding of the labels
    data['one_hot'] = data['categories'].apply(lambda x:extract_label(x))
    del data['categories']
    # add other columns of each category and it is corresponding value 0/1, this will serve to do data analysis later on
    one_hot_transposed = np.array(data.one_hot.to_list()).transpose()
    for i,l in enumerate(categories):
        data[l] = one_hot_transposed[i]
    return data

def data_analysis_and_fix_data(df):

    print('Unique comments: ', df.message.nunique() == df.shape[0])
    # I droped all duplicates (261 samples) because 1) they represent problems as their labels are always the same
    # which will prone the model to errors 2) the # of droped samples is relatively small but we will see later on
    # if this affects the model
    df.drop_duplicates(subset=['message'],keep=False, inplace=True)
    # check null data
    print('Null values: ', df.isnull().values.any())
    # check average length and std of samples
    print('average sentence length: ', df.message.str.split().str.len().mean())
    print('stdev sentence length: ', df.message.str.split().str.len().std())

    num_labels = len(categories)
    torch.save(num_labels,'num_labels')
    print('Label columns: ', categories)
    # Label counts, may need to downsample or upsample
    print('Count of 1 per label: \n', df[categories].sum(), '\n')
    print('Count of 0 per label: \n', df[categories].eq(0).sum())

    return df

def prepare_data(df):
    # shuffle rows
    df = df.sample(frac=1).reset_index(drop=True)

    labels = list(df.one_hot.values)
    samples = list(df.message.values)

    # Identifying indices of 'one_hot' entries that only occur once - this will allow me to stratify split
    # the training data later on
    label_counts = df.one_hot.astype(str).value_counts()
    one_freq = label_counts[label_counts == 1].keys()
    one_freq_idxs = sorted(list(df[df.one_hot.astype(str).isin(one_freq)].index), reverse=True)
    print('df label indices with only one instance: ', one_freq_idxs)

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)  # tokenizer
    # used the depricated version of the method because it outputs lists which I need to do the stratified split
    encodings = tokenizer.batch_encode_plus(samples,max_length=max_length, padding=True,truncation=True)
    input_ids = encodings['input_ids']  # tokenized and encoded sentences
    attention_masks = encodings['attention_mask']  # attention masks

    # Gathering single instance inputs to force into the training set after stratified split
    one_freq_input_ids = [input_ids.pop(i) for i in one_freq_idxs]
    one_freq_attention_masks = [attention_masks.pop(i) for i in one_freq_idxs]
    one_freq_labels = [labels.pop(i) for i in one_freq_idxs]

    # Use train_test_split to split the data into train, validation and test sets
    tot_inputs, test_inputs, tot_labels, test_labels, tot_masks, test_masks = train_test_split(
        input_ids, labels, attention_masks, random_state=2020, test_size=0.10, stratify=labels)
    train_inputs, validation_inputs, train_labels, validation_labels, train_masks, validation_masks = train_test_split(
        tot_inputs, tot_labels, tot_masks,
        random_state=2020, test_size=0.10, stratify=tot_labels)

    # Add one frequency data to train data
    train_inputs.extend(one_freq_input_ids)
    train_labels.extend(one_freq_labels)
    train_masks.extend(one_freq_attention_masks)

    # Convert all of our data into torch tensors, the required datatype for our model
    train_inputs = torch.tensor(train_inputs)
    train_labels = torch.tensor(train_labels)
    train_masks = torch.tensor(train_masks)

    validation_inputs = torch.tensor(validation_inputs)
    validation_labels = torch.tensor(validation_labels)
    validation_masks = torch.tensor(validation_masks)

    test_inputs = torch.tensor(test_inputs)
    test_labels = torch.tensor(test_labels)
    test_masks = torch.tensor(test_masks)

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    torch.save(validation_dataloader, 'validation_data_loader')
    torch.save(train_dataloader, 'train_data_loader')
    torch.save(test_dataloader,'test_data_loader')
    torch.save(categories, 'labels')

def train_and_validate():

    num_labels = torch.load('num_labels')
    # empty cache to allow more memory for gpu
    gc.collect()
    torch.cuda.empty_cache()
    # Load model, the pretrained model will include a single linear classification layer on top for classification.
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)
    model.cuda()

    train_dataloader = torch.load('train_data_loader')

    # setting custom optimization parameters.
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, correct_bias=True)

    # Store the loss and accuracy for plotting if needed
    train_loss_set = []

    # Store the F1 to stop training if F1 flattens
    f1_set = []

    # trange is a tqdm wrapper around the normal python range
    for i in trange(epochs, desc="Epoch"):
        # check if F1 has flattened, note here that because of computational capacity limitation, I assume that flatten
        # for me is only to check that for 2 consecutive epochs F1 haven't increased by at least 1 %, otherwise if
        # F1 continues to grow then it will be capped by the # epochs
        if i > 2:
            f1_current = f1_set[-1]
            f1_previous = f1_set[-2]
            f1_before_previous = f1_set[-3]
            diff_curr = f1_current - f1_previous
            diff_prev = f1_previous - f1_before_previous
            diff = (diff_curr + diff_prev)/2
            if diff < 1:
                break

        # Training

        # Set our model to training mode (as opposed to evaluation mode)
        model.train()

        # Tracking variables
        tr_loss = 0  # running loss
        nb_tr_examples, nb_tr_steps = 0, 0

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()

            # Forward pass for multilabel classification
            outputs = model(b_input_ids, attention_mask=b_input_mask)
            logits = outputs[0]
            loss_func = BCEWithLogitsLoss()
            loss = loss_func(logits.view(-1, num_labels),
                             b_labels.type_as(logits).view(-1, num_labels))  # convert labels to float for calculation
            train_loss_set.append(loss.item())

            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss / nb_tr_steps))

        ###############################################################################

        # Validation

        validation_dataloader = torch.load('validation_data_loader')

        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        # Variables to gather full output
        logit_preds, true_labels, pred_labels, tokenized_texts = [], [], [], []

        # Predict
        for i, batch in enumerate(validation_dataloader):
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
                # Forward pass
                outs = model(b_input_ids, attention_mask=b_input_mask)
                b_logit_pred = outs[0]
                pred_label = torch.sigmoid(b_logit_pred)

                b_logit_pred = b_logit_pred.detach().cpu().numpy()
                pred_label = pred_label.to('cpu').numpy()
                b_labels = b_labels.to('cpu').numpy()

            tokenized_texts.append(b_input_ids)
            logit_preds.append(b_logit_pred)
            true_labels.append(b_labels)
            pred_labels.append(pred_label)

        # Flatten outputs
        pred_labels = [item for sublist in pred_labels for item in sublist]
        true_labels = [item for sublist in true_labels for item in sublist]

        # Calculate Accuracy
        threshold = 0.50
        pred_bools = [pl > threshold for pl in pred_labels]
        true_bools = [tl == 1 for tl in true_labels]
        val_f1_accuracy_macro = f1_score(true_bools, pred_bools, average='macro') * 100
        val_f1_accuracy_micro = f1_score(true_bools, pred_bools, average='micro') * 100
        val_flat_accuracy = accuracy_score(true_bools, pred_bools) * 100

        f1_set.append(val_f1_accuracy_micro)

        print('F1 Validation Score Micro: ', val_f1_accuracy_micro)
        print('F1 Validation Score Macro: ', val_f1_accuracy_macro)
        print('Flat Validation Accuracy: ', val_flat_accuracy)

    # store the best model
    model.save_pretrained('finetuned_distilbert')

    # finetune the threshold by maximizing F1 first with macro_thresholds on the order of e ^ -1 then with
    # micro_thresholds on the order of e ^ -2
    macro_thresholds = np.array(range(1, 10)) / 10

    f1_results, flat_acc_results = [], []
    for th in macro_thresholds:
        pred_bools = [pl > th for pl in pred_labels]
        validation_f1_accuracy = f1_score(true_bools, pred_bools, average='macro')
        f1_results.append(validation_f1_accuracy)

    best_macro_th = macro_thresholds[np.argmax(f1_results)]  # best macro threshold value

    micro_thresholds = (np.array(range(10)) / 100) + best_macro_th  # calculating micro threshold values

    f1_results, flat_acc_results = [], []
    for th in micro_thresholds:
        pred_bools = [pl > th for pl in pred_labels]
        test_f1_accuracy = f1_score(true_bools, pred_bools, average='micro')
        f1_results.append(test_f1_accuracy)

    best_threshold = micro_thresholds[np.argmax(f1_results)]  # best threshold value
    # store the best threshold value
    torch.save(best_threshold, 'best_classification_threshold')


def test_and_print_report():

    # load model
    model = DistilBertForSequenceClassification.from_pretrained('finetuned_distilbert').to(device)
    # load test dataloader
    test_dataloader = torch.load('test_data_loader')
    # load list of labels
    labels = torch.load('labels')
    # load best classification threshold
    threshold = torch.load('best_classification_threshold')

    # Put model in evaluation mode to evaluate loss on the test set
    model.eval()

    # track variables
    logit_preds, true_labels, pred_labels, tokenized_texts = [], [], [], []

    # Predict
    for i, batch in enumerate(test_dataloader):
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from the dataloader
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            # Forward pass
            outs = model(b_input_ids, attention_mask=b_input_mask)
            b_logit_pred = outs[0]
            pred_label = torch.sigmoid(b_logit_pred)

            b_logit_pred = b_logit_pred.detach().cpu().numpy()
            pred_label = pred_label.to('cpu').numpy()
            b_labels = b_labels.to('cpu').numpy()

        tokenized_texts.append(b_input_ids)
        logit_preds.append(b_logit_pred)
        true_labels.append(b_labels)
        pred_labels.append(pred_label)

    # Flatten outputs
    tokenized_texts = [item for sublist in tokenized_texts for item in sublist]
    pred_labels = [item for sublist in pred_labels for item in sublist]
    true_labels = [item for sublist in true_labels for item in sublist]
    # Converting flattened binary values to boolean values
    true_bools = [tl == 1 for tl in true_labels]

    pred_bools = [pl > threshold for pl in pred_labels]  # boolean output after thresholding

    # Print and save classification report
    print('Test F1 Accuracy Micro: ', f1_score(true_bools, pred_bools, average='micro'))
    print('Test F1 Accuracy Macro: ', f1_score(true_bools, pred_bools, average='macro'))
    print('Test Flat Accuracy: ', accuracy_score(true_bools, pred_bools), '\n')
    clf_report = classification_report(true_bools, pred_bools, target_names=labels)
    pickle.dump(clf_report, open('classification_report.txt', 'wb'))  # save report
    print(clf_report)



train , validation, test = Path('./train_data_loader'), Path('./validation_data_loader'), Path('./test_data_loader')
finetuned_model = './finetuned_distilbert'
if not (train.is_file() and validation.is_file() and test.is_file()):
    data = load_data_and_build_labels()
    data = data_analysis_and_fix_data(data)
    prepare_data(data)
if not os.path.exists(finetuned_model):
    train_and_validate()
test_and_print_report()