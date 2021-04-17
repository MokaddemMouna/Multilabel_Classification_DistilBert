import json
import joblib

import plotly
import pandas as pd
import numpy as np
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine
import sqlite3

import torch
from transformers import *

from Disaster_Classification.models.train_classifier import max_length

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
threshold = torch.load('../models/best_classification_threshold')
labels = torch.load('../models/labels')

app = Flask(__name__)

# TODO: implement tokenizer to tokenize input text
def tokenize(text):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)  # tokenizer
    encoding = tokenizer(text, return_tensors='pt' , max_length=max_length, padding=True, truncation=True)
    input_ids = encoding['input_ids']  # tokenized and encoded sentence
    attention_masks = encoding['attention_mask']  # attention masks
    return input_ids.to(device), attention_masks.to(device)

# TODO: load data form db into pandas df
def load_data():
    conn = sqlite3.connect('../data/db.sqlite')
    df = pd.read_sql('select * from disaster', conn)
    return df


# TODO: load model
def load_model():
    model = DistilBertForSequenceClassification.from_pretrained('../models/finetuned_distilbert').to(device)
    return model



# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # load data
    df = load_data()

    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    category_ratio = list(df[df.columns[4:]].sum())
    category_name = df.columns[4:]

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_name,
                    y=category_ratio
                )
            ],

            'layout': {
                'title': 'Distribution of help category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "category",
                    'tickangle' : 30
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # TODO: save user input in query
    query = request.args.get('query')
    # TODO: use model to predict classification for query
    model = load_model()
    model.eval()
    input_ids , attention_masks = tokenize(query)
    out = model(input_ids, attention_mask=attention_masks)
    logit_pred = out[0]
    pred_label = torch.sigmoid(logit_pred)
    pred_label = pred_label.detach().to('cpu').tolist()[0]
    predictions = [1 if pl > threshold else 0 for pl in pred_label]
    classification_results = dict(zip(labels,predictions))
    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()