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


# max_length = train_classifier.max_length
max_length = 300

app = Flask(__name__)

# TODO: implement tokenizer to tokenize input text
def tokenize(text):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)  # tokenizer
    encoding = tokenizer(text, return_tensors='pt' , max_length=max_length, padding=True, truncation=True)
    input_ids = encoding['input_ids']  # tokenized and encoded sentence
    attention_masks = encoding['attention_mask']  # attention masks
    return input_ids, attention_masks

# TODO: load data form db into pandas df
def load_data():
    conn = sqlite3.connect('../data/db.sqlite')
    df = pd.read_sql('select id, message, original, genre, categories from disaster', conn)
    df.set_index('id')
    categ = df.values[0][4].split(';')
    categories = list(map(lambda el: el.split('-')[0], categ))

    def extract_label(s):
        l = s.split(';')
        # there are some error in collecting the data where some labels are given 2 instead of 0/1,
        # I will assume that 2 means 1 and fix it here when constructing one hot encoding
        one_hot = list(map(
            lambda el: int(el.split('-')[-1]) if int(el.split('-')[-1]) == 0 or int(el.split('-')[-1]) == 1 else 1, l)
        )
        return one_hot

    one_hot = df['categories'].apply(lambda x: extract_label(x))
    del df['categories']
    # add other columns of each category and it is corresponding value 0/1, this will serve to do data analysis later on
    one_hot_transposed = np.array(one_hot.to_list()).transpose()
    for i, l in enumerate(categories):
        df[l] = one_hot_transposed[i]
    return df


# TODO: load model
def load_model():
    model = torch.load('../models/finetuned_distilbert')
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
    query = None
    # TODO: use model to predict classification for query
    classification_result = None

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    # main()
    index()