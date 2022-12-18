from flask import Flask, render_template, Response
from flask_restful import reqparse, Api
import flask

import numpy as np
import pandas as pd
import ast

import os
import json

from model import predict_yield

curr_path = os.path.dirname(os.path.realpath(__file__))

feature_cols = ['AverageRainingDays', 'clonesize', 'AverageOfLowerTRange',
    'AverageOfUpperTRange', 'honeybee', 'osmia', 'bumbles', 'andrena']

context_dict = {
    'feats': feature_cols,
    'zip': zip,
    'range': range,
    'len': len,
    'list': list,
}

app = Flask(__name__)
api = Api(app)

# # FOR FORM PARSING
parser = reqparse.RequestParser()
parser.add_argument('list', type=list)

@app.route('/api/predict', methods=['GET','POST'])
def api_predict():
    data = flask.request.form.get('single input')
    
    # converts json to int 
    i = ast.literal_eval(data)
    
    y_pred = predict_yield(np.array(i).reshape(1,-1))
    
    return {'message':"success", "pred":json.dumps(int(y_pred))}

@app.route('/')
def index():
    
    # render the index.html templete
    
    return render_template("index.html", **context_dict)

@app.route('/predict', methods=['POST'])
def predict():
    # flask.request.form.keys() will print all the input from form
    test_data = []
    for val in flask.request.form.values():
        test_data.append(float(val))
    test_data = np.array(test_data).reshape(1,-1)

    y_pred = predict_yield(test_data)
    context_dict['pred']= y_pred

    print(y_pred)

    return render_template('index.html', **context_dict)

if __name__ == "__main__":
    app.run()