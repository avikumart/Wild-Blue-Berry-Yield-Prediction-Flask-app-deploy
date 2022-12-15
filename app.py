from flask import Flask, render_template, Response
from flask_restful import reqparse, Api
import flask

import numpy as np
import pandas as pd
import joblib
import ast

import os
import json

curr_path = os.path.dirname(os.path.realpath(__file__))

feature_cols = ['AverageRainingDays', 'clonesize', 'AverageOfLowerTRange',
    'AverageOfUpperTRange', 'honeybee', 'osmia', 'bumbles', 'andrena']

app = Flask(__name__)

@app.route('/api/predict', methods=['GET','POST'])
def api_predict():
    return {'message':"gotcha"}


if __name__ == "__main__":
    app.run()