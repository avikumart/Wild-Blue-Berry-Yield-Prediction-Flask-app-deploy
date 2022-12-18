import joblib 
import pandas as pd
import numpy as np
import os

curr_path = os.path.dirname(os.path.realpath(__file__))
xgb_model = joblib.load(curr_path + "/model/wbb_xgb_model2.joblib")

def predict_yield(attributes: np.ndarray):
    """ Returns Blueberry Yield value"""
    # print(attributes.shape) # (1,10)

    pred = xgb_model.predict(attributes)
    print("Yield predicted")

    return pred[0]