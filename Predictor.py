import streamlit as st
import joblib
import numpy as np
import pandas as pd 
import shap
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
#pip install xgboost==2.0.3 --no-deps
#import xgboost
#model = xgboost.Booster()
#model.load_model('XGB.json')
df2 =pd.read_csv('x_test.csv')

