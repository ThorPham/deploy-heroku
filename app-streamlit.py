import pandas as pd
import numpy as np
import joblib
import streamlit as st

#load the model
model=open("linear_regression_model.pkl","rb")
lr_model=joblib.load(model)


def lr_prediction(var_1,var_2,var_3,var_4,var_5):
    pred_arr=np.array([var_1,var_2,var_3,var_4,var_5])
    preds=pred_arr.reshape(1,-1)
    preds=preds.astype(int)
    model_prediction=lr_model.predict(preds)
    return model_prediction

st.title('Hello my app')
var_1 = st.text_input('var_1')
var_2 = st.text_input('var_2')
var_3 = st.text_input('var_3')
var_4 = st.text_input('var_4')
var_5 = st.text_input('var_5')
if st.button('predict'):
    predict = lr_prediction(var_1,var_2,var_3,var_4,var_5)
    st.text(f'predict {predict}')