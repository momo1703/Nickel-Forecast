import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from utils.preprocessing import load_and_preprocess_data
from utils.prediction import predict_prices, plot_predictions

st.title("Nickel Price Forecast (LME-based)")
st.write("AI-powered forecast using historical data, macro indicators, and LSTM model")

uploaded_file = st.file_uploader("Upload your CSV data", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("data/lme_nickel_data.csv")

X_test, y_test, scaler, features_tail = load_and_preprocess_data(df)

model = load_model("model/lstm_model.h5")

predicted, actual = predict_prices(model, X_test, y_test, scaler, features_tail)

plot_predictions(predicted, actual)