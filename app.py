import streamlit as st
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
from utils.preprocessing import load_and_preprocess_data
from utils.prediction import predict_prices, plot_predictions
from train_model_streamlit import train_lstm_model

st.set_page_config(page_title="Nickel Price Forecast", layout="centered")
st.title("🔮 Nickel Price Forecast (LME-based)")
st.markdown("Forecast LME Nickel prices using AI (LSTM) + macro indicators")

uploaded_file = st.file_uploader("📁 Upload your CSV data", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully.")
    except Exception as e:
        st.error(f"❌ Failed to read the file: {e}")
        st.stop()
else:
    try:
        df = pd.read_csv("data/lme_nickel_data.csv")
        st.info("ℹ️ Using default dataset from `/data/lme_nickel_data.csv`")
    except FileNotFoundError:
        st.error("❌ No data found. Please upload a CSV file.")
        st.stop()

try:
    X_test, y_test, scaler, features_tail = load_and_preprocess_data(df)
except Exception as e:
    st.error(f"❌ Preprocessing error: {e}")
    st.stop()

if st.button("🔁 Retrain
