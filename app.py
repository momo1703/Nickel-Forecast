import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# 🔧 Make sure current directory is in Python path (for safe imports)
sys.path.append(os.path.dirname(__file__))

from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from utils.preprocessing import load_and_preprocess_data
from utils.prediction import predict_prices, plot_predictions
from train_model_streamlit import train_lstm_model  # ✅ Works if file is next to app.py

# Streamlit page setup
st.set_page_config(page_title="Nickel Price Forecast", layout="centered")
st.title("🔮 Nickel Price Forecast (LME-based)")
st.markdown("Forecast LME Nickel prices using AI (LSTM) + macro indicators")

# Upload CSV or load default
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
    except File
