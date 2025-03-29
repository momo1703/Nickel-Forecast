import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# === Setup ===
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)

# === Local imports ===
from train_model_streamlit import train_lstm_model
from utils.preprocessing import load_and_preprocess_data
from utils.prediction import predict_prices, plot_predictions

# === Streamlit UI ===
st.set_page_config(page_title="Nickel Price Forecast", layout="centered")
st.title("🔮 Nickel Price Forecast (LME-based)")
st.markdown("Forecast LME Nickel prices using AI (LSTM) and macro indicators.")

# === Upload or Load Data ===
uploaded_file = st.file_uploader("📁 Upload your CSV data", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully.")
    except Exception as e:
        st.error(f"❌ Failed to read uploaded file: {e}")
        st.stop()
else:
    default_path = os.path.join(BASE_DIR, "data", "lme_nickel_data.csv")
    try:
        df = pd.read_csv(default_path)
        st.info
