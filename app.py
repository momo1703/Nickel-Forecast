import streamlit as st
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
from utils.preprocessing import load_and_preprocess_data
from utils.prediction import predict_prices, plot_predictions

# App title
st.set_page_config(page_title="Nickel Price Forecast", layout="centered")
st.title("🔮 Nickel Price Forecast (LME-based)")
st.write("Forecast LME Nickel prices using AI (LSTM model) and macroeconomic indicators.")

# Load default CSV or user-uploaded file
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
        st.info("ℹ️ Using default dataset from /data/lme_nickel_data.csv")
    except FileNotFoundError:
        st.error("❌ Default dataset not found. Please upload a CSV file.")
        st.stop()

# Load and preprocess data
try:
    X_test, y_test, scaler, features_tail = load_and_preprocess_data(df)
except Exception as e:
    st.error(f"❌ Data preprocessing error: {e}")
    st.stop()

# Load trained model
model_path = "model/lstm_model.h5"
if not os.path.exists(model_path):
    st.warning("⚠️ Trained model not found. Please upload `lstm_model.h5` to the `model/` folder.")
    st.stop()

try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Predict and plot
try:
    predicted, actual = predict_prices(model, X_test, y_test, scaler, features_tail)
    plot_predictions(predicted, actual)
    st.success("✅ Forecast complete.")
except Exception as e:
    st.error(f"❌ Prediction error: {e}")
