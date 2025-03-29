import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# === Set up paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UTILS_DIR = os.path.join(BASE_DIR, "utils")
sys.path.append(BASE_DIR)
sys.path.append(UTILS_DIR)

# ✅ Safe dynamic imports (alternative to "from utils.prediction import ...")
try:
        from utils.prediction import predict_prices, plot_predictions
        from utils.preprocessing import load_and_preprocess_data
except ImportError:
    st.error("❌ Could not import utility functions. Make sure 'utils/' folder has prediction.py and preprocessing.py")
    st.stop()

from train_model_streamlit import train_lstm_model

# === Streamlit UI ===
st.set_page_config(page_title="Nickel Price Forecast", layout="centered")
st.title("🔮 Nickel Price Forecast (LME-based)")
st.markdown("Forecast LME Nickel prices using AI (LSTM) and macro indicators.")

# === Upload or Load Default CSV ===
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
        st.info("ℹ️ Using default dataset from /data/lme_nickel_data.csv")
    except FileNotFoundError:
        st.error("❌ No dataset found. Please upload a CSV.")
        st.stop()

# === Preprocess the data ===
try:
    X_test, y_test, scaler, features_tail = load_and_preprocess_data(df)
except Exception as e:
    st.error(f"❌ Preprocessing error: {e}")
    st.stop()

# === Load or Retrain Model ===
model_path = os.path.join(BASE_DIR, "model", "lstm_model.h5")
if not os.path.exists(model_path):
    st.warning("⚠️ No trained model found. Training a new one...")
    trained_model = train_lstm_model(df)
    if trained_model:
        model = trained_model
        st.success("✅ New model trained and ready.")
    else:
        st.error("❌ Model training failed.")
        st.stop()
else:
    try:
        model = load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss=MeanSquaredError())
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

# === Run Prediction ===
try:
    predicted, actual = predict_prices(model, X_test, y_test, scaler, features_tail)
    plot_predictions(predicted, actual)
    st.success("✅ Forecast complete.")
except Exception as e:
    st.error(f"❌ Prediction error: {e}")
    st.info("📌 Tip: Ensure your CSV has required columns and ~20+ clean rows.")
