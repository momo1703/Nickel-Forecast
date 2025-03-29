import os
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.losses import MeanSquaredError

def train_lstm_model(df, seq_len=10, model_path="model/lstm_model.h5"):
    try:
        # Detect date column
        date_column = next((col for col in df.columns if "date" in col.lower()), None)
        if not date_column:
            st.error("❌ Date column not found.")
            return None

        df[date_column] = pd.to_datetime(df[date_column], dayfirst=True)
        df.set_index(date_column, inplace=True)

        features = ['LME_Nickel_Spot', 'USD_Index', 'Oil_Price', 'PMI_Index']
        missing = [f for f in features if f not in df.columns]
        if missing:
            st.error(f"❌ Missing columns: {', '.join(missing)}")
            return None

        df = df[features].dropna()
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df)

        X, y = [], []
        for i in range(len(scaled) - seq_len):
            X.append(scaled[i:i+seq_len])
            y.append(scaled[i+seq_len][0])

        X, y = np.array(X), np.array(y)

        if len(X) == 0:
            st.error("❌ Not enough data to train. Try reducing SEQ_LEN.")
            return None

        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(seq_len, X.shape[2_
