import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

# === Load Data ===
df = pd.read_csv("data/lme_nickel_data.csv")
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df.set_index('Date', inplace=True)

# === Select Features ===
features = ['LME_Nickel_Spot', 'USD_Index', 'Oil_Price', 'PMI_Index']
df = df[features].dropna()

# === Normalize ===
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

# === Create Sequences ===
SEQ_LEN = 10
X, y = [], []

for i in range(len(scaled) - SEQ_LEN):
    X.append(scaled[i:i+SEQ_LEN])
    y.append(scaled[i+SEQ_LEN][0])  # Predict Nickel spot price

X, y = np.array(X), np.array(y)

# === Build LSTM Model ===
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(SEQ_LEN, X.shape[2])),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# === Train ===
print("Training LSTM model...")
model.fit(X, y, epochs=50, batch_size=8, verbose=1)
print("✅ Training complete.")

# === Save the model ===
os.makedirs("model", exist_ok=True)
model.save("model/lstm_model.h5")
print("✅ Model saved as model/lstm_model.h5")
