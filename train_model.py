import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.losses import MeanSquaredError

# === Load and clean data ===
df = pd.read_csv("data/lme_nickel_data.csv")
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df.dropna(subset=['Date'], inplace=True)
df.set_index('Date', inplace=True)

# === Define features ===
features = ['LME_Nickel_Spot', 'USD_Index', 'Oil_Price', 'PMI_Index']
df = df[features].dropna()

# === Scale ===
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

# === Sequence generation ===
SEQ_LEN = 3  # ✅ Reduced from 10 to 3
X, y = [], []
for i in range(len(scaled) - SEQ_LEN):
    X.append(scaled[i:i+SEQ_LEN])
    y.append(scaled[i+SEQ_LEN][0])

X, y = np.array(X), np.array(y)

if len(X) == 0:
    raise ValueError(f"❌ Not enough data to create sequences with SEQ_LEN={SEQ_LEN}. Add more rows or lower SEQ_LEN.")

# === Build LSTM model ===
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(SEQ_LEN, X.shape[2])),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss=MeanSquaredError())

# === Train ===
model.fit(X, y, epochs=50, batch_size=8)

# === Save model ===
os.makedirs("model", exist_ok=True)
model.save("model/lstm_model.h5")
print("✅ Model trained and saved as model/lstm_model.h5")
