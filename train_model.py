import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

# Load CSV data
df = pd.read_csv("data/lme_nickel_data.csv")
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df.set_index('Date', inplace=True)

# Select features
features = ['LME_Nickel_Spot', 'USD_Index', 'Oil_Price', 'PMI_Index']
df = df[features].dropna()

# Scale features
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

# Prepare sequences
SEQ_LEN = 10
X, y = [], []
for i in range(len(scaled) - SEQ_LEN):
    X.append(scaled[i:i+SEQ_LEN])
    y.append(scaled[i+SEQ_LEN][0])  # Predict LME Nickel Spot Price

X = np.array(X)
y = np.array(y)

# Build LSTM model
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(SEQ_LEN, X.shape[2])),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X, y, epochs=50, batch_size=8)

# ✅ Ensure model folder exists before saving
os.makedirs("model", exist_ok=True)

# Save the trained model
model.save("model/lstm_model.h5")
print("✅ Model saved at model/lstm_model.h5")
