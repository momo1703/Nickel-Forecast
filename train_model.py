import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.losses import MeanSquaredError  # ✅ Use real object

df = pd.read_csv("data/lme_nickel_data.csv")
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df.set_index('Date', inplace=True)

features = ['LME_Nickel_Spot', 'USD_Index', 'Oil_Price', 'PMI_Index']
df = df[features].dropna()

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

SEQ_LEN = 10
X, y = [], []
for i in range(len(scaled) - SEQ_LEN):
    X.append(scaled[i:i+SEQ_LEN])
    y.append(scaled[i+SEQ_LEN][0])

X, y = np.array(X), np.array(y)

if len(X) == 0:
    raise ValueError("❌ Not enough data to create sequences. Reduce SEQ_LEN or add more data.")

model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(SEQ_LEN, X.shape[2])),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

model.compile()
