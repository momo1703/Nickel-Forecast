import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(df, seq_len=60):
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    features = ['LME_Nickel_Spot', 'USD_Index', 'Oil_Price', 'PMI_Index']
    df = df[features].dropna()

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(len(scaled) - seq_len):
        X.append(scaled[i:i+seq_len])
        y.append(scaled[i+seq_len][0])
    
    X = np.array(X)
    y = np.array(y)

    return X[-100:], y[-100:], scaler, scaled[-100:]