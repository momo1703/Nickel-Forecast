import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(df, seq_len=10):
    date_column = next((col for col in df.columns if 'date' in col.lower()), None)
    if not date_column:
        raise ValueError("No date column found. Include a column like 'Date'.")

    df[date_column] = pd.to_datetime(df[date_column], dayfirst=True)
    df.set_index(date_column, inplace=True)

    features = ['LME_Nickel_Spot', 'USD_Index', 'Oil_Price', 'PMI_Index']
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    df = df[features].dropna()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(len(scaled) - seq_len):
        X.append(scaled[i:i+seq_len])
        y.append(scaled[i+seq_len][0])

    return np.array(X[-100:]), np.array(y[-100:]), scaler, scaled[-100:]
