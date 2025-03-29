import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(df, seq_len=10):
    date_column = next((col for col in df.columns if 'date' in col.lower()), None)
    if not date_column:
        raise ValueError("No date column found. Include a column like 'Date' in your CSV.")

    df[date_column] = pd.to_datetime(df[date_column], dayfirst=True)
    df.set_index(date_column, inplace=True)

    expected_features = ['LME_Nickel_Spot', 'USD_Index', 'Oil_Price', 'PMI_Index']
    missing = [feat for feat in expected_features if feat not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {', '.join(missing)}")

    df = df[expected_features].dropna()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(len(scaled) - seq_len):
        X.append(scaled[i:i+seq_len])
        y.append(scaled[i+seq_len][0])

    return np.array(X[-100:]), np.array(y[-100:]), scaler, scaled[-100:]
