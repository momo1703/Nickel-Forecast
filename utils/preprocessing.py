import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(df, seq_len=60):
    # Attempt to find a date column
    date_column = next((col for col in df.columns if 'date' in col.lower()), None)
    
    if date_column is None:
        raise ValueError("No date column found. Make sure your CSV includes a column like 'Date'.")

    df[date_column] = pd.to_datetime(df[date_column])
    df.set_index(date_column, inplace=True)

    # Define expected features
    features = ['LME_Nickel_Spot', 'USD_Index', 'Oil_Price', 'PMI_Index']
    
    for feat in features:
        if feat not in df.columns:
            raise ValueError(f"Missing expected feature: '{feat}' in the uploaded data.")

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
