import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_data(df, seq_len=10):
    # === Detect date column ===
    date_column = next((col for col in df.columns if 'date' in col.lower()), None)
    if not date_column:
        raise ValueError("❌ No date column found. Include a column like 'Date'.")

    # === Convert and sort date ===
    df[date_column] = pd.to_datetime(df[date_column], dayfirst=True, errors='coerce')
    df = df.dropna(subset=[date_column])
    df.sort_values(by=date_column, inplace=True)
    df.set_index(date_column, inplace=True)

    # === Required feature columns ===
    features = ['LME_Nickel_Spot', 'USD_Index', 'Oil_Price', 'PMI_Index']
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"❌ Missing required columns: {', '.join(missing)}")

    # === Drop rows with NaNs and scale ===
    df = df[features].dropna()
    if len(df) <= seq_len:
        raise ValueError(f"❌ Not enough rows to build sequences. Only {len(df)} rows after cleaning, but need at least {seq_len + 1}.")

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(len(scaled) - seq_len):
        X.append(scaled[i:i+seq_len])
        y.append(scaled[i+seq_len][0])  # Target is Nickel Spot Price

    # === Final checks before return ===
    X = np.array(X)
    y = np.array(y)
    features_tail = scaled[seq_len:]

    if X.shape[0] != y.shape[0] or y.shape[0] != features_tail.shape[0]:
        raise ValueError("❌ Mismatch between X, y, and tail shapes after preprocessing.")

    return X, y, scaler, features_tail
