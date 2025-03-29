import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def predict_prices(model, X, y, scaler, tail_features):
    predictions = model.predict(X)

    # ✅ Safety Checks
    if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
        raise ValueError("❌ Model predictions contain NaN or Inf values.")
    if np.any(np.isnan(tail_features)) or np.any(np.isinf(tail_features)):
        raise ValueError("❌ Tail features contain NaN or Inf values.")
    if predictions.shape[0] != tail_features.shape[0]:
        raise ValueError("❌ Prediction and tail feature lengths do not match.")

    try:
        predicted_full = np.hstack((predictions, tail_features[:, 1:]))
        actual_full = np.hstack((y.reshape(-1, 1), tail_features[:, 1:]))

        # ✅ Make sure data is within scaler range
        predicted_prices = scaler.inverse_transform(predicted_full)[:, 0]
        actual_prices = scaler.inverse_transform(actual_full)[:, 0]

    except Exception as e:
        raise ValueError(f"❌ Inverse transform failed: {e}")

    return predicted_prices, actual_prices
