import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def predict_prices(model, X, y, scaler, tail_features):
    # Predict using the model
    predictions = model.predict(X)

    # Safety checks before inverse_transform
    if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
        raise ValueError("Model predictions contain NaN or Inf values.")

    if np.any(np.isnan(tail_features)) or np.any(np.isinf(tail_features)):
        raise ValueError("Tail features contain NaN or Inf values.")

    # Combine predicted/actual target with tail features (excluding 1st column)
    predicted_full = np.hstack((predictions, tail_features[:, 1:]))
    actual_full = np.hstack((y.reshape(-1, 1), tail_features[:, 1:]))

    # Inverse scale
    predicted_prices = scaler.inverse_transform(predicted_full)[:, 0]
    actual_prices = scaler.inverse_transform(actual_full)[:, 0]

    return predicted_prices, actual_prices

def plot_predictions(predicted, actual_
