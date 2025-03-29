import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def predict_prices(model, X, y, scaler, tail_features):
    # Predict using the model
    predictions = model.predict(X)

    # Check for NaNs or Infs before inverse_transform
    if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
        raise ValueError("Model predictions contain NaN or Inf values.")

    if np.any(np.isnan(tail_features)) or np.any(np.isinf(tail_features)):
        raise ValueError("Tail features contain NaN or Inf values.")

    # Reconstruct full feature vectors for inverse_transform
    predicted_full = np.hstack((predictions, tail_features[:, 1:]))
    actual_full = np.hstack((y.reshape(-1, 1), tail_features[:, 1:]))

    predicted_prices = scaler.inverse_transform(predicted_full)[:, 0]
    actual_prices = scaler.inverse_transform(actual_full)[:, 0]

    return predicted_prices, actual_prices

def plot_predictions(predicted, actual):
    fig, ax = plt.subplots()
    ax.plot(actual, label="Actual Price")
    ax.plot(predicted, label="Predicted Price")
    ax.set_title("Nickel Price Forecast")
    ax.legend()
    st.pyplot(fig)
