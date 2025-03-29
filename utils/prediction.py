import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def predict_prices(model, X, y, scaler, tail_features):
    try:
        predictions = model.predict(X)

        # === Validation ===
        if predictions.shape[0] != tail_features.shape[0]:
            raise ValueError(f"Prediction shape {predictions.shape} vs tail_features {tail_features.shape} mismatch.")
        if predictions.shape[1] != 1:
            raise ValueError("Predictions must be shape (n_samples, 1).")
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # === Check original scaler input shape ===
        expected_shape = scaler.n_features_in_
        tail_cols = tail_features[:, 1:].shape[1]
        total_cols = 1 + tail_cols

        if total_cols != expected_shape:
            raise ValueError(f"Scaler was fit on {expected_shape} features but prediction input has {total_cols} columns.")

        # === Prepare inverse scaling input ===
        predicted_full = np.hstack((predictions, tail_features[:, 1:]))
        actual_full = np.hstack((y, tail_features[:, 1:]))

        if not np.isfinite(predicted_full).all():
            raise ValueError("predicted_full contains invalid values.")

        predicted_prices = scaler.inverse_transform(predicted_full)[:, 0]
        actual_prices = scaler.inverse_transform(actual_full)[:, 0]

        return predicted_prices, actual_prices

    except Exception as e:
        raise ValueError(f"Prediction failed: {e}")

def plot_predictions(predicted, actual):
    fig, ax = plt.subplots()
    ax.plot(actual, label="Actual Price", linewidth=2)
    ax.plot(predicted, label="Predicted Price", linestyle="--", linewidth=2)
    ax.set_title("Nickel Price Forecast")
    ax.set_xlabel("Time")
    ax.set_ylabel("LME Nickel Spot Price")
    ax.legend()
    st.pyplot(fig)
