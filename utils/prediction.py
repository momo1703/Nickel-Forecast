import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def predict_prices(model, X, y, scaler, tail_features):
    try:
        # === Run model prediction ===
        predictions = model.predict(X)

        # === Check for shape mismatches ===
        if predictions.shape[0] != tail_features.shape[0]:
            raise ValueError(f"❌ Predictions and tail features row mismatch: {predictions.shape[0]} vs {tail_features.shape[0]}")
        if predictions.shape[1] != 1:
            raise ValueError("❌ Predictions must be shape (n_samples, 1).")

        # === Get how many features the scaler expects ===
        expected_cols = scaler.n_features_in_

        # === Match scaler input by trimming tail_features ===
        # Use only the right number of trailing features to match scaler.fit() input
        cols_needed_from_tail = expected_cols - 1
        if tail_features.shape[1] < cols_needed_from_tail + 1:
            raise ValueError(f"❌ Tail features have {tail_features.shape[1]} columns but expected at least {cols_needed_from_tail + 1}")

        tail_trimmed = tail_features[:, 1:1+cols_needed_from_tail]

        # === Build full arrays for inverse_transform ===
        predicted_full = np.hstack((predictions, tail_trimmed))
        actual_full = np.hstack((y.reshape(-1, 1), tail_trimmed))

        if predicted_full.shape[1] != expected_cols:
            raise ValueError(f"❌ Predicted input shape mismatch: expected {expected_cols}, got {predicted_full.shape[1]}")

        # === Inverse scale to original price ===
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
