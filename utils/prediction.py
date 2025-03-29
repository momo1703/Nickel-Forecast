import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def predict_prices(model, X, y, scaler, tail_features):
    try:
        # === Model Prediction ===
        predictions = model.predict(X)

        # === Validation Checks ===
        arrays = {
            "X": X,
            "y": y,
            "tail_features": tail_features,
            "predictions": predictions
        }

        for name, arr in arrays.items():
            if np.any(np.isnan(arr)):
                raise ValueError(f"❌ {name} contains NaN values.")
            if np.any(np.isinf(arr)):
                raise ValueError(f"❌ {name} contains Inf values.")

        if predictions.shape[0] != tail_features.shape[0]:
            raise ValueError("❌ Prediction and tail feature count mismatch.")

        if predictions.shape[1] != 1:
            raise ValueError("❌ Predictions must be shape (n_samples, 1).")

        if tail_features.shape[1] < 4:
            raise ValueError("❌ Tail features must have at least 4 columns.")

        # === Prepare Inputs for Inverse Scaling ===
        predicted_full = np.hstack((predictions, tail_features[:, 1:]))
        actual_full = np.hstack((y.reshape(-1, 1), tail_features[:, 1:]))

        # === Inverse Transform ===
        predicted_prices = scaler.inverse_transform(predicted_full)[:, 0]
        actual_prices = scaler.inverse_transform(actual_full)[:, 0]

        return predicted_prices, actual_prices

    except Exception as e:
        raise ValueError(f"Prediction failed: {e}")  # 🔁 raise instead of silent return

def plot_predictions(predicted, actual):
    fig, ax = plt.subplots()
    ax.plot(actual, label="Actual Price")
    ax.plot(predicted, label="Predicted Price")
    ax.set_title("Nickel Price Forecast")
