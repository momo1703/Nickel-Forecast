import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def predict_prices(model, X, y, scaler, tail_features):
    try:
        predictions = model.predict(X)

        # === Validate Arrays ===
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
            if not np.isfinite(arr).all():
                raise ValueError(f"❌ {name} contains non-finite values.")

        # === Check Shapes ===
        if predictions.shape[0] != tail_features.shape[0]:
            raise ValueError("❌ Prediction and tail feature count mismatch.")
        if predictions.shape[1] != 1:
            raise ValueError("❌ Predictions must be shape (n_samples, 1).")
        if tail_features.shape[1] < 2:
            raise ValueError("❌ Tail features must have at least 2 columns.")

        # === Prepare Data for Inverse Scaling ===
        predicted_full = np.hstack((predictions, tail_features[:, 1:]))
        actual_full = np.hstack((y.reshape(-1, 1), tail_features[:, 1:]))

        if not np.isfinite(predicted_full).all():
            raise ValueError("❌ predicted_full contains invalid values.")
        if not np.isfinite(actual_full).all():
            raise ValueError("❌ actual_full contains invalid values.")

        # === Inverse Transform ===
        predicted_prices = scaler.inverse_transform(predicted_full)[:, 0]
        actual_prices = scaler.inverse_transform(actual_full)[:, 0]

        return predicted_prices, actual_prices

    except Exception as e:
        raise ValueError(f"Prediction failed: {e}")  # Ensure this propagates cleanly


def plot_predictions(predicted, actual):
    fig, ax = plt.subplots()
    ax.plot(actual, label="Actual Price", linewidth=2)
    ax.plot(predicted, label="Predicted Price", linestyle="--", linewidth=2)
    ax.set_title("Nickel Price Forecast")
    ax.set_xlabel("Time")
    ax.set_ylabel("LME Nickel Spot Price")
    ax.legend()
    st.pyplot(fig)
