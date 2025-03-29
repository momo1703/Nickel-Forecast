import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def predict_prices(model, X, y, scaler, tail_features):
    try:
        # === Check for invalid values ===
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("❌ Input features (X) contain NaN or Inf values.")
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            raise ValueError("❌ Target values (y) contain NaN or Inf values.")
        if np.any(np.isnan(tail_features)) or np.any(np.isinf(tail_features)):
            raise ValueError("❌ Tail features contain NaN or Inf values.")

        # === Predict using the model ===
        predictions = model.predict(X)

        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            raise ValueError("❌ Model predictions contain NaN or Inf values.")

        # === Check shapes ===
        if predictions.shape[0] != tail_features.shape[0]:
            raise ValueError(f"❌ Shape mismatch: predictions ({predictions.shape[0]}) vs tail_features ({tail_features.shape[0]})")

        # === Combine for inverse scaling ===
        predicted_full = np.hstack((predictions, tail_features[:, 1:]))
        actual_full = np.hstack((y.reshape(-1, 1), tail_features[:, 1:]))

        # === Attempt inverse scaling ===
        predicted_prices = scaler.inverse_transform(predicted_full)[:, 0]
        actual_prices = scaler.inverse_transform(actual_full)[:, 0]

        return predicted_prices, actual_prices

    except Exception as e:
        raise ValueError(f"❌ Prediction pipeline error: {e}")

def plot_predictions(predicted, actual):
    fig, ax = plt.subplots()
    ax.plot(actual, label="Actual Price", linewidth=2)
    ax.plot(predicted, label="Predicted Price", linestyle="--", linewidth=2)
    ax.set_title("Nickel Price Forecast vs Actual")
    ax.set_xlabel("Time")
    ax.set_ylabel("Nickel Spot Price")
    ax.legend()
    st.pyplot(fig)
