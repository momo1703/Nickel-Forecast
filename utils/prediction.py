import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def predict_prices(model, X, y, scaler, tail_features):
    predictions = model.predict(X)
    predicted_prices = scaler.inverse_transform(
        np.hstack((predictions, tail_features[:, 1:]))
    )[:, 0]

    actual_prices = scaler.inverse_transform(
        np.hstack((y.reshape(-1, 1), tail_features[:, 1:]))
    )[:, 0]

    return predicted_prices, actual_prices

def plot_predictions(predicted, actual):
    fig, ax = plt.subplots()
    ax.plot(actual, label="Actual Price")
    ax.plot(predicted, label="Predicted Price")
    ax.set_title("Nickel Price Forecast")
    ax.legend()
    st.pyplot(fig)