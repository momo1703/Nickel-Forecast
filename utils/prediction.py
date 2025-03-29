def predict_prices(model, X, y, scaler, tail_features):
    try:
        # === Predict using the model ===
        predictions = model.predict(X)

        # === Check for invalid values ===
        for name, array in {
            "X": X,
            "y": y,
            "tail_features": tail_features,
            "predictions": predictions
        }.items():
            if np.any(np.isnan(array)):
                raise ValueError(f"❌ {name} contains NaN values.")
            if np.any(np.isinf(array)):
                raise ValueError(f"❌ {name} contains Inf values.")

        # === Check shape compatibility ===
        if predictions.shape[0] != tail_features.shape[0]:
            raise ValueError(
                f"❌ Shape mismatch: predictions ({predictions.shape}) vs tail_features ({tail_features.shape})"
            )

        # === Prepare data for inverse transform ===
        predicted_full = np.hstack((predictions, tail_features[:, 1:]))
        actual_full = np.hstack((y.reshape(-1, 1), tail_features[:, 1:]))

        # === Final safety before inverse transform ===
        if np.any(np.isnan(predicted_full)) or np.any(np.isinf(predicted_full)):
            raise ValueError("❌ predicted_full has NaN or Inf before inverse_transform.")
        if np.any(np.isnan(actual_full)) or np.any(np.isinf(actual_full)):
            raise ValueError("❌ actual_full has NaN or Inf before inverse_transform.")

        # === Run inverse transform ===
        predicted_prices = scaler.inverse_transform(predicted_full)[:, 0]
        actual_prices = scaler.inverse_transform(actual_full)[:, 0]

        return predicted_prices, actual_prices

    except Exception as e:
        raise ValueError(f"❌ Prediction pipeline error: {e}")
