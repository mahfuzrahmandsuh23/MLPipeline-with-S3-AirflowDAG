import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import load_data, preprocess_data


def evaluate_model():
    # Load and preprocess data
    data = load_data("data/carbon_emission_dataset.csv")
    X, y = preprocess_data(data)

    # Split into train/test (same logic as training step to keep consistency)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load trained model
    model = joblib.load("artifacts/trained_model.pkl")

    # Predict
    y_pred = model.predict(X_test)

    # Evaluation metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5

    # Print and optionally save results
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")

    # Optional: Save metrics to file
    metrics = {
        "r2_score": r2,
        "mae": mae,
        "mse": mse,
        "rmse": rmse
    }
    pd.DataFrame([metrics]).to_csv("artifacts/evaluation_metrics.csv", index=False)

if __name__ == "__main__":
    evaluate_model()
