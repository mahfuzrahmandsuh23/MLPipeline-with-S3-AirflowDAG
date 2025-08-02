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
    X, y = preprocess_data(data, return_arrays=True)  # ✅ Fix here

    # Train/test split (same as training)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load trained model
    model = joblib.load("artifacts/trained_model.pkl")

    # Predict and evaluate
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5

    # Print metrics
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")

    # Save to file
    metrics = {
        "r2_score": round(r2, 4),
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "rmse": round(rmse, 4)
    }
    os.makedirs("artifacts", exist_ok=True)
    pd.DataFrame([metrics]).to_csv("artifacts/evaluation_metrics.csv", index=False)

    # ✅ Safe return for Airflow task
    return f"✅ Evaluation complete. R²: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}"

if __name__ == "__main__":
    evaluate_model()
