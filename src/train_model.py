import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

def train_main():
    # Ensure artifacts directory exists
    os.makedirs("artifacts", exist_ok=True)

    # Load the processed data
    df = pd.read_csv("processed/carbon_processed.csv")

    # Target and features
    y = df["Carbon_Emission_kg_per_day"]
    X = df.drop("Carbon_Emission_kg_per_day", axis=1)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save model
    model_path = "artifacts/trained_model.pkl"
    joblib.dump(model, model_path)
    print(f"✅ Model training completed and saved to {model_path}")

    # ✅ Return a simple string for Airflow
    return f"✅ Model trained and saved to {model_path}"

# Optional: allow script to run directly
if __name__ == "__main__":
    train_main()
