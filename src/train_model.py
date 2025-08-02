import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

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
joblib.dump(model, "artifacts/trained_model.pkl")
print("✅ Model training completed and saved to artifacts/trained_model.pkl")
