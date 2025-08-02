import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    X = df.drop("Carbon_Emission_kg_per_day", axis=1)
    y = df["Carbon_Emission_kg_per_day"]

    numeric_features = ["Daily_Distance_Travelled_km", "Electricity_Usage_kWh", "Flights_Per_Month"]
    categorical_features = ["Transport_Mode", "Food_Type", "Home_Heating_Type"]

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    X_processed = preprocessor.fit_transform(X)

    print("✅ Preprocessing completed.")
    print("📐 Processed shape:", X_processed.shape)

    # Save processed features and labels together
    processed_df = pd.DataFrame(X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed)
    processed_df["Carbon_Emission_kg_per_day"] = y.reset_index(drop=True)

    os.makedirs("processed", exist_ok=True)
    processed_df.to_csv("processed/carbon_processed.csv", index=False)
    print("📁 Saved to processed/carbon_processed.csv")

    return X_processed, y

# ✅ Add this block so running the script directly works
if __name__ == "__main__":
    df = load_data("data/carbon_emission_dataset.csv")
    preprocess_data(df)
