# src/data_ingestion.py

import pandas as pd
from pathlib import Path

def load_data(data_path: str = 'processed/carbon_processed.csv') -> pd.DataFrame:
    df = pd.read_csv(data_path)
    return df
