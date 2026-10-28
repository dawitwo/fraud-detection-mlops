# src/preprocessing.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from src.config import *

def preprocess_data():
    df = pd.read_csv(RAW_DATA_PATH)
    
    # 1. Feature Engineering: Convert seconds to hour of day
    df['hour_of_day'] = (df['Time'] / 3600) % 24
    df = df.drop(columns=['Time'])
    
    # 2. Scale Amount using RobustScaler
    scaler = RobustScaler()
    df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
    df = df.drop(columns=['Amount'])
    
    # Save the scaler for inference
    joblib.dump(scaler, SCALER_PATH)
    
    # 3. Stratified split to preserve fraud ratio
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df['Class'], random_state=RANDOM_STATE
    )
    
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

if __name__ == "__main__":
    preprocess_data()