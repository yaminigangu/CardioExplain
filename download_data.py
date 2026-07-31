import urllib.request
import os
import pandas as pd

DATA_DIR = "data"
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# UCI Cleveland Heart Disease dataset column names
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

def download_dataset():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    dest_path = os.path.join(DATA_DIR, "heart_disease.csv")
    print("Downloading Cleveland Heart Disease dataset from UCI...")
    
    try:
        # Download raw data
        raw_path = os.path.join(DATA_DIR, "raw_data.data")
        urllib.request.urlretrieve(DATA_URL, raw_path)
        
        # Load and clean dataset (handling missing values marked as '?')
        df = pd.read_csv(raw_path, names=COLUMNS, na_values="?")
        
        # Fill missing values with median (common research standard)
        df.fillna(df.median(), inplace=True)
        
        # Save cleaned dataset as CSV
        df.to_csv(dest_path, index=False)
        print(f"Dataset successfully saved to: {dest_path}")
        
        # Remove temporary raw file
        if os.path.exists(raw_path):
            os.remove(raw_path)
            
    except Exception as e:
        print(f"Error downloading data: {e}")

if __name__ == "__main__":
    download_dataset()