import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

# Load the dataset
def train_model():
    print("Loading cleaned dataset...")
    df = pd.read_csv("data/heart_disease.csv")
    
    # 1. Preprocessing
    # Target in Cleveland dataset is 0 (healthy) or 1,2,3,4 (heart disease stages).
    # We convert it to binary classification: 0 (No Disease) and 1 (Disease).
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    # 2. Train/Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Initialize and Train XGBoost Classifier
    print("Training optimized XGBoost model...")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    # 4. Model Evaluation
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print("\n--- MODEL PERFORMANCE METRICS ---")
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print("---------------------------------\n")
    
    # 5. Save the trained model and dataset splits (needed for SHAP and LIME in UI)
    print("Saving model and training artifacts...")
    artifacts = {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }
    
    with open("heart_model.pkl", "wb") as f:
        pickle.dump(artifacts, f)
        
    print("Saved model artifacts to 'heart_model.pkl' successfully.")

if __name__ == "__main__":
    train_model()