import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

def train():
    os.makedirs("model", exist_ok=True)
    
    # 1. Load data
    df = pd.read_csv("dataset.csv")
    print(f"Loaded dataset successfully. Shape: {df.shape}")
    
    # Target column is 'price_range'
    target_col = 'price_range' if 'price_range' in df.columns else df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 2. Train-test split (80-20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Save the test data with ground truth for Streamlit evaluation
    test_df = pd.concat([X_test, y_test], axis=1)
    test_df.to_csv("test_data.csv", index=False)
    print("Saved test_data.csv with shape:", test_df.shape)
    
    # 3. Fit feature scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "model/scaler.joblib")
    
    # 4. Initialize the 5 models
    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=42),
            "model/logistic_regression.joblib",
            True
        ),
        "Decision Tree": (
            DecisionTreeClassifier(max_depth=7, random_state=42),
            "model/decision_tree.joblib",
            False
        ),
        "KNN": (
            KNeighborsClassifier(n_neighbors=9),
            "model/knn.joblib",
            True
        ),
        "Naive Bayes": (
            GaussianNB(),
            "model/naive_bayes.joblib",
            True
        ),
        "Support Vector Machine": (
            SVC(probability=True, kernel='linear', random_state=42),
            "model/svm.joblib",
            True
        )
    }
    
    results = []
    
    print("\n--- Training Models and Calculating Evaluation Metrics ---")
    for name, (model, save_path, use_scaled) in models.items():
        X_tr = X_train_scaled if use_scaled else X_train
        X_te = X_test_scaled if use_scaled else X_test
        
        # Fit model
        model.fit(X_tr, y_train)
        joblib.dump(model, save_path)
        
        # Predictions & Probabilities
        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te)
        
        # Multiclass metrics calculation
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        results.append({
            "ML Model Name": name,
            "Accuracy": round(acc, 4),
            "AUC": round(auc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "MCC": round(mcc, 4)
        })
        print(f"Completed {name} -> Acc: {acc:.4f}, AUC: {auc:.4f}")
    
    # 5. Output Markdown Table for README.md
    metrics_df = pd.DataFrame(results)
    print("\n================== README.md Comparison Table ==================")
    print(metrics_df.to_markdown(index=False))
    print("=================================================================\n")

if __name__ == "__main__":
    train()
    