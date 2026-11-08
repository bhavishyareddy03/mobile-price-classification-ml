import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(page_title="Mobile Price Classification ML Dashboard", layout="wide")

st.title("📱 Mobile Price Classification - Model Evaluation Dashboard")
st.markdown("Evaluate pre-trained machine learning classifiers on test data without retraining.")

# Sidebar: File Uploader & Model Selector
st.sidebar.header("Evaluation Controls")

uploaded_file = st.sidebar.file_uploader("Upload Test Data (CSV)", type=["csv"])

# Fallback to local test_data.csv if none uploaded
if uploaded_file is not None:
    df_test = pd.read_csv(uploaded_file)
    st.sidebar.success("Custom test dataset loaded.")
elif os.path.exists("test_data.csv"):
    df_test = pd.read_csv("test_data.csv")
    st.sidebar.info("Using default test_data.csv from repository.")
else:
    df_test = None
    st.warning("Please upload a test dataset CSV containing features and the target column.")

model_options = {
    "Logistic Regression": ("model/logistic_regression.joblib", True),
    "Decision Tree": ("model/decision_tree.joblib", False),
    "KNN": ("model/knn.joblib", True),
    "Naive Bayes": ("model/naive_bayes.joblib", True),
    "Support Vector Machine": ("model/svm.joblib", True)
}

selected_model_name = st.sidebar.selectbox("Choose Classification Model", list(model_options.keys()))

if df_test is not None:
    target_col = 'price_range' if 'price_range' in df_test.columns else df_test.columns[-1]
    X_eval = df_test.drop(columns=[target_col])
    y_eval = df_test[target_col]
    
    st.subheader(f"Current Evaluation: {selected_model_name}")
    st.write(f"Evaluating **{len(df_test)}** test samples across **{X_eval.shape[1]}** features.")

    # Load artifacts
    model_path, use_scaler = model_options[selected_model_name]
    
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        
        # Apply scaling if required
        if use_scaler and os.path.exists("model/scaler.joblib"):
            scaler = joblib.load("model/scaler.joblib")
            X_eval_processed = scaler.transform(X_eval)
        else:
            X_eval_processed = X_eval

        # Generate predictions
        y_pred = model.predict(X_eval_processed)
        y_proba = model.predict_proba(X_eval_processed)

        # Compute required evaluation metrics
        acc = accuracy_score(y_eval, y_pred)
        auc = roc_auc_score(y_eval, y_proba, multi_class='ovr', average='weighted')
        prec = precision_score(y_eval, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_eval, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_eval, y_pred, average='weighted', zero_division=0)
        mcc = matthews_corrcoef(y_eval, y_pred)

        # Display Metrics in Columns
        c1, c2, c3 = st.columns(3)
        c4, c5, c6 = st.columns(3)
        
        c1.metric("Accuracy", f"{acc:.4f}")
        c2.metric("AUC Score (OVR)", f"{auc:.4f}")
        c3.metric("Precision (Weighted)", f"{prec:.4f}")
        c4.metric("Recall (Weighted)", f"{rec:.4f}")
        c5.metric("F1 Score (Weighted)", f"{f1:.4f}")
        c6.metric("MCC Score", f"{mcc:.4f}")

        st.markdown("---")

        # Visualization Section
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_eval, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
            ax.set_xlabel("Predicted Class")
            ax.set_ylabel("True Class")
            st.pyplot(fig)

        with col_right:
            st.subheader("Classification Report")
            rep_dict = classification_report(y_eval, y_pred, output_dict=True, zero_division=0)
            rep_df = pd.DataFrame(rep_dict).transpose()
            st.dataframe(rep_df.style.format(precision=3))

    else:
        st.error(f"Model file `{model_path}` not found. Ensure `train.py` has run completely.")