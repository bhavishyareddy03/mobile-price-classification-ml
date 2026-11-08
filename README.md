# Mobile Price Classification - Machine Learning Assignment 2

## a. Problem Statement
The objective of this project is to build and evaluate multiple supervised machine learning models to classify mobile phones into four distinct price tiers (`0: Low Cost`, `1: Medium Cost`, `2: High Cost`, `3: Very High Cost`) based on their hardware and hardware-related performance specifications.

## b. Dataset Description
- **Source:** Kaggle Mobile Price Classification dataset
- **Instances:** 2,000 mobile device records
- **Features (20 input attributes):** `battery_power`, `blue`, `clock_speed`, `dual_sim`, `fc`, `four_g`, `int_memory`, `m_dep`, `mobile_wt`, `n_cores`, `pc`, `px_height`, `px_width`, `ram`, `sc_h`, `sc_w`, `talk_time`, `three_g`, `touch_screen`, `wifi`
- **Target Variable:** `price_range` (Multiclass: 0, 1, 2, 3)

## c. Repository & Deployment Links
- **GitHub Repository Link:** [https://github.com/bhavishyareddy03/mobile-price-classification-ml](https://github.com/bhavishyareddy03/mobile-price-classification-ml)
- **Live Streamlit App Link:** [https://mobile-price-classification-ml-04mgwk5roqwpudcmf9kl47.streamlit.app/](https://mobile-price-classification-ml-04mgwk5roqwpudcmf9kl47.streamlit.app/)

## d. Model Comparison Table
The models were trained on 80% of the dataset and evaluated on the remaining 20% stratified holdout test set (400 samples).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.9600 | 0.9950 | 0.9603 | 0.9600 | 0.9600 | 0.9468 |
| Decision Tree | 0.8125 | 0.8924 | 0.8143 | 0.8125 | 0.8127 | 0.7771 |
| KNN | 0.5625 | 0.7991 | 0.5678 | 0.5625 | 0.5674 | 0.4174 |
| Naive Bayes | 0.8125 | 0.9500 | 0.8113 | 0.8125 | 0.8125 | 0.7498 |
| Support Vector Machine | 0.9625 | 0.9985 | 0.9628 | 0.9625 | 0.9625 | 0.9500 |

## Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved excellent classification performance (96.00% Accuracy, 0.9950 AUC, 0.9468 MCC). The mobile price tiers exhibit an orderly, linear boundary with key driver features (especially RAM, battery capacity, and screen resolution), allowing the multinomial logit model to separate classes with minimal overlap. |
| **Decision Tree** | Delivered 81.25% Accuracy and an MCC of 0.7771. While decision trees effectively identify non-linear feature splits, single unpruned/shallow trees are prone to variance and struggle to establish the smooth, multi-attribute decision boundaries required across continuous numeric hardware metrics. |
| **KNN** | Recorded the lowest performance across all evaluation metrics (56.25% Accuracy, 0.7991 AUC, 0.4174 MCC). KNN suffers heavily from the curse of dimensionality across 20 features, where Euclidean distance metrics become diluted and less discriminative without targeted feature reduction or weighted distances. |
| **Naive Bayes** | Provided a solid baseline with 81.25% Accuracy and 0.9500 AUC. The Gaussian model captures class-conditional distributions reasonably well, though the conditional independence assumption causes slight probability calibration distortion due to natural correlations among camera resolutions (`fc`, `pc`) and screen dimensions (`px_height`, `px_width`). |
| **Support Vector Machine** | Achieved the highest performance across all six metrics (96.25% Accuracy, 0.9985 AUC, 0.9628 Precision, 0.9625 Recall, 0.9625 F1, and 0.9500 MCC). Maximizing the geometric margin over the standardized 20-dimensional feature space provided superior generalization and robust class separation. |
| **Overall Winner** | **Support Vector Machine (SVM)** is the clear overall winner, demonstrating the highest accuracy (96.25%), near-perfect discrimination (0.9985 AUC), and the strongest correlation across all multiclass categories (0.9500 MCC). |
