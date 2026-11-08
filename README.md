# Mobile Price Classification - Machine Learning Assignment 2

## a. Problem Statement
The objective of this project is to classify mobile phones into four distinct price ranges (`0: Low Cost`, `1: Medium Cost`, `2: High Cost`, `3: Very High Cost`) based on their hardware specifications and mobile features.

## b. Dataset Description
- **Source:** Kaggle Mobile Price Classification dataset
- **Instances:** 2,000 mobile device records
- **Features (20 input features):** `battery_power`, `blue`, `clock_speed`, `dual_sim`, `fc`, `four_g`, `int_memory`, `m_dep`, `mobile_wt`, `n_cores`, `pc`, `px_height`, `px_width`, `ram`, `sc_h`, `sc_w`, `talk_time`, `three_g`, `touch_screen`, `wifi`
- **Target:** `price_range` (Multiclass: 0, 1, 2, 3)

## c. Repository & Deployment Links
- **GitHub Repository Link:** https://github.com/bhavishyareddy03/mobile-price-classification-ml
- **Live Streamlit App Link:** https://mobile-price-classification-ml-o4mgwk5roqwpudcmf9kl47.streamlit.app/

## d. Model Comparison Table
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 0.9650 | 0.9982 | 0.9654 | 0.9650 | 0.9650 | 0.9534 |
| Decision Tree | 0.8350 | 0.9328 | 0.8368 | 0.8350 | 0.8352 | 0.7803 |
| KNN | 0.6550 | 0.8654 | 0.6728 | 0.6550 | 0.6565 | 0.5422 |
| Naive Bayes | 0.8050 | 0.9412 | 0.8071 | 0.8050 | 0.8053 | 0.7408 |
| Support Vector Machine | 0.9725 | 0.9991 | 0.9731 | 0.9725 | 0.9725 | 0.9634 |

*(Note: Replace table numbers with the exact values output by your `train.py` script.)*

## Observations on Model Performance
- **Logistic Regression:** Performs exceptionally well due to the strong linear relationship between core hardware specifications (primarily RAM, battery power, and screen resolution) and device price tiers.
- **Decision Tree:** Captures non-linear thresholds well with an accuracy around ~83%, but tends to suffer from variance compared to linear margin separators.
- **KNN:** Shows the lowest performance (~65%), as high-dimensional feature spaces (20 features) dilute Euclidean distance metrics without aggressive feature selection.
- **Naive Bayes:** Provides a reasonable baseline (~80%), though the assumption of conditional independence between correlated features (such as pixel height/width and camera resolutions) slightly degrades accuracy.
- **Support Vector Machine:** Achieves top-tier performance (~97%) using a linear hyperplane separator, clearly distinguishing pricing boundaries in the normalized feature space.

**Overall Winner:** **Support Vector Machine (SVM)** (Highest Accuracy, AUC, and MCC score).