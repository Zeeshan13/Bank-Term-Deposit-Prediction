# Bank-Term-Deposit-Prediction


A data science project aimed at predicting a customer’s likelihood to subscribe to a bank term deposit based on demographic, campaign, and financial information. The project leverages machine learning models to identify critical factors influencing a customer's decision, allowing banks to optimize marketing strategies.

## Introduction
This project analyzes a bank marketing dataset to uncover insights into customer behavior and improve marketing campaign effectiveness. Machine learning models, including logistic regression, decision trees, and random forests, were applied to predict whether a customer is likely to subscribe to a term deposit.

## Dataset

+ **Source:** A Portuguese banking institution’s marketing campaign data
+ **Features:** Customer demographics, campaign interactions, subscription outcomes, and financial information.


# Project Structure

├── data                    # Dataset files

├── notebooks               # Jupyter notebooks for EDA and modeling

├── models                  # Trained model files

├── app.py                  # Streamlit app for interactive predictions

├── requirements.txt        # Required libraries for the project

└── README.md               # Project overview and instructions


# Data Preprocessing

+ **Handling Missing Values:** Ensured data integrity by checking and handling missing values.
+ **Outlier Detection:** Identified and managed outliers, especially in critical columns.
+ **Encoding:** Converted categorical features to numerical values using one-hot encoding.
+ **Scaling:** Standardized numerical features to improve model performance.
+ **Class Balancing:** Used SMOTE to address imbalanced classes in the target variable.


# Exploratory Data Analysis (EDA)
EDA involved examining correlations and patterns in demographic data such as job, marital status, and education, which were found to impact subscription likelihood. Key findings include:

+ Positive correlation between the number of contacts and the likelihood of subscription.
+ Higher subscription rates among certain job categories and education levels.

# Feature Engineering
Categorized features like age, duration, and balance to provide nuanced insights, enhancing the model’s predictive power by creating high-level representations of customer behavior.

# Modeling

Several models were trained and evaluated:
+ **Logistic Regression:** Baseline model with high recall, suitable for identifying potential subscribers.
+ **Random Forest:** Best model with the highest accuracy and F1 score, providing a balance of precision and recall.
+ **Decision Tree:** An interpretable model, useful for understanding decision rules but less accurate than Random Forest.


### Model Performance

| Model                     | Accuracy | Precision | Recall | F1 Score |
|---------------------------|----------|-----------|--------|----------|
| Logistic Regression       | 84.08%   | 41.88%    | 82.31% | 55.52%   |
| PCA + Logistic Regression | 80.42%   | 34.93%    | 78.16% | 48.29%   |
| Random Forest             | 90.01%   | 58.38%    | 60.04% | 59.20%   |
| Decision Tree             | 87.12%   | 45.71%    | 53.88% | 49.46%   |

# App Interface
An interactive Streamlit app provides users with a simple interface for inputting customer details and generating predictions about the likelihood of subscription. The app also displays model accuracy and feature importances.

