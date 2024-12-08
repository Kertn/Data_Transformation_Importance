# The Importance of Data Transformation - Model Performance Analysis using Kaggle Competition: Loan Approval Prediction
## Project Overview

This project demonstrates the critical importance of data transformation in machine learning modeling through an interactive and visual exploration. Using the dataset provided in Kaggle Playground Series - Season 4, Episode 10, the task is to predict loan approval. The project leverages advanced techniques to enhance model performance and explores the impact of various data transformation strategies.

The end result is an interactive graph that visualizes the outcomes of multiple experiments, capturing the results of 3072 trained models—each representing a unique combination of techniques. It enables users to select different advanced techniques and observe their effects on model training and evaluation metrics such as loss, accuracy, ROC AUC along with t-SNE and Confusion Matrix visualizations.

To optimize performance and manage storage efficiently, t-SNE and Confusion Matrix graphs were saved to Base64-encoded image strings

You can run interactive graph here: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Kertn/Data_Transformation_Importance/main?filepath=Iteractive_plot.ipynb)

## Exploratory Data Analysis (EDA)

In the EDA notebook (Iteractive_plot_EDA.ipynb), the following techniques were applied:

* **Data Overview:** Inspected missing values, summary statistics, and feature distributions.
* **Target Variable Analysis:** Visualized the distribution of loan approval outcomes.
* **Categorical Feature Analysis:** Explored categorical features against the target variable with count plots.
* **Numerical Feature Analysis:** Used pairplots and heatmaps to analyze relationships between numerical features and the target.
* **Outlier Detection:** Identified anomalies with box plots and Isolation Forest.

The notebook includes plots for:

* Distribution of the target variable (loan_status).
* Categorical feature distributions segmented by loan status.
* Pairwise relationships and correlations between numerical features.
* Visualizations of outliers and feature distributions.


## Techniques Applied

### 1. Outlier Detection Techniques

* **Isolation Forest:** Identifies anomalies in the dataset by building an ensemble of decision trees. 
* **Interquartile Range (IQR):** Detects and removes data points lying beyond the interquartile range.

### 2. Data Transformation and Scaling Techniques

* **Box-Cox Transformation:** Stabilizes variance and normalizes distributions.
* **Robust Scaler:** Scales features using statistics that are robust to outliers.
* **Normalization:** Converts feature values to a range between 0 and 1.
* **Standardization:** Scales data to have zero mean and unit variance.

### 3. Resampling Techniques

* **SMOTENC:** Synthetic Minority Over-sampling Technique for Nominal and Continuous features. It addresses class imbalance by generating synthetic samples.

### 4. Encoding Techniques

* **One-Hot Encoding:** Transforms categorical variables into binary vectors.
* **Label Encoding:** Assigns a unique integer to each category.

### 5. Model Regularization and Normalization Techniques

* **Dropout:** Prevents overfitting by randomly deactivating a subset of neurons during training.
* **Batch Normalization:** Normalizes layer inputs to improve training stability.

## Results

