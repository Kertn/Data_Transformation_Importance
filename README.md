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

## Insights from Interactive Graphical Analysis

### SMOTENC

Using the SMOTENC method, we observe the following:

* **Confusion Matrix Observations:** Each increase in the SMOTENC parameter improves the prediction accuracy for the minority class. However, this comes at a significant cost to the prediction accuracy of the majority class.

* **Performance Metrics:** A consistent decline in accuracy and ROC AUC scores is noted. This is because the dataset is predominantly composed of majority class samples, and sacrificing its prediction capability for better minority class prediction negatively impacts the overall testing results.

* **t-SNE Visualization:** The t-SNE plot highlights the issue: the synthetic data lacks the same degree of structure or separability as the original data, which affects the model's performance.



### Box-Cox Transformation

* Among various techniques, the Box-Cox transformation consistently improves prediction quality.

* **Reason for Improvement:** This transformation likely makes the dataset more conducive to the model's learning process by improving the separability of certain data points.

### Robust Scaler

* Improvements are noticeable only when normalization or standardization methods are not applied.

### IQR and Isolation Forest

* **t-SNE Insights:** A slight improvement in clustering capabilities is observed, particularly for the yellow dots on the t-SNE plot. This indicates that removing outliers helped the model identify and isolate a specific group of samples sharing similar characteristics.

* **Performance Metrics:** Despite this improvement, accuracy and ROC AUC metrics slightly decline. This suggests that outlier detection techniques were potentially detrimental in this case due to the small dataset size and the relevance of the outliers.

### Batch Normalization (BatchNorm)

* Across all observed cases, BatchNorm consistently harms prediction quality.

* **Reason for Deterioration:** The heterogeneous nature of the data, as revealed by patterns in the pairplot (refer to Interactive_plot_EDA.ipynb), likely causes this decline.


### Standardization and Normalization

* **Standardization:** Outperforms normalization in most cases.

* **Reason for Preference:** The data distribution deviates from normality, making standardization more suitable.

### Dropout

* **Graphical Insights:** The validation and training lines on the plot align closely when dropout is applied, indicating effective regularization.

* **Model Performance:** A slight decline in model performance is observed, likely due to underfitting caused by the implementation of this technique.

## Conclusion

This study has provided an in-depth look at how different preprocessing and regularization techniques directly impact model learning in the context of a specific case study. Through detailed observations and interactive graphical analysis as well as EDA (Iteractive_plot_EDA.ipynb), we were able to identify key patterns and relationships in the data, shedding light on the effectiveness of these methods.

The results highlight the trade-offs involved in optimizing model performance. For instance, while techniques like Box-Cox transformation and SMOTENC address specific challenges like class imbalance and data structure, they may also introduce new issues, such as a loss of accuracy in the majority class. Similarly, while dropout effectively reduces overfitting, it may lead to slight underfitting if not carefully tuned.

Moreover, the role of data preprocessing methods, such as standardization and normalization, proved to be pivotal. Standardization consistently showed better outcomes, especially when dealing with data that is not normally distributed. Conversely, methods like BatchNorm or outlier detection techniques often resulted in performance degradation, underscoring the importance of understanding the dataset’s characteristics before applying these approaches.

Ultimately, this analysis emphasizes the importance of tailoring preprocessing and regularization strategies to the specific dataset and problem at hand. The insights derived from interactive graphical tools serve as valuable guidance for making informed decisions, enabling a more nuanced and effective approach to enhancing model performance.
