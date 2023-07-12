# TelcoCustomerChurn

## Table of Contents
- [Introduction](#Introduction)
- [Data Source](#Data-Source)
- [Data Cleaning](#Data-Cleaning)
- [Exploratory Data Analyis](#Exploratory-Data-Analysis)
- [Feature Engineering](#Feature-Engineering)
- [Modeling](#Modeling)
- [Model Understanding](#Model-Understanding)

## Introduction
In this case, I will build a classification model using machine learning algorithms for predicting customer churn indicated by labels "Yes" or "No"

## Data Source
The dataset for this case is obtained from ![Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

## Data Cleaning
- Convert "TotalCharges" which contains object data type to float using pd.to_numeric with error parameter coerce.
- After converting "TotalCharges" data type, there are some missing values found in the column since I used coerce parameter to parse invalid data to NaN. The missing values were imputed using "TotalCharges" mean.

## Exploratory Data Analysis
- The distribution of the target (Churn column) is imbalanced due to the fact that retention comprises nearly 3/4 of the overall distributions.
- Using heatmap, the feature "MonthlyCharges" has a high positive Pearson correlation with the "Churn" column compared to other continuous columns.

## Feature Engineering
- Using one hot encoding to transform all categorical features except the target column.
- To prevent data leakage, I split the data into 80% training data and 20% test data before applying the MinMaxScaler to transform the continuous features. Additionally, I stratify the target label during the splitting process.

## Modeling
Four different models were applied to predict the label. Before that, we need to address that the dataset is imbalanced. To handle this, by utilizing the SMOTE technique, the model's performance could predict the minority class better. Furthermore, when building the model I used RandomizedSearchCV in order to find the best parameter before we actually fit and predict the data. 
Here are the four models:
- Logistic Regression
- Random Forest
- XGboost
- LightGBM

## Model Understanding
By a surprise, logistic regression gives the best performance in terms of roc auc score. So, we will break down the top defining features to detect churn customers. We found that Monthly Charges, tenure, Total Charges, and Internet Service are the top defining feature of the model.
