# Project Plan

- [Project Instructions](project-overview.md)

## Team
 - Michael Bolens
 - Jeff Flachman
 - Ava Lee
 - Elia Porter


 ## Potential Datasets

- These are listed separately in [datasets](data_sets.md)

## Project Overview

### Executive Summary


### Selected Dataset


### Approach

### DataCleaning 

- Jeff will write



# What can we write on:

- Selected features
    - [data_cleaning.md](data_cleaning.md)
    - Printed selected features in [diabetes_features.md](diabetes_features.md)

    For Analysis you can look at [analysis.md](analysis.md)
    
### Data Analysis Readme: A Comprehensive Guide

This document provides a detailed walkthrough of the data cleaning and analysis process for the CDC data from 2021. The analysis focuses on several key steps, including handling unbalanced data, evaluating overfitting, and improving model performance through hyperparameter tuning. The analysis uses Python and various machine learning libraries to achieve these objectives.

#### Data Cleaning and Initial Exploration

1. **Loading the Data:**
   - The data is read from a parquet file using `pandas`.

2. **Exploratory Data Analysis:**
   - Initial exploration includes checking the data structure and basic statistics to understand the features and their distributions.

3. **Feature Selection:**
   - Key features relevant to diabetes analysis are selected. These features include general health, days health not good, mental health, primary insurance source, personal provider, years since last checkup, exercise, high blood pressure, cholesterol check, high cholesterol, heart disease, stroke, depressive disorder, kidney disease, marital status, education level, home ownership, employment, income level, weight, hearing, sight, difficulty walking, flu shot, race, sex, age, weight in kilos, body mass index (BMI), and several others.

#### Handling Unbalanced Data

1. **Balancing the Dataset:**
   - To address the issue of unbalanced data, undersampling is employed. The majority class (non-diabetic cases) is downsampled to match the size of the minority class (diabetic cases), resulting in a balanced dataset.

#### Evaluating Overfitting

1. **Model Training and Evaluation:**
   - Models are trained on the balanced dataset, and their performance is evaluated to check for overfitting. Key metrics include accuracy, precision, recall, and F1 score.
   - Cross-validation is used to evaluate model performance more robustly.

2. **Reviewing Scores:**
   - Scores from the cross-validation are reviewed to determine if balancing the data helped reduce overfitting. The comparison between the original and balanced datasets' performance indicates the effectiveness of undersampling.

#### Hyperparameter Tuning

1. **Grid Search CV:**
   - A Grid Search is performed to find the best hyperparameters for the model. This involves testing all possible combinations of specified hyperparameters and identifying the configuration that yields the best performance.

2. **Randomized Search CV:**
   - A Randomized Search is also conducted to explore a broader range of hyperparameters. Unlike Grid Search, which tests all possible combinations, Randomized Search samples a fixed number of parameter settings from the specified ranges, making it more efficient for large parameter spaces.

3. **Selecting the Best Model:**
   - The best model is selected based on the highest cross-validation score. The final parameters and their corresponding scores are documented.

#### Final Parameter Settings and Score

1. **Best Model and Parameters:**
   - The best model, identified through hyperparameter tuning, shows a significant improvement in performance metrics. The final parameters and scores reflect the optimized model's ability to predict diabetes with higher accuracy and reliability.

2. **Conclusion:**
   - The data cleaning and analysis process results in a well-balanced, high-performing model. The balancing of the dataset and hyperparameter tuning significantly contribute to reducing overfitting and enhancing the model's predictive power.

### Analysis Summary

- **Raw Data Scores:**
  - Accuracy: `0.75`
  - Precision: `0.70`
  - Recall: `0.65`
  - F1 Score: `0.67`

- **Balanced Data Scores:**
  - Accuracy: `0.78`
  - Precision: `0.75`
  - Recall: `0.72`
  - F1 Score: `0.73`

- **Best Model Parameters (Grid Search):**
  - Param1: `value1`
  - Param2: `value2`

- **Best Model Score:**
  - Accuracy: `0.80`

##Assuming the metrics are stored in the variables as mentioned, update your Readme with the actual values (NEEDS UPDATING)

### References
- CDC Diabetes Data: [CDC Data and Research](https://www.cdc.gov/diabetes/php/data-research/index.html)

This comprehensive guide ensures a clear understanding of the data analysis process, from initial cleaning to model optimization, and highlights the improvements achieved through these methods. The placeholders for scores should be replaced with the actual results obtained from your analysis.





- The Plan
    - Run models on basic data
    - 