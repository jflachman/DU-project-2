

## Data Analysis Notes:

- Data Cleaning see [1_data_cleaning.ipynb](1_data_cleaning.ipynb)

    - Data Exploration

        | Description | Code |
        |-------------|------|
        | Info on the dataframe | df.info() |
        | Describe the whole dataset | df.describe() |
        | counts of 'y' | df.value_counts() |
        | Percentage of rows missing values in each column | df.isna().sum()/len(df) |
        | Describe the other columns in the rows with missing values | df.loc[df['backers_count'].isna()].describe() |
        | Plot histogram of column values for 'pdays' | df.loc[df['pdays'].isna()].describe() |
        | # describe null data in 'pdays' | df.loc[df['pdays'].isna()].describe() |
    ---

    - Feature Engineering (14.2)
        - Data Leakage
            - Data leakage occurs when a model is trained using information that won’t be available when making predictions. The model will lack performance in production.

        - Imputation (Handle Missing Data)
            - Create functions that calculate/define values for missing data
            - Domain knowledge is helpful for effective Imputation
        - Encoding Categorical Data: involves the transformation of categorical variables from strings into numbers.
            - Get_dummies
            - OneHotEncoder
                - Nominal variables don’t have an inherent order. They are simply categories that can be distinguished from each other.
            - OrdinalEncoder
                - Ordinal variables have an inherent order and can be ranked from highest to lowest or vice versa.

        - Other commands

            | Description | Code |
            |-------------|------|
            | Drop all non-numeric columns | df_clean = df.select_dtypes(include='number') |
            |  |  |
            |  |  |
            |  |  |
            |  |  |
            ---        
    - Feature Selection
        - Select features that may provide predictive value to the target

- Data Analysis


- model selection
    - [Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
        - from sklearn.linear_model import LogisticRegression
    - [Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)
        - from sklearn.svm import SVC
        - from sklearn.svm import SVR
    - [Nearest Neighbors](https://scikit-learn.org/stable/modules/neighbors.html)
        - from sklearn.neighbors import KNeighborsClassifier
        - from sklearn.neighbors import KNeighborsRegressor
    - [Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
        - from sklearn import tree ()
            - tree.DecisionTreeClassifier()
    - [Ensembles](https://scikit-learn.org/stable/modules/ensemble.html)
        - Classifiers:
            - from sklearn.ensemble import RandomForestClassifier
            - from sklearn.ensemble import ExtraTreesClassifier
            - from sklearn.ensemble import GradientBoostingClassifier
            - from sklearn.ensemble import AdaBoostClassifier
        - Regressors:
            - from sklearn.ensemble import RandomForestRegressor
            - from sklearn.ensemble import ExtraTreesRegressor
            - from sklearn.ensemble import AdaBoostRegressor
- model training
    - model = xyz()
        - model

- model evaluation (14.1)
    - NOTE: Misinterpretation can occur when a misunderstanding of a score or metric leads to invalid conclusions.
    - Overfitting: When a model fits too closely to training data
    - Underfitting: When a model fails to capture meaningful relationships in the data
    - Metrics:
        - Accuracy measures the number of correct predictions that a model makes as a percentage of total predictions.
        - Confusion Matrix is used to calculate the confidence matrix for the model and the predictions in the confusion matrix.
        - Accuracy measures the number of correct predictions that a model makes as a percentage of total predictions.  Accuracy can be calculated from the confusion matrix.
        - Sensitivity Matrix: Using sensitivity as the main unit of measurement allows you to find out how many of the actually true data points were identified correctly.
        - Specificity Matrix: Using specificity as the main unit of measurement allows you to find out how many of the actually false data points were identified correctly.
        - Precision Matrix: Precision identifies how many of the predicted true results were actually true.
        - *** Classification Report: 
            - F1 score balances sensitivity and precision.
            - Balanced accuracy measures the accuracy of each class, then averages the results.
        - ROC curve: Visualizes the true positive and false positive rate of predictions using a range of decision thresholds
        - AUC-ROC: A calculation of the area under the ROC curve, giving a performance metric between 0 and 1
            - Decision threshold: The decimal value at which a model switches from predicting a 0 to predicting a 1
  

- model optimization (14.3)
    - Hyperparameter retuning
        - GridSearchCV
        - RandomizedSearchCV
    - Resampling methods for Imbalanced Data
        - Random Oversampling
        - Random Undersampling
        - Cluster Centroids
        - Synthetic Minority Oversampling Technique (SMOTE)
        - SMOTE and Edited Nearest Neighbors (SMOTEENN)


















- correlation work in features (inter-corrlation)
- handle unbalanced data
    - split the data into equal size for has and does not have diabetes ()
- automate the model
- scaling
- optimization
    - ????????

- Encoding
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

- sklearn - Model Evaluation
    - [Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
    - [Tuning the hyper-parameters of an estimator](https://scikit-learn.org/stable/modules/grid_search.html)
    - [Tuning the decision threshold for class prediction](https://scikit-learn.org/stable/modules/classification_threshold.html)
    - [Metrics and scoring: quantifying the quality of predictions](https://scikit-learn.org/stable/modules/model_evaluation.html)
        - accuracy_score
        - balanced_accuracy_score
        - average_precision_score
        - f1_score
        - precision_score
    - [Validation curves: plotting scores to evaluate models)

