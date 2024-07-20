import pandas as pd
from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor

# from sklearn.svm import SVR
from sklearn.svm import SVC 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree

from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error

#from sklearn.model_selection import train_test_split

import sklearn.model_selection
from ml_clean_feature import clean_features_list

import time
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
# ----------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------


# --------------------------------------
# ------- train_test_split(df, target)
# ---------- 
# --------------------------------------
def train_test_data(df, target):
    # Create X and y datasets
    X = df.copy().drop(columns=[target])
    y = df[target]

    # Create Train Test Split
    data = sklearn.model_selection.train_test_split(X, y)
    return data, True


# --------------------------------------
# ------- run_classification_models(data, k_value)
# ---------- 
# --------------------------------------
def run_classification_models(data, k_value):
#def run_classification_models_test(data, k_value):
    start_time = time.time()
    performance_summary = []

    print(f"\n*************************************************************************************")
    print(f"********* Classification Models")
    print(f"*************************************************************************************")


#    test_model(SVC(kernel='linear'), data)
    performance_summary.append(test_model(KNeighborsClassifier(n_neighbors=k_value), data))
    performance_summary.append(test_model(tree.DecisionTreeClassifier(), data))
    performance_summary.append(test_model(RandomForestClassifier(), data))
    performance_summary.append(test_model(ExtraTreesClassifier(random_state=1), data))
    performance_summary.append(test_model(GradientBoostingClassifier(random_state=1), data))
    performance_summary.append(test_model(AdaBoostClassifier(random_state=1), data))
    performance_summary.append(test_model(LogisticRegression(), data))

    perf_df = pd.concat(performance_summary, ignore_index=True)

    print(f"\n************************************************")
    print(f"****** Classification Models Performance Summary")
    print(f"************************************************")

    print(perf_df)

    print(f"\n*************************************************************************************")
    print(f"********* Classification Models  **************** Completed: Execution Time %s seconds:" % (time.time() - start_time))
    print(f"*************************************************************************************")

    return perf_df


# --------------------------------------
# ------- run_classification_models_test(data, k_value)
# ---------- Comments out most of the models
# --------------------------------------
#def run_classification_models(data, k_value):
def run_classification_models_test(data, k_value):
    start_time = time.time()
    performance_summary = []

    print(f"\n*************************************************************************************")
    print(f"********* Classification Models")
    print(f"*************************************************************************************")


#    test_model(SVC(kernel='linear'), data)
#    performance_summary.append(test_model(KNeighborsClassifier(n_neighbors=k_value), data))
    performance_summary.append(test_model(tree.DecisionTreeClassifier(), data))
    # performance_summary.append(test_model(RandomForestClassifier(), data))
    # performance_summary.append(test_model(ExtraTreesClassifier(random_state=1), data))
    # performance_summary.append(test_model(GradientBoostingClassifier(random_state=1), data))
    # performance_summary.append(test_model(AdaBoostClassifier(random_state=1), data))
    performance_summary.append(test_model(LogisticRegression(), data))

    perf_df = pd.concat(performance_summary, ignore_index=True)

    print(f"\n************************************************")
    print(f"****** Classification Models Performance Summary")
    print(f"************************************************")

    print(perf_df)

    print(f"\n*************************************************************************************")
    print(f"********* Classification Models  **************** Completed: Execution Time %s seconds:" % (time.time() - start_time))
    print(f"*************************************************************************************")

    return perf_df

# --------------------------------------
# ------- run_regression_models(data, k_value)
# ---------- 
# --------------------------------------
def run_regression_models(data, k_value):
    start_time = time.time()

    print(f"\n*************************************************************************************")
    print(f"********* Regression Models")
    print(f"*************************************************************************************")

    test_model(LinearRegression(), data)
    test_model(KNeighborsRegressor(), data)
    test_model(RandomForestRegressor(), data)
    test_model(ExtraTreesRegressor(), data)
    test_model(AdaBoostRegressor(), data)
    test_model(SVR(C=1.0, epsilon=0.2), data)

    print(f"\n*************************************************************************************")
    print(f"********* Regression Models  **************** Completed: Execution Time %s seconds:" % (time.time() - start_time))
    print(f"*************************************************************************************")


# --------------------------------------
# ------- Test_model()
# ---------- 
# --------------------------------------
def test_model(model, data):
    start_time = time.time()

    print(f"\n-----------------------------------------------------------------------------------------")
    print(f"-----------------------------------------------------------------------------------------")
    print(f'Model: {type(model).__name__}')
    print(f"-----------------------------------------------------------------------------------------")
    print(f"-----------------------------------------------------------------------------------------")

    X_train, X_test, y_train, y_test = data

    # Train the model
    model = model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train = [ X_train, y_train, y_train_pred]
    test  = [ X_test, y_test, y_test_pred]

    train_performance = model_performance_metrics(model, train, "Train" )
    test_performance  = model_performance_metrics(model, test, "Test" )

    performance_df = pd.DataFrame([train_performance, test_performance])

    print(f"-------------------------------------------------------")
    print(f"{type(model).__name__} Performance Summary:")
    print(f"-------------------------------------------------------")
    print(f"{performance_df}")

    print(f"-----------------------------------------------------------------------------------------")
    print(f"Model: {type(model).__name__}  --- Completed: Execution Time %s seconds:" % (time.time() - start_time))
    print(f"-----------------------------------------------------------------------------------------")

    return performance_df


# --------------------------------------
# ------- Random Search
# ---------- 
# --------------------------------------
# optimize_report_params = {
#     param_distributions: param_distributions,
#     optimize_path: config.optimize_path,
#     dataset: dataset_label,
#     scoring: 'accuracy',
#     verbose: 0,
#     print_results: True
#     }
def run_random_search(model, o_params, data):
    import os
    import pickle

    # Generate the Performance Report and send prints to osc.stdout
    with OutStreamCapture_stdout() as osc:
        perf_df = random_search(model, o_params['param_distributions'], data, o_params['scoring'], o_params['verbose'])
    
    optimize_report_file    = o_params['optimize_path'] + o_params['dataset'] + '_' + type(model).__name__ + '.txt'

    # osc.stdout contains the details of the performance report
    # write the performance report to the optimize_report_file
    with open(optimize_report_file, "w") as file:
        file.write(osc.stdout)

    performance_report    = o_params['optimization_report']

#----------------------------------------------------------------------------------------------------

    # perf_df is a dataframe of performance statistics
    dataset = o_params['dataset']
    # Add the dataset label as the first column in perf_df
    dataset_column = pd.Series([dataset] * len(perf_df), name=dataset)
    perf_df.insert(0, 'dataset', dataset_column)

    analysis_perf_summary = { 'dataset_size': list(perf_df.shape), 'report': perf_df}

    # Performance_report is a file containing all the performance summary statistics
    if os.path.exists(performance_report):
#        print(f"The file {performance_report} exists.")
        # Load Performance Report
        with open(performance_report, 'rb') as file: perf_report = pickle.load(file)
    else:
#        print(f"The file {performance_report} does not exist.")
        perf_report = {}
    
    per_report_key = 'RandomizedSearchCV_' + dataset + '_' + type(model).__name__
    perf_report[per_report_key] = analysis_perf_summary

    # Save Performance Report
    with open(performance_report, 'wb') as file: pickle.dump(perf_report, file)

#----------------------------------------------------------------------------------------------------
    # Display the performance report here:
    if o_params['print_results']:
        print(osc.stdout)

    return perf_df

# --------------------------------------
# ------- Random Search
# ---------- 
# --------------------------------------

def random_search(model, param_dist, data, scoring='accuracy', verbose=0):
    from sklearn.model_selection import RandomizedSearchCV

    import numpy as np
    # Classification Options:
    #   'accuracy', 'balanced_accuracy', 'top_k_accuracy', 'average_precision', 
    #   'neg_brier_score', 'f1', 'f1_micro', 'f1_macro', 'f1_weighted', 'f1_samples', 
    #   'neg_log_loss', 'precision' etc., 'recall' etc., 'jaccard' etc., 'roc_auc', 
    #   'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted', 
    #   'd2_log_loss_score', 
    # Clustering Options:
    #   'adjusted_mutual_info_score', 'adjusted_rand_score', 'completeness_score', 
    #   'fowlkes_mallows_score', 'homogeneity_score', 'mutual_info_score', 
    #   'normalized_mutual_info_score', 'rand_score', 'v_measure_score',
    # Regresson Options:
    #   'explained_variance', 'max_error', 'neg_mean_absolute_error', 'neg_mean_squared_error', 
    #   'neg_root_mean_squared_error', 'neg_mean_squared_log_error', 'neg_root_mean_squared_log_error', 
    #   'neg_median_absolute_error', 'r2', 'neg_mean_poisson_deviance', 'neg_mean_gamma_deviance', 
    #   'neg_mean_absolute_percentage_error', 'd2_absolute_error_score', , 

    # verbose =0 : No display
    # verbose >1 : the computation time for each fold and parameter candidate is displayed
    # verbose >2 : the score is also displayed;
    # verbose >3 : the fold and candidate parameter indexes are also displayed together with the starting time of the computation.

    # ----------------------------------------------------- Optimization
    # Expand Data
    X_train, X_test, y_train, y_test = data

    # Initialize RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=model, 
        param_distributions=param_dist, 
        n_iter=50,         # Number of parameter settings that are sampled
        cv=5,               # 5-fold cross-validation
        verbose=verbose,          # Verbosity mode
        scoring=scoring,    # Scoring Mode
        random_state=42,    # Seed for reproducibility
        n_jobs=-1 )

    # Fit the model to the training data using RandomizedSearchCV
    random_search.fit(X_train, y_train)

    # Get the best parameters and best model
    best_params_random = random_search.best_params_
    best_model_random = random_search.best_estimator_

    print(f"Best Parameters: Model: {type(model).__name__}: Parameters: {best_params_random}")

    # # Make predictions on the test set using the best model
    y_pred_best_random = best_model_random.predict(X_test)

    test_data = [ X_test, y_test, y_pred_best_random]

    random_search_performance = model_performance_metrics(best_model_random, test_data, 'optimized' )
    random_search_performance['Optimizer'] = 'RandomizedSearchCV'
    random_search_performance['best_parameters'] = best_params_random
    # ----------------------------------------------------- Optimization

    # X_train, X_test, y_train, y_test = data should be unchanged

    # Train the model
    model = model.fit(X_train, y_train)

#    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    test_data = [ X_test, y_test, y_test_pred]

    original_performance = model_performance_metrics(model, test_data, "un-optimized" )
    original_performance['Optimizer'] = 'RandomizedSearchCV'
    original_performance['best_parameters'] = 'None'
    # ----------------------------------------------------- Pass back performance metrics
    
    performance_df = pd.DataFrame([original_performance, random_search_performance])

    return performance_df


# --------------------------------------
# ------- model_performance_metrics
# ---------- 
# --------------------------------------
def model_performance_metrics(model, data, datalabel):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.metrics import roc_auc_score    
    import numpy as np

    # Expand the model data
    X, y, y_pred = data

    # -------------------------------------- Model Performance
    print(f'------------------------------------------------------------------------')
    print(f'---------- {datalabel}ing Data Performance\n------------------------------------------------------------------------')
    # -----  Create a confusion matrix
    conf_mat = confusion_matrix(y, y_pred)
    print(f"Confusion Matrix\n{conf_mat}")
#        print(f"Confusion Matrix\n{confusion_matrix(y, y_pred, labels = [1,0])}")
    
    # -----  Score
    score_model = round( model.score(X, y), 4)
    print(f'\n-----------------------\n{datalabel} score: {score_model}')

    # -----  Balanced Accuracy
    score_ba = round( balanced_accuracy_score(y, y_pred), 4)
    print(f"Balanced Accuracy Score: {score_ba}")

    # -----  ROC AUC Score
    if len(y.value_counts())>2:
        score_roc_auc = round( roc_auc_score(y, model.predict_proba(X), multi_class='ovr'), 4)
    else:
        score_roc_auc = round( roc_auc_score(y, model.predict_proba(X)[:, 1]), 4)

    print(f"ROC AUC Score: {score_roc_auc}")


    mse = round( mean_squared_error(y, y_pred), 4)

    print(f"Mean Squared Error: {mse}")

    # ----- metrics caluculated from the Confusion Matrix
    if len(conf_mat) > 2:
        # Compute TP, TN, FP, FN for each class
        for i in range(3):
            tp = conf_mat[i, i]
            fn = np.sum(conf_mat[i, :]) - tp
            fp = np.sum(conf_mat[:, i]) - tp
            tn = np.sum(conf_mat) - tp - fn - fp
            
            # print(f"Class {i}:")
            # print(f"TP: {tp}, FN: {fn}, FP: {fp}, TN: {tn}")
            # print()
    else:
        # Unravel the confusion matrix
        tn, fp, fn, tp = conf_mat.ravel()

    print(f"------------------------------")
    print(f"--- Classification values")
    print(f"------------------------------")
    # Computer acuracy
    accuracy = round( ( (tp + tn) / (tp + tn + fp + fn) ),4)
    print("Accuracy:", accuracy)

    precision = round( (tp / (tp + fp) ),4)
    print("Precision:", precision)

    recall = round( (tp / (tp + fn) ),4)
    print("Recall:", recall)

    f1 = round( (2 * (precision * recall) / (precision + recall) ),4)
    print("F1-score:", f1)

    specificity = round( (tn / (tn + fp) ),4)
    print("Specificity:", specificity)

    fpr = round( (fp / (fp + tn) ),4)
    print("False Positive Rate:", fpr)

    mcc = round( ( (tp * tn - fp * fn) / (((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5) ),4)
    print("Matthews Correlation Coefficient:", mcc)

    # -----  Create a classification report
    print(f"\n-----------------------\nClassification Report\n{classification_report(y, y_pred)}")

    
    return {'model': type(model).__name__, 
            'slice': datalabel,
            'score':score_model, 
            'balanced_accuracy': score_ba, 
            'roc_auc_score':score_roc_auc,
            "Mean Squared Error": mse,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1,
            'Specificity': specificity,
            'False Positive Rate': fpr,
            'Matthews Correlation Coefficient': mcc
            }

# --------------------------------------
# ------- knn_plot()
# ---------- KNN Plot to select best n_neighbors value
# --------------------------------------
def knn_plot( data ):
    X_train, X_test, y_train, y_test = data
    k_range = 10
    train_scores = []
    test_scores = []
    for k in range(1, k_range, 2):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        train_score = knn.score(X_train, y_train)
        test_score = knn.score(X_test, y_test)
        train_scores.append(train_score)
        test_scores.append(test_score)
        print(f"k: {k}, Train/Test Score: {train_score:.3f}/{test_score:.3f}")
        
    # Plot the results
    plt.plot(range(1, k_range, 2), train_scores, marker='o', label="training scores")
    plt.plot(range(1, k_range, 2), test_scores, marker="x", label="testing scores")
    plt.xlabel("k neighbors")
    plt.ylabel("accuracy score")
    plt.legend()
    plt.show()



# --------------------------------------
# ------- modify_base_dataset(df, parameter_dict)
# ---------- 
# --------------------------------------
# Sample parameter_dict
"""
operation_dict = {  'target_column'    : 'diabetes',
                    'convert_to_binary':  True,
                    'scaler'           : 'standard', # options: none, standard, minmax
                    'random_sample'    : 'none'      # options: none, undersample, oversample, cluster, smote, smoteenn
                    }
"""
def modify_base_dataset(df, operation_dict):
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.under_sampling import ClusterCentroids
    from imblearn.over_sampling import SMOTE
    from imblearn.combine import SMOTEENN
#    from sklearn.model_selection import train_test_split

    if 'target_column' in operation_dict:
        target = operation_dict['target_column']
        if len(target) < 2:
            print(f"ERROR: 'target_column' not specified in operation_dict")
            return df
    else:
        print(f"ERROR: 'target_column' not specified in operation_dict")
        return df

    split = False
    # print(f"The dataset will be modified in order as follows:")
    # print(operation_dict)

    print(f"Base Dataset Modifications in Process")
    print(f"-------------------------------------")
    for operation in operation_dict.keys():
        print(f"**Operation:{operation}  {operation_dict[operation]}")
        # print(f"Start: df.shape: {df.shape}")
        # print(f"df[{target}].value_counts:  {df[target].value_counts()}")
        if operation in ['scaler', 'random_sample']:
            if not split:
                print(f"  -- Performing train_test_split on dataframe with target:'{target}'\n     -- Run automatically before scalar or random_sample operations")
                # Create X and y datasets
                X = df.copy().drop(columns=[target])
                y = df[target]
                
#                print(f"-- Dataframe target:{df[target].value_counts()}  y:{y.value_counts()}")

                # Create Train Test Split
                data = sklearn.model_selection.train_test_split(X, y)
                split = True

        match operation:
            # Convert to binary is performed before splitting the dataset into train/test 
            case 'convert_to_binary':
                if operation_dict[operation]:
                    print(f"  -- Converting dataset to binary (0,1) from (0,1,2)")
                    clean_config = {
                                    'diabetes':{ 'scale': {},
                                    'translate': {1:0, 2:1},
                                    'values_to_drop': [] } }
                    df = clean_features_list(df, clean_config)

            # Scale, Random_oversample and Random_undersample are performed after the train/test split
            case 'split':
                if operation_dict[operation]:
                    if not split:
                        print(f"  -- Performing train_test_split on dataframe with target:'{target}'")
                        # Create X and y datasets
                        X = df.copy().drop(columns=[target])
                        y = df[target]
#                        print(f"-- Dataframe target:{df[target].value_counts()}  y:{y.value_counts()}")

                        # Create Train Test Split
                        data = sklearn.model_selection.train_test_split(X, y)

            case 'scaler':
#                print(f"Diabetes valuecounts s2  {df['diabetes'].value_counts()}")
                match operation_dict[operation]:
                    case 'standard':
                        print(f"  -- Performing StandardScaler on X_train: Updates X_train, y_test")
                        X_train, X_test, y_train, y_test = data
                        # Scale the X data by using StandardScaler()
                        scaler = StandardScaler().fit(X_train)
                        X_train_scaled = scaler.transform(X_train)
                        X_test_scaled = scaler.transform(X_test)

                        data = X_train_scaled, X_test_scaled, y_train, y_test
                    case 'minmax':
                        print(f"  -- Performing MinMaxScaler on X_train: Updates X_train, y_test")
                        X_train, X_test, y_train, y_test = data

                        # Scale the X data by using MinMaxScaler()
                        scaler = MinMaxScaler().fit(X_train)
                        X_train_scaled = scaler.transform(X_train)
                        X_test_scaled = scaler.transform(X_test)

                        data = X_train_scaled, X_test_scaled, y_train, y_test

            case 'random_sample':
#                print(f"Diabetes valuecounts rs  {df['diabetes'].value_counts()}")
                match operation_dict[operation]:
                    case 'oversample':
                        print(f"  -- Performing RandomOverSampler on X_train, y_train: Updates X_train, y_train")
                        X_train, X_test, y_train, y_test = data
                        # Instantiate the RandomOverSampler instance
                        random_oversampler = RandomOverSampler(random_state=1)
                        # Fit the data to the model
                        X_resampled, y_resampled = random_oversampler.fit_resample(X_train, y_train)
                        data = X_resampled, X_test, y_resampled, y_test
                    case 'undersample':
                        print(f"  -- Performing RandomUnderSampler on X_train, y_train: Updates X_train, y_train")
                        X_train, X_test, y_train, y_test = data
                        # Instantiate a RandomUnderSampler instance
                        rus = RandomUnderSampler(random_state=1)
                        # Fit the training data to the random undersampler model
                        X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
                        data = X_resampled, X_test, y_resampled, y_test
                    case 'cluster':
                        print(f"  -- Performing ClusterCentroids on X_train, y_train: Updates X_train, y_train")
                        X_train, X_test, y_train, y_test = data
                        # Instantiate a ClusterCentroids instance
                        cc_sampler = ClusterCentroids(random_state=1)
                        # Fit the training data to the cluster centroids model
                        X_resampled, y_resampled = cc_sampler.fit_resample(X_train, y_train)
                        data = X_resampled, X_test, y_resampled, y_test
                    case 'smote':
                        print(f"  -- Performing SMOTE on X_train, y_train: Updates X_train, y_train")
                        X_train, X_test, y_train, y_test = data
                        # Set the sampling_strategy parameter equal to auto
                        smote_sampler = SMOTE(random_state=1, sampling_strategy='auto')
                        # Fit the training data to the smote_sampler model
                        X_resampled, y_resampled = smote_sampler.fit_resample(X_train_scaled, y_train)
                        data = X_resampled, X_test, y_resampled, y_test
                    case 'smoteenn':
                        print(f"  -- Performing SMOTEENN on X_train, y_train: Updates X_train, y_train")
                        X_train, X_test, y_train, y_test = data
                        # Instantiate the SMOTEENN instance
                        smote_enn = SMOTEENN(random_state=1)
                        # Fit the model to the training data
                        X_resampled, y_resampled = smote_enn.fit_resample(X_train_scaled, y_train)
                        data = X_resampled, X_test, y_resampled, y_test

        # print(f"Finish: df.shape: {df.shape}")
        # print(f"df[{target}].value_counts:  {df[target].value_counts()}")
    X_train, X_test, y_train, y_test = data
    print(f"\nDataframe, Train Test Summary")
    print(f"-----------------------------")
    print(f"Dataframe: {df.shape}  Data:{len(data)}, X_train:{len(X_train)}, y_train:{len(y_train)}, X_test:{len(X_test)}, y_test:{len(y_test)}")
    counts = [y_train.value_counts(), y_test.value_counts()]
    print(f"ValueCounts:   y_train: len:{len(counts)}   0:{counts[0][0]:7}   1:{counts[0][1]:6}")
    print(f"ValueCounts:   y_test : len:{len(counts)}   0:{counts[1][0]:7}   1:{counts[1][1]:6}")
    return data



# --------------------------------------------------------------------------------------------------

# --------------------------------------
# ------- Capture stdout and stderr of a function to a string
# ---------- From Stack Overflow: https://stackoverflow.com/questions/23270456/pipe-console-outputs-from-a-specific-function-into-a-file-python
# --------------------------------------
# --- Note: Caching the current value of sys.stdout in the context manager rather than using sys.__stdout__ to restore
# --- use:
# with OutStreamCapture() as osc:
#    somefunc(x)
#
# osc.stdout and osc.stderr are two strings that contain the output from the function
# --------------------------------------
import sys
from io import StringIO

class OutStreamCapture(object):
    """
    A context manager to replace stdout and stderr with StringIO objects and
    cache all output.
    """

    def __init__(self):
        self._stdout = None
        self._stderr = None
        self.stdout = None
        self.stderr = None

    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = StringIO()
        sys.stderr = StringIO()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Restore original values of stderr and stdout.
        The captured contents are stored as strings in the stdout and stderr
        members.
        """
        self.stdout = sys.stdout.getvalue()
        self.stderr = sys.stderr.getvalue()
        sys.stdout = self._stdout
        sys.stderr = self._stderr


class OutStreamCapture_stdout(object):
    """
    A context manager to replace stdout and stderr with StringIO objects and
    cache all output.
    """

    def __init__(self):
        self._stdout = None
        self.stdout = None

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = StringIO()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Restore original values of stdout.
        The captured contents are stored as strings in the stdout and stderr
        members.
        """
        self.stdout = sys.stdout.getvalue()
        sys.stdout = self._stdout



# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
#  OLD CODE
# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------



# --------------------------------------
# ------- model_performance_metrics
# ---------- 
# --------------------------------------
def model_performance_metrics_old2(model, data, datalabel):
    # Expand the model data
    X, y, y_pred = data

    # -------------------------------------- Model Performance
    print(f'------------------------------------------------------------------------')
    print(f'---------- {datalabel}ing Data Performance\n------------------------------------------------------------------------')
    # -----  Create a confusion matrix
    # if len(y.value_counts()) > 2:
    #     print(f"Confusion Matrix\n{multilabel_confusion_matrix(y, y_pred)}")
    # else:
    print(f"Confusion Matrix\n{confusion_matrix(y, y_pred)}")
#        print(f"Confusion Matrix\n{confusion_matrix(y, y_pred, labels = [1,0])}")
    
    # -----  Score
    score_model = model.score(X, y)
    print(f'\n-----------------------\n{datalabel} score: {score_model}')

    # -----  Balanced Accuracy
    score_ba = balanced_accuracy_score(y, y_pred)
    print(f"Balanced Accuracy Score: {score_ba}")

    # -----  ROC AUC Score
    if len(y.value_counts())>2:
        score_roc_auc = roc_auc_score(y, model.predict_proba(X), multi_class='ovr')
    else:
        score_roc_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])

    print(f"ROC AUC Score: {score_roc_auc}")

    # -----  Create a classification report
    print(f"\n-----------------------\nClassification Report\n{classification_report(y, y_pred)}")

    return {'model': type(model).__name__, 'slice': datalabel,'score':score_model, 'balanced_accuracy': score_ba, 'roc_auc_score':score_roc_auc}



# --------------------------------------
# ------- model_performance_metrics
# ---------- 
# --------------------------------------
def model_performance_metrics_details_old(model, data, datalabel):
    # Expand the model data
    X, y, y_pred = data

    # -------------------------------------- Model Performance
    print(f'------------------------------------')
    print(f'---------- {datalabel}ing Data Performance\n------------------------------------')
    # -----  Create a confusion matrix
    # if len(y.value_counts()) > 2:
    #     print(f"Confusion Matrix\n{multilabel_confusion_matrix(y, y_pred)}")
    # else:
    conf_matrix = confusion_matrix(y, y_pred)
    print(f"Confusion Matrix\n{conf_matrix}")
#        print(f"Confusion Matrix\n{confusion_matrix(y, y_pred, labels = [1,0])}")
    
    # -----  Score
    score_model = model.score(X, y)
    print(f'\n-----------------------\n{datalabel} score: {score_model}')

    # -----  Balanced Accuracy
    score_ba = balanced_accuracy_score(y, y_pred)
    print(f"Balanced Accuracy Score: {score_ba}")

    # -----  ROC AUC Score
    if len(y.value_counts())>2:
        score_roc_auc = roc_auc_score(y, model.predict_proba(X), multi_class='ovr')
    else:
        score_roc_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])

    print(f"ROC AUC Score: {score_roc_auc}")

    # -----  Create a classification report
    class_report = classification_report(y, y_pred)
    print(f"\n-----------------------\nClassification Report\n{class_report}")

    return {'model': type(model).__name__, 'slice': datalabel,'score':score_model, 'balanced_accuracy': score_ba, 'roc_auc_score':score_roc_auc, 'confusion_matrix': conf_matrix,'classification_report':class_report}


# --------------------------------------
# ------- modify_base_dataset_old(df, parameter_dict)
# ---------- 
# --------------------------------------
# Sample parameter_dict
"""
operation_dict = {  'target_column'    : 'diabetes',
                    'convert_to_binary':  True,
                    'scaler'           : 'standard', # options: none, standard, minmax
                    'random_sample'    : 'none'      # options: none, undersample, oversample
                    }
"""
def modify_base_dataset_old(df, operation_dict, target):
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
#    from sklearn.model_selection import train_test_split

    if 'target_column' not in operation_dict:
        print(f"ERROR: 'target_column' not specified in operation_dict")
        return df

    split = False
    print(f"The dataset will be modified in order as follows:")
    print(operation_dict)

    for operation in operation_dict.keys():
        print(f"{operation}  {operation_dict[operation]}")
        if operation in ['scaler', 'random_sample']:
            if not split:
                print(f"\nPerforming train_test_split on dataframe with target:'{target}'")
                # Create X and y datasets
                X = df.copy().drop(columns=[target])
                y = df[target]
                
#                print(f"-- Dataframe target:{df[target].value_counts()}  y:{y.value_counts()}")

                # Create Train Test Split
                data = sklearn.model_selection.train_test_split(X, y)
                split = True

        match operation:
            # Convert to binary is performed before splitting the dataset into train/test 
            case 'convert_to_binary':
                if operation_dict[operation]:
                    print(f"\nConverting dataset to binary (0,1) from (0,1,2)")
                    clean_config = {
                                    'diabetes':{ 'scale': {},
                                    'translate': {1:0, 2:1},
                                    'values_to_drop': [] } }
                    df = clean_features_list(df, clean_config)

            # Scale, Random_oversample and Random_undersample are performed after the train/test split
            case 'split':
                if operation_dict[operation]:
                    if not split:
                        print(f"\nPerforming train_test_split on dataframe with target:'{target}'")
                        # Create X and y datasets
                        X = df.copy().drop(columns=[target])
                        y = df[target]
#                        print(f"-- Dataframe target:{df[target].value_counts()}  y:{y.value_counts()}")

                        # Create Train Test Split
                        data = sklearn.model_selection.train_test_split(X, y)

            case 'scaler':
#                print(f"Diabetes valuecounts s2  {df['diabetes'].value_counts()}")
                match operation_dict[operation]:
                    case 'standard':
                        print(f"\nPerforming StandardScaler on X_train: Updates X_train, y_test")
                        X_train, X_test, y_train, y_test = data
                        # Scale the X data by using StandardScaler()
                        scaler = StandardScaler().fit(X_train)
                        X_train_scaled = scaler.transform(X_train)
                        X_test_scaled = scaler.transform(X_test)

                        data = X_train_scaled, X_test_scaled, y_train, y_test
                    case 'minmax':
                        print(f"\nPerforming MinMaxScaler on X_train: Updates X_train, y_test")
                        X_train, X_test, y_train, y_test = data

                        # Scale the X data by using MinMaxScaler()
                        scaler = MinMaxScaler().fit(X_train)
                        X_train_scaled = scaler.transform(X_train)
                        X_test_scaled = scaler.transform(X_test)

                        data = X_train_scaled, X_test_scaled, y_train, y_test

            case 'random_sample':
#                print(f"Diabetes valuecounts rs  {df['diabetes'].value_counts()}")
                match operation_dict[operation]:
                    case 'oversample':
                        print(f"\nPerforming RandomOverSampler on X_train, y_train: Updates X_train, y_train")
                        X_train, X_test, y_train, y_test = data
                        # Instantiate the RandomOverSampler instance
                        random_oversampler = RandomOverSampler(random_state=1)
                        # Fit the data to the model
                        X_resampled, y_resampled = random_oversampler.fit_resample(X_train, y_train)
                        data = X_resampled, X_test, y_resampled, y_test = data
                    case 'undersample':
                        print(f"\nPerforming RandomUnderSampler on X_train, y_train: Updates X_train, y_train")
                        X_train, X_test, y_train, y_test = data
                        df = random_undersample(df)
                        # Instantiate a RandomUnderSampler instance
                        rus = RandomUnderSampler(random_state=1)
                        # Fit the training data to the random undersampler model
                        X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
                        data = X_resampled, X_test, y_resampled, y_test = data

    print(f"\nDataframe, Train Test Summary")
    print(f"-----------------------------")
    print(f"Dataframe: {df.shape}  Data:{len(data)}, X_train:{len(X_train)}, y_train:{len(y_train)}, X_test:{len(X_test)}, y_test:{len(y_test)}")
    counts = [y_train.value_counts(), y_test.value_counts()]
    print(f"ValueCounts:   y_train: len:{len(counts)}   0:{counts[0][0]:7}   1:{counts[0][1]:6}")
    print(f"ValueCounts:   y_test : len:{len(counts)}   0:{counts[1][0]:7}   1:{counts[1][1]:6}")
    return data










# --------------------------------------
# ------- model_performance_metrics
# ---------- 
# --------------------------------------
def model_performance_metrics_old(model, data):
    # Expand the model data
    X_train, X_test, y_train, y_test = data
    y_train_pred, y_test_pred        = predict

    # -------------------------------------- Training Performance
    print(f'Training Data Performance:')
    # -----  Create a confusion matrix
    # if len(y_train.value_counts()) > 2:
    #     print(f"Confusion Matrix\n{multilabel_confusion_matrix(y_train, y_train_pred)}")
    # else:
    print(f"Confusion Matrix\n{confusion_matrix(y_train, y_train_pred)}")
#        print(f"Confusion Matrix\n{confusion_matrix(y_train, y_train_pred, labels = [1,0])}")
    
    # -----  Training Score
    print(f'\nTrain score: {model.score(X_train, y_train)}')
    # -----  Balanced Accuracy
    print(f"Balanced Accuracy Score: {balanced_accuracy_score(y_train, y_train_pred)}")

    # -----  Create a classification report
    print(f"\nClassification Report\n{classification_report(y_train, y_train_pred)}")

    if len(y_train.value_counts())>2:
        print(f"\nROC AUC Score: {roc_auc_score(y_train, model.predict_proba(X_train), multi_class='ovr')}")
    else:
        print(f"\nROC AUC Score: {roc_auc_score(y_train, model.decision_function(X_train))}")

    # # Predict values with probabilities
    # pred_probas = model.predict_proba(X_train)
    # pred_probas_firsts = [prob[1] for prob in pred_probas]
    # # Print the first 5 probabilities
    # print(f"\nPredict Probabilities - First 5:")
    # for i in range(5):
    #     print(f"{i}: {pred_probas_firsts[i]:9.6f}")
    # print(f"\nROC AUC Score: {roc_auc_score(y_train, pred_probas_firsts)}")


    # print(f"--------------------------------------------------")

    # # -------------------------------------- Testing Performance
    # print(f'Testing Data Performance:')
    # # -----  Create a confusion matrix
    # print(f"Confusion Matrix\n{confusion_matrix(y_test, y_test_pred, labels = [1,0])}")

    # # -----  Testing Score
    # print(f'\ntest score: {model.score(X_test, y_test)}')
    # # -----  Balanced Accuracy
    # print(f"Balanced Accuracy Score: {balanced_accuracy_score(y_test, y_test_pred)}")

    # # -----  Create a classification report
    # print(f"\nClassification Report\n{classification_report(y_test, y_test_pred, labels = [1, 0])}")

    # # Predict values with probabilities
    # pred_probas = model.predict_proba(X_test)
    # pred_probas_firsts = [prob[1] for prob in pred_probas]
    # # Print the first 5 probabilities
    # print(f"\nPredict Probabilities - First 5:\n{pred_probas_firsts[0:5]:9.6f}")
    # print(f"\nROC AUC Score: {roc_auc_score(y_test, pred_probas_firsts)}")
