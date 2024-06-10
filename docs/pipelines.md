# Pipelines developed for this analysis

**Data Prep Pipeline**<br>
A pipeline was prototyped to run multiple models on the 2015 dataset.  The prototpye was then productionized into a more robust pipeline that allowed additional feature transformation, scaling and sampling methods to address issues found with unbalanced data and overfitting.

- transformation (options):
    - convert_to_binary(True/False):
        - True:  converts the target feature `diabetes` from (0,1,2) to (binary:0,1): 0: no diabetes, 1: diabetes
        - False: retains the base target feature values: (0,1,2): 0: no diabetes, 1: pre-diabetes, 2: diabetes
- Scaling (options):
    - `none`: No scaling is performed
    - `standard`: StandardScaler is performed
    - `minmax`: MinMaxScaler is performed
- random_sample (options):
    - `oversample`: RandomOverSampler is performed on X_train, y_train
    - `undersample`: RandomUnderSampler is performed on on X_train, y_train
    - `cluster`: ClusterCentroids is performed on X_train, y_train
    - `smote`: Smote is performed on X_train, y_train
    - `smoteenn`: Smoteenn is performed on X_train, y_train

These options are controlled by passing the following dictionary to the `modify_base_datasets` function.  

    operation_dict = {  'target_column'    : 'diabetes',
            'convert_to_binary':  True,
            'scaler'           : 'standard', # options: none, standard, minmax
            'random_sample'    : 'none'      # options: none, undersample, oversample, cluster, smote, smoteen
            }

The `modify_base_data` function returns `data` where:

    data = modify_base_datasets(....)
    X_train, X_test, y_train, y_test = data

Each file (2_data_analysis_#...ipynb) file starts with the same base data that was cleaned as described above.  The data was then transformed, scaled and sampled using 

**model run pipeline**

The model run pipeline take `data`

    X_train, X_test, y_train, y_test = data
