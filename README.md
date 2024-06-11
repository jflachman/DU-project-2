# DU-project-2

## Diabetes Prediction from CDC Behavioral Risk Factor Surveillance System (BRFSS) Survey data
- See the [Project Instructions](project-2-overview.md) for more details about the project requirements


## Team
 
 - Jeff Flachman
 - Ava Lee
 - Elia Porter

## Project Checklist:

<hr color="red">
<span style="color: red;">

***Please see the [project checklist](project-2-checklist.md)*** for artifacts supporting the project objectives.

</span>
<hr color="red">

# Executive Summary

This project aims to analyze the factors contributing to the prevalence of diabetes in the United State and determine if they provide some predictive value in determining a diagnosis of diabetes.

A dataset pulled from the 2015 BRFSS was available on the [**UC Irvine Machine Learning Repository**](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators).  This dataset was cleaned from the [2015 BRFSS survey](https://www.cdc.gov/brfss/annual_data/annual_2015.html) data.  The team also pulled an cleaned data from the [2021 BRFSS survey](https://www.cdc.gov/brfss/annual_data/annual_2021.html)


# Project Overview

Diabetes is the eighth leading cause of death in the United Stages.  But many rank it second behind heart disease as a chronic illness that leads to death.  Diabetes also has a daily implact of those who live with it.

The team was interested in diabetes predictions using data from the The CDC Behavioral Risk Factor Surveillance System (BRFSS).  The BRFSS is an annual phone survey of 300K-400K respondents.  

A dataset pulled from the 2015 BRFSS was available on the [**UC Irvine Machine Learning Repository**](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators).  This dataset had already been cleaned from the [2015 BRFSS survey](https://www.cdc.gov/brfss/annual_data/annual_2015.html) data.  The 2015 dataset was older and already cleaned.  Therefore, the team also pulled and cleaned the [2021 BRFSS survey](https://www.cdc.gov/brfss/annual_data/annual_2021.html).

Diabetes ***risk factors*** listed below were used to select survey results from BRFSS dataset.  The [2015 dataset contained 21 features](docs/diabetes_features-2015.md).  The team selected [35 features from the 2021 dataset](docs/diabetes_features_2021.md).

Two targets were evaluated:  

- **target 1** (0,1,2): 0: no diabetes, 1: pre-diabetes, 2: diabetes
- **target 2** (binary:0,1): 0: no diabetes, 1: diabetes

Classification models were trained and the metrics were computed.  In addition, alternate scaling and sampling techniques were used to handle the imbalance in the datasets.

In all, 63 configurations of binary/012, scaling & sampling method, and models were trained and the metrics were computed for each base dataset (2015 & 2021).

The metrics were evaluated and a few targeted dataset configurations were selected to optimize.  These included:
    - binary target, standard scalar, randomUnderSample data with models:
        - LogisticRegression optimized with with RandomizedSearchCV 
        - AdaBoost optimized with with RandomizedSearchCV

For more information, see the Details below.



# Project Details:

## Ideation

### Potential Datasets Evaluated

The team brainstormed multiple dataset options for this project.  Some of the datasets reviewed are listed in the [datasets](data_sets.md) files.  The team reviewed candidate datasets for abalone, mushroom, bike sharing, and diabetes.

The team was most interested in diabetes predictions using data from the The CDC Behavioral Risk Factor Surveillance System (BRFSS). The BRFSS is an annual phone survey of 300K-400K respondents.

As a starting point, a dataset pulled from the 2015 BRFSS was used from the [**UC Irvine Machine Learning Repository**](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)

## Project directory structure

README.md - This file provides a description of the project

Directories:

**Note:** The project ***classification analysis and results*** of the CDC Behavioral Risk Factor Surveillance System (BRFSS) Survey data are contained in the `brfss_2015` and `brfss_2021` directories.

| directory | description |
| --------- | ----------- |
| `brfss_2015` | Contains all the anlaysis of the 2015 BRFSS data.  A description of the files in this directory can be found in the [brfss_2015 README](brfss_2015/README.md) |
| `brfss_2021` | Contains all the anlaysis of the 2021 BRFSS data.  A description of the files in this directory can be found in the [brfss_2021 README](brfss_2021/README.md) |
| `data_cleaning` | Contains all the work the read codebooks, refine features, transform and modify feature values.  This work was then moved into the brfss_2015 and brfss_2021 directories as signle 1_....ipynb files which pulled and cleaned the data for each year|
|`pkgs` | python files containing pipline code writting for data | data | files used by the notebooks in data_cleaning |
| `docs` | other markdown files referenced by this README and other docs |
|`imgs` | graphics included in documentation markdowns |
| `prototype_ava` | Prototype code written by Ava Lee |
| `prototype_elia` | Prototype code written by Elia Porter |
| `prototype_jeff` | Prototype code written by Jeff Flachman |

**Note:** Additonal READMEs are available in some subdirectories.


## Feature Engineering


### Understanding Diabetes
---
I order to understand the features, it is important to understand the risks and indicators of diabetes.

**Risk Factors**

There are many risk factors for developing type 2 diabetes, including:

- **Age:** Being over 40 increases your risk.
- **Family history**: Having a parent, sibling, or other relative with type 1 or type 2 diabetes increases your risk.
- **Ethnicity**: People of certain races and ethnicities, including African Americans, Hispanics, American Indians, and Asian-Americans, are more likely to develop type 2 diabetes.
- **Inactivity**: The less active you are, the greater your risk.
- **Weight**: Being overweight or obese increases your risk. You can estimate your risk by measuring your waist circumference. Men have a higher risk if their waist circumference is more than 40 inches, while women who are not pregnant have a higher risk if their waist circumference is more than 35 inches.
- **Blood pressure**: High blood pressure can lead to insulin resistance and eventually type 2 diabetes.
- **Cholesterol**: High cholesterol can raise your risk for diabetes and heart disease.
- **Smoking**: Smokers are more 30-40% more likely than non-smokers to develop type 2 diabetes. 

**Diabetes Indicators / Symptoms**

Diabetes is a chronic condition that can be diagnosed by a medical professional. While it often has no symptoms, some indicators include:

- **Urination**: Frequent urination, especially at night
- **Thirst**: Excessive thirst
- **Hunger**: Increased hunger, even when eating
- **Weight loss**: Unintentional weight loss
- **Fatigue**: Feeling more tired than usual
- **Vision**: Blurred vision
- **Wounds**: Cuts and bruises that take longer to heal
- **Skin**: Itchy skin or genital itching
- **Infections**: Urinary tract infections (UTIs) or yeast infections
- **Sensations**: Unusual sensations like tingling, burning, or pricklin

### Feature Selection

A list of features was pulled from the UCI/Kaggle documentation on the 2015 dataset.  In addition, the [2021 codebook](https://www.cdc.gov/brfss/annual_data/2021/pdf/codebook21_llcp-v2-508.pdf) was imported and parsed.  See the work in [data_cleaning](data_cleaning).  The features in the codebook were evaluated, selected and a summare of the selected [2021 features](docs/diabetes_features_2021.md) was written to a file.

Key features relevant to diabetes analysis were selected. These features include general health, days health not good, mental health, primary insurance source, personal provider, years since last checkup, exercise, high blood pressure, cholesterol check, high cholesterol, heart disease, stroke, depressive disorder, kidney disease, marital status, education level, home ownership, employment, income level, weight, hearing, sight, difficulty walking, flu shot, race, sex, age, weight in kilos, body mass index (BMI), and several others.

## Data Cleaning

A contributing factor to including 2021 data was that the features on 2015 data on UCI/Kaggle were already selected and cleaned.  Therefore, the team put in a considerable effort to automate including other CDC BRFSS survy years and clean the data.  Post 2015, 2021 had the most features related to the risk factors for diabetes.  Thus 2021 was selected as the best year to clean.  A list of years and features counts pulled from the CDC website is recorded in the [CDC - BRFSS Datasets by year](docs/CDC_BRFSS_Datasets_by_year.md) file.

The CDC also has a list of [Diabetes indicators for machine learning](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)

The files [ml_clean_config.py](pkgs/ml_clean_config.py) and [ml_clean_features.py](pkgs/ml_clean_features.py) contain the functions written to handle processing the codebooks, selecting features and cleaning the data.

**Cleaning**<br>
The CDC BRFSS Survery data responses were already provided a numeric values.  Therefore, Get_dummies, OneHotEncoder and OrdinalEncoder were not required.  However, it was necessary to do some cleaning.  Some responses were unknown or refused and those rows needed to be dropped.  Other values needed to be scaled (i.e. weight of 4015 kg needed to be scaled to 40.15 kg).  Finally, the numeric values for some responses needed to be transformed.  i.e. for exercise, the value 88 (no days) was transformed to 0 days, where 1-30 was number of days of month of exercise.    

Substantial time was spent productionizing (pipeline) the processing of the CDC Codebooks, simplifying feature extraction and feature cleaning and imputation.  The files supporting processing the codebooks and cleaning the data can be found in the [data_cleaning](data_cleaning) directory.  Ultimately, the feature descriptions for the 2021 dataset were automatically generated into the following file: [2021 features](docs/diabetes_features_2021.md).  A dictionary based configuration file was used to define the operations to be made on each feature in the dataset and a function then performed **imputation** on all features in a single function call.

Finally, the 2021 data originally had 55 feature.  A correlation matrix was plotted and the highest correlated feature to features were reviewed for potential duplication of information.  For example, the dataset started with 5 Race feature and this was reduced to 1 Race feature.  Several education features were reduced to 1.  The feature reduction as well as the other feature engineering steps can be found in the [2021 Data Cleaning Notebook](1_data_cleaning_2021.ipynb).  The final 2021 feature set has 36 feature and the target (diabetes)

This cleaning process produced the `base data` used in the Data Analysis process below.  The base data target feature `diabetes` consisted of three values:  

- (0,1,2): 0: no diabetes, 1: pre-diabetes, 2: diabetes

## Data Analysis

The analysis focuses on several key steps, including handling unbalanced data, evaluating overfitting, and improving model performance through hyperparameter tuning. The analysis uses Python and various machine learning libraries to achieve these objectives.

Several [pipelines](docs/pipelines.md) were used to streamline the analysis.

- `data preparation pipeline`: apply additional feature transformation, scaling and sampling methods to base data address issues found with unbalanced data and overfitting
    - This pipeline were used to run models and generate metrics for the following `modified datasets`:
        | # |feature | scaling | sampling | Dataset |
        | - |------- | ------- | -------- |------ |
        | 1 | diabetes 0/1/2 | none | none | Base dataset |
        | 2.0 | diabetes 0/1/2 | StandardScaler | none | standard_scaled |
        | 2.1 | diabetes 0/1/2 | MinMaxScaler | none | minmax_scaled |
        | 3 | diabetes 0/1 | StandardScaler | none | ss_binary |
        | 4 | diabetes 0/1 | StandardScaler | RandomUnderSampler | sb_random_undersample |
        | 5 | diabetes 0/1 | StandardScaler | RandomOverSampler | sb_random_oversample |
        | 6 | diabetes 0/1 | StandardScaler | ClusterCentroids | sb_cluster |
        | 7 | diabetes 0/1 | StandardScaler | SMOTE | sb_smote |
        | 8 | diabetes 0/1 | StandardScaler | SMPOTEENN | sb_smoteenn |

- `model execution pipeline`: ran a series of `9 models` collected metrics, displayed the metrics in the jupyter file and pushed them to a file.
    - models included:
        - KNeighborsClassifier(n_neighbors=k_value), data)
        - tree.DecisionTreeClassifier(), data)
        - RandomForestClassifier(), data)
        - ExtraTreesClassifier(random_state=1), data)
        - GradientBoostingClassifier(random_state=1), data)
        - AdaBoostClassifier(random_state=1), data)
        - LogisticRegression(), data)

**Evaluating Overfitting**

All the models were greatly overfit with the base dataset.

All models were then run against each modified dataset.  The metrics were prepared and archived in the [reports/](reports/) directory.

It was determined that overfitting occured in most cases.  However, it was minimized by using the `binary` target feature, scaling with `StandardScaler` or `MinMaxScaler` and resampled using `RandomOverSampling` or `RandomUnderSampling`.

**Imbalanced data**

- valuecount % of base data:
    | target value | % | description |
    |:------------:|:-:|:--:|
    | 0 | 84% | No diabetes |
    | 1 | 2% | Pre-diabetes |
    | 2 | 14% | Diabetes |

- Valuecount % of binary data has:
    | target value | % | description |
    |:------------:|:-:|:--:|
    | 0 | 86% | No diabetes |
    | 1 | 14% | Diabetes |

Using the following sampling methods improved the metric results:
- RandomOverSampler
- RandomUnderSampler
- ClusterCentroids
- SMOTE
- SMOTEENN

`RandomOverSampler` & `RandomUnderSampler` performed as well as the others and had a better execution time.  `RandomUnderSampler` provided the smallest dataset to train and fit.  Therefore it was used in the optimization phase.

**Metric Evaluation**

The metrics for all the `modified datasets` and `models` are provided in the [reports directory](reports/).  The [performance summary](reports/performance_report.txt) shows the performance of all model executions.  The detailed reports are listed below:
- The details of the 2015 dataset runs are contained in these file:
    - [base_dataset_detailed_performance_report.txt](reports/base_dataset_detailed_performance_report.txt)
    - [binary_dataset_detailed_performance_report.txt](reports/binary_dataset_detailed_performance_report.txt)
    - [standard_scaled_dataset_detailed_performance_report.txt](reports/standard_scaled_dataset_detailed_performance_report.txt)
    - [minmax_scaled_dataset_detailed_performance_report.txt](reports/minmax_scaled_dataset_detailed_performance_report.txt)
    - [randomundersampled_dataset_detailed_performance_report.txt](reports/randomundersampled_dataset_detailed_performance_report.txt)
    - [randomoversample_dataset_detailed_performance_report.txt](reports/randomoversample_dataset_detailed_performance.txt)
    - [cluster_dataset_detailed_performance_report.txt](reports/cluster_dataset_detailed_performance_report.txt)
    - [smote_dataset_detailed_performance_report.txt](reports/smote_dataset_detailed_performance_report.txt)
    - [smoteen_dataset_detailed_performance_report.txt](reports/smoteen_dataset_detailed_performance_report.txt)

- The details of the 2015 dataset runs are contained in these file:
    - [base_dataset_detailed_performance_report.txt](reports/base_dataset_detailed_performance_report.txt)
    - [binary_dataset_detailed_performance_report.txt](reports/binary_dataset_detailed_performance_report.txt)
    - [standard_scaled_dataset_detailed_performance_report.txt](reports/standard_scaled_dataset_detailed_performance_report.txt)
    - [minmax_scaled_dataset_detailed_performance_report.txt](reports/minmax_scaled_dataset_detailed_performance_report.txt)
    - [randomundersampled_dataset_detailed_performance_report.txt](reports/randomundersampled_dataset_detailed_performance_report.txt)
    - [randomoversample_dataset_detailed_performance_report.txt](reports/randomoversample_dataset_detailed_performance.txt)
    - [cluster_dataset_detailed_performance_report.txt](reports/cluster_dataset_detailed_performance_report.txt)
    - [smote_dataset_detailed_performance_report.txt](reports/smote_dataset_detailed_performance_report.txt)
    - [smoteen_dataset_detailed_performance_report.txt](reports/smoteen_dataset_detailed_performance_report.txt)


## Optimization / Hyperparameter tuning

#### Hyperparameter Tuning

1. **Decision Tree Classifier + Randomized Search CV:**
   - We sampled a fixed number of parameter settings from specified ranges for efficiency
   - The optimization helped but not a substantial amount on this dataset
   - We sorted the highest F1 score, precision, and accuracy
   - The results were these 4 data sets:
(imgs/2021_best_models.png)

   - The final parameters and scores reflect the optimized model's ability to predict diabetes with higher accuracy and reliability.
  
# Conclusions - Project Goal Achievement?

**Conclusion:**
   - Conclusions from 63 Model/Dataset Runs for each year (126 total dataset/model combinations)
   - We achieved good accuracy; but because of imbalance struggled with Precision Optimization helped some, but did not make large gains for most models.

**Top Models**
 - GradientBoostingClassifier
 - AdaBoostClassifier
 - LogisticRegression

**Top Datasets**
- Binary dataset with StandardScalar
- Binary, Standard Scalar & SMOTEEN sampling.

**Project Goal: Achieved**
- Successfully identified key factors contributing to diabetes prevalence.
- Developed predictive models with significant accuracy and reliability.Strong Predictive performance through application of pipelines, optimized datasets, advanced classification models, model performance ranking, and model optimization.
