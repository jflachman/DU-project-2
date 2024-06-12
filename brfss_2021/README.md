# BRFSS 2021 Dataset:

## Directories:

The the BRFSS data for 2021 was pulled automaticall, converted to a datafile and downloaded the the data directory.  The results of the cleaned 2021 data was also place there

- **data**

Results are saved in the following two directories for generating reports:

- **reports**: results for the analysis runs
- **optimize:** results for the optimization runs



## Files:

**Data Cleaning:**  The Data cleaning process was refined in ../data_cleaning.  However, this file performed all the 2021 data cleaning in a single file.  It does include description of the cleaning steps and decisions made during feature reduction, feature imputaiton etc.
- **1_data_cleaning_2021.ipynb**

**Batch processing:** all the analysis was performed by the following two file
- **2_data_analysis_0_notebooks.txt **: list of files to process
- **2_data_analysis_0_run_all.ipynb** : file that ran the process 

**Analsis files:** Our dataset of 247K rows and 37 columns and was 86% imbalanced.  The files here ran all 7 models against the dataset defined in the file.  
- **2_data_analysis_1_base_data.ipynb:** Base dataset with no changes to the feature of 0/1/2 no diabetes, pre-diabetes, diabetes
- **2_data_analysis_2.0_standard_scaled_data.ipynb**: Performed StandardScaler on the base dataset
- **2_data_analysis_2.1_minmax_scaled_data.ipynb:** Performed MinMaxScaler on the base dataset
- **2_data_analysis_3_ss_binary_data.ipynb**  
- **2_data_analysis_4_ss_b_random_undersample.ipynb**
- **2_data_analysis_5_ss_b_random_oversample.ipynb**
- **2_data_analysis_6_sb_cluster.ipynb**
- **2_data_analysis_7_sb_smote.ipynb**
- **2_data_analysis_8_sb_smoteen.ipynb**

**Performance Report:** for the Analysis performed above (2_data_analysis...ipynb)
- **3_performance_report.ipynb:** This file generates a consolidated report of the analysis runs.  Individual detailed reports are contained in the reports directory.

**Optimizaton:**  
- **4_data_analysis_optimization_0_notebooks.txt**
- **4_data_analysis_optimization_0_run_all.ipynb**  Runs all the optimization notebooks
- **4_data_analysis_optimization_1_DecisionTreeClassifier.ipynb** Initial notebook to run decision tree optimizer
- **4_data_analysis_optimization_1_binary_standardScaler.ipynb** Initial notebook to run 
- **4_data_analysis_optimization_1_bs_SMOTE.ipynb**
- **4_data_analysis_optimization_2_LogisticRegression.ipynb**

**Optimization Report: **
- **5_optimization_report.ipynb:** This file generates a consolidated report of the optimization runs.  Individual detailed reports are contained in the optimize directory.

