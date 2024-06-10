year = 2015
package_path        = '../../pkgs'
source_path         = 'data/brfss/'
data_path           = 'data/'
report_path         = 'reports/'
prepared_data_path  = '../../../prepared_data/'

source_file                 = source_path + 'archive.zip'
clean_file                  = data_path + 'brfss_' + str(year) + '_clean.parquet.gzip'
performance_report          = report_path + 'performance_report.pkl'

prepared_data_standard              = prepared_data_path + 'standard_scaled.pkl'
prepared_data_minmax                = prepared_data_path + 'minmax_scaled.pkl'
prepared_data_binary                = prepared_data_path + 'binary_target.pkl'
prepared_data_sb_random_undersample = prepared_data_path + 'sb_random_undersample_scaled.pkl'
prepared_data_sb_random_oversample  = prepared_data_path + 'sb_random_oversample_scaled.pkl'
prepared_data_sb_cluster            = prepared_data_path + 'sb_cluster_scaled.pkl'
prepared_data_sb_smote              = prepared_data_path + 'sb_smote_scaled.pkl'
prepared_data_sb_smoteenn           = prepared_data_path + 'sb_smoteenn_scaled.pkl'


# # Config Settings
# year = year         = config.year

# source_path         = config.source_path
# data_path           = config.data_path
# report_path         = config.report_path

# source_file         = config.source_file
# clean_file          = config.clean_file
# performance_report  = config.performance_report


# # BE SURE TO UPDATE THE LABEL FOR THIS ANALYSIS
# # -----------------------------
# dataset_label = 'Base Dataset'
# # -----------------------------
# file_label = dataset_label.lower().replace(' ','_')
# detailed_performance_report = report_path + file_label + '_detailed_performance_report.txt'