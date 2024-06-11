year = 2021
package_path    = '../pkgs'
source_path     = 'data/brfss/'
data_path       = 'data/'
report_path     = 'reports/'
optimize_path   = 'optimize/'

source_file                 = source_path + 'archive.zip'
clean_file                  = data_path + 'brfss_' + str(year) + '_clean.parquet.gzip'
performance_report          = report_path + 'performance_report.pkl'
optimization_report         = optimize_path + 'optimization_report.pkl'

target                      = 'diabetes'

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