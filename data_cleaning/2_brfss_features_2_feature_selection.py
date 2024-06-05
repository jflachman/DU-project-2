# package imports go here
import pandas as pd
import numpy as np
import fastparquet as fp
import os
import sys
import zipfile
import requests
import io
import pickle


sys.path.insert(1, '../pkgs')
import ml_functions as mlfuncs

source_file = '../data/codebooks/codebook2021.pkl'
result_file = '../data/codebooks/codebook2021_candidates.pkl'


# Load list/dict/dataframe structure from  file
with open(source_file, 'rb') as file:
    brfss_codebook_list = pickle.load(file)

start_item = int(input('\nReview start item #(0..N): '))

i, brfss_survey_selected = mlfuncs.select_feature_candidates(start_item, brfss_codebook_list)

print(f"----------------------------------------------------")
print(f'\nProcessed thru item: {i} of {len(brfss_codebook_list)-1}')

# Save list/dict/dataframe structure to file
with open(result_file, 'wb') as file:
    pickle.dump(brfss_survey_selected, file)

print(f"\nResults written to {result_file}\n")
print(f"----------------------------------------------------")
