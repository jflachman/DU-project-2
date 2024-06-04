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

# Load list/dict/dataframe structure from  file
with open('../data/codebook2022_firstpass.pkl', 'rb') as file:
    brfss_survey_list = pickle.load(file)

#for i in range(0,2):
new_list = []
key_list = 
for i in range(0, len(brfss_survey_list)):
    dict = brfss_survey_list[i]
    if 'key1' in dict.keys():

    print(f"{i}  {dict['Label']}")
    new_list.append(
        [ i, 
#        dict['Module Number'],
        dict['Question Number'],
#        dict['candidate'],
        dict['Label'],
        dict['Section Name'],
        dict['Question'] ]
    )

df = pd.DataFrame(new_list, columns=['num', 'Question Number', 'Label', 'Section Name', 'Question'])

df.to_csv('../data/condidate_summary.csv')

print(f"----------------------------------------------------")
print(f"\nResults written to ../data/condidate_summary.csv\n")
print(f"----------------------------------------------------")
