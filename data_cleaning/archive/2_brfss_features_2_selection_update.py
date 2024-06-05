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

sys.path.insert(1, 'pkgs')
import ml_functions as mlfuncs

# Load list/dict/dataframe structure from  file
with open('../data/codebook2022.pkl', 'rb') as file:
    brfss_survey_list = pickle.load(file)

start_item = int(input('\nReview start item #(0..N): '))


#for i in range(0,2):
for i in range(start_item, len(brfss_survey_list)):
    os.system('clear')
    print(f"\n\n-----------------------------------")
    item_dict = brfss_survey_list[i]
    for key in item_dict.keys():
        if (key == 'Section Name'):
            print(f"-----------------------------------")
            print(f"{key}:       {item_dict[key]}\n")
        elif(key == 'Question'):
            print(f"{key}:       {item_dict[key]}\n")
        elif(key == 'Label'):
            print(f"{i:3d}: {key}:       {item_dict[key]}\n")
        elif(key == 'table'):
#            A=1
            print(item_dict[key])
        else:
            print(f"{key}:       {item_dict[key]}")

    notDone = True
    while notDone:
        select = input('\nSurvey Item for Diabetes Study (y/n/q): ')
        select = select.lower()
        if (select == 'y' or select == 'n'):
            notDone = False
            brfss_survey_list[i]['candidate'] = select
        elif (select == 'q'):
            notDone = False

    if select == 'q':
        i -= 1
        break

print(f"----------------------------------------------------")
print(f'\nProcessed thru item: {i} of {len(brfss_survey_list)-1}')

# Save list/dict/dataframe structure to file
with open('../data/codebook2022-partial.pkl', 'wb') as file:
    pickle.dump(brfss_survey_list, file)

print(f"\nResults written to ../data/codebook2022-partial.pkl\n")
print(f"----------------------------------------------------")