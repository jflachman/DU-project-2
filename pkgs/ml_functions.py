# --------------------------------------
# ------- Imports
# --------------------------------------
import pandas as pd
import numpy as np
import fastparquet as fp
import os
import pickle

# --------------------------------------
# ------- File components
# ---------- 
# --------------------------------------
cdc_base_URL    = "https://www.cdc.gov/brfss/annual_data/"
cdc_1990_2010   = "/files/CDBRFS"
cdc_2011_2022   = "/files/LLCP"


# --------------------------------------
# ------- list_files
# --------- create a list files of a type from a directory and its subdirectories
# --------------------------------------
# Note: requires import os
def recursive_file_list(directory, file_type):
    files_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(file_type):
                relative_path = os.path.relpath(os.path.join(root, file), directory)
                files_list.append(relative_path)
    return files_list

# --------------------------------------
# ------- get_subdirectories(filename, root_path)
# ---------- 
# --------------------------------------
def get_subdirectories(filename, root_path):
    directory_path = os.path.dirname(filename)
    directory_path = directory_path.replace(root_path, '')
    if (directory_path != ''):
        directory_path = directory_path + '/'
    return directory_path

# --------------------------------------
# ------- 
# ---------- 
# --------------------------------------
def create_subdirectory( directory):   
    # Check if the directory already exists
    if not os.path.exists(directory):
        # Create the directory if it doesn't exist
        os.makedirs(directory)
#        print(f"Directory '{directory}' created successfully.")
#    else:
#        print(f"Directory '{directory}' already exists.")

# --------------------------------------
# ------- bfrss_cdc_file
# ---------- 
# --------------------------------------
def brfss_cdc_file( year ):
    if (year >= 2011):
        cdc_file = cdc_base_URL + str(year) + cdc_2011_2022 + str(year) + "XPT.zip"
        # print(cdc_file)
    elif (year >= 1990):
        cdc_file = cdc_base_URL + str(year) + cdc_1990_2010 + str(year)[2:] + "XPT.zip"
    else:
        print(f"Invalid Date (1990-2022)")

    return cdc_file

# --------------------------------------
# ------- brfss_parquet_file
# ---------- 
# --------------------------------------
def brfss_parquet_file( result_path, year ):
    return result_path + "cdc_brfss_" + str( year) + ".parquet.gzip"

# --------------------------------------
# ------- process_cdc_file
# ---------- 
# --------------------------------------
def process_cdc_file( result_path, year ):
    create_subdirectory( result_path )
    if (year != 2009):           # 2009 file is corrupted on CDC site.
        file = brfss_cdc_file( year )
        print(f"Year: {year}  {file}")
        df = pd.read_sas(file, compression='zip', format='xport')
        parquet_file = brfss_parquet_file( result_path, year )
        print(f"            Writing File: {parquet_file}")
        df.to_parquet(parquet_file, compression='gzip', engine="fastparquet")
    # else:
    #     file = 'https://www.amazon.com/clouddrive/share/HAfuNnNSbFqKmdyuodrVAQMpgcyqoFACuBoKWIqoWeG/folder/k68TM2nPRy6XHf9GrpSh3Q/EhgTUKi6TBaezEBKRLY0_g?ref_=cd_ph_share_link_copy'
    #     print(f"Year: {year}  {file}")
    #     df = pd.read_csv(file,low_memory=False)
    #     parquet_file = brfss_parquet_file( result_path, year )
    #     print(f"            Writing File: {parquet_file}")
    #     df.to_parquet(parquet_file, compression='gzip', engine="fastparquet")

# --------------------------------------
# ------- Read CDC parquet file
# ---------- 
# --------------------------------------
def read_cdc_brfss( result_path, year ):
    return pd.read_parquet(brfss_parquet_file( result_path, year ), engine="fastparquet")


# --------------------------------------
# ------- get_label_dict
# ---------- 
# --------------------------------------
def get_label_dict( index, labels ):
    dict = {}
    item_list = []
    for label in labels:
        item_list = item_list + label.replace('\xa0', ' ').split(':')

    items = []
    for i, item in enumerate(item_list):
        item = item.strip()
        item_list[i] = item
        match item:
            case 'Label':
                dict['Label'] = i
                items.append(item)
            case 'Section Name':
                dict['Section Name'] = i
                items.append(item)
            case 'Section Number':
                dict['Section Number'] = i
                items.append(item)
            case 'Core Section Number':
                dict['Core Section Number'] = i
                items.append(item)
            case 'Module Number':
                dict['Module Number'] = i
                items.append(item)
            case 'Question Number':
                dict['Question Number'] = i
                items.append(item)
            case 'Column':
                dict['Column'] = i
                items.append(item)
            case 'Type of Variable':
                dict['Type of Variable'] = i
                items.append(item)
            case 'SAS Variable Name':
                dict['SAS Variable Name'] = i
                items.append(item)
            case 'Question Prologue':
                dict['Question Prologue'] = i
                items.append(item)
            case 'Question':
                dict['Question'] = i
                items.append(item)
            # case _:
            #     print(f"Error {i}: Don't know how to process {item}")   

    # print(f"\n{i}  #Items: {len(items)}   #Item_List:{len(item_list   )}")
    # print(f"Items: {items}")
    # print(f"Item List: {item_list}\n")

    for i, item in enumerate(items):
        if (i < len(items) - 1):
#            print(f"{i} {item} {items[i]} - NOT last")
            start = dict[item] + 1
            end = dict[items[i+1]]-1
            if end >= start:
                if end > start:
                    dict[item] = ': '.join(item_list[start:end])
                else:
                    dict[item] = item_list[start]
            else:
                dict[item] = ''
        else:
            if (dict[item]+1 < len(item_list)-1):
#                print(f"Last Item A:  {dict[item]+1}  {len(item_list)-1}")
                dict[item] = ": ".join(item_list[dict[item]+1:len(item_list)])
            else:
                dict[item] = item_list[dict[item]+1]
#                print(f"Last Item B:  {dict[item]+1}  {len(item_list)-1}")
    return dict



# --------------------------------------
# ------- Read, display and process items
# ---------- 
# --------------------------------------
def select_feature_candidates(start_item, brfss_codebook_list):
    #for i in range(0,2):
    for i in range(start_item, len(brfss_codebook_list)):
        os.system('clear')
        print(f"\n\n-----------------------------------")
        item_dict = brfss_codebook_list[i]
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
                brfss_codebook_list[i]['candidate'] = select
            elif (select == 'q'):
                notDone = False

        if select == 'q':
            i -= 1
            break
    return i, brfss_codebook_list

# --------------------------------------
# ------- Read Codebooks for 2019 and 2022
# ---------- 
# --------------------------------------
def read_codebook( codebook_data ):
    codebook_list = []
    #for i in range(2, 8, 2):
    for i in range(2, len(codebook_data), 2):
        df = codebook_data[i]
    #     print(df)
        labels = df.columns[1][0].replace(': ',':').split(' ')
        df.columns = [col[1] for col in df.columns.values]
        dict = get_label_dict( i, labels)
        dict['table'] = df
        codebook_list.append(dict)
    return codebook_list

# --------------------------------------
# ------- read_codebook_pdf_html (2020 & 2021)
# ---------- Use for pdf codebooks that were saved as HTML files
# --------------------------------------
def read_codebook_pdf_html( codebook_data):
    label_list = ['Label', 'candidate', 'Section Name', 'Section Number', 'Core Section Number', 'Module Number', 'Question Number', 'Column', 'Type of Variable', 'SAS Variable Name', 'Question Prologue', 'Question', 'table']

    codebook_list = []

    for i in range(0, len(codebook_data)):
        df = codebook_data[1]
        labels = df[0][0]

        labels = labels.replace(': ',':')
        for label in label_list:
            labels = labels.replace(label, ':'+label)
        labels = labels.replace('  ',' ').replace(' :',':')
        labels = labels.replace('::',':').split(':')
        labels.remove('')

        new_header = df.iloc[1]
        df = df[2:]
        df.columns = new_header
        dict = get_label_dict( i, labels)
        dict['table'] = df
        codebook_list.append(dict)

    return codebook_list




# --------------------------------------
# ------- Create summary of features for year.
# ---------- 
# --------------------------------------
def codebook_summary( source_file, destination_file):

    print(f"----------------------------------------------------")
    print(f"Codebook details read from {source_file}")

    # Load list/dict/dataframe structure from  file
    with open(source_file, 'rb') as file: brfss_survey_list = pickle.load(file)

    #for i in range(0,2):
    codebook_list = []
    key_list = ['Label', 'candidate', 'Section Name', 'Section Number', 'Core Section Number', 'Module Number', 'Question Number', 'Column', 'Type of Variable', 'SAS Variable Name', 'Question Prologue', 'Question', 'table']
    key_list_section = ['Section Number', 'Core Section Number', 'Module Number']
    for i in range(0, len(brfss_survey_list)):
        dict = brfss_survey_list[i]
        temp_list = []
        for key in key_list:
            if key in dict.keys():
                if key in key_list_section:
                    temp_list.append(dict[key])
                elif key == 'table':
                    A = 1
                else:
                    temp_list.append(dict[key])
            else:
                if key == 'candidate':
                    temp_list.append('n')
                else:
                    if key not in key_list_section:
                        print(f'Missing Key: {key}')
                    
        codebook_list.append( temp_list )
    df = pd.DataFrame(codebook_list, columns=['Label', 'candidate', 'Section Name', 'Section Number', 'Question Number', 'Column', 'Type of Variable', 'SAS Variable Name', 'Question Prologue', 'Question'])
    df.to_csv(destination_file)

    print(f"Results written to {destination_file}")
    print(f"----------------------------------------------------")

# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# ------- OLD OLD OLD OLD OLD 
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------

