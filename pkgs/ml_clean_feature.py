# --------------------------------------
# ------- Imports
# --------------------------------------
import pandas as pd
import numpy as np
import fastparquet as fp
import os
import pickle



# --------------------------------------
# ------- clean_features
# ---------- 
# --------------------------------------



# --------------------------------------
# ------- clean_features_translate_values
# ---------- 
# --------------------------------------



# --------------------------------------
# ------- clean_features_
# ---------- 
# --------------------------------------


# --------------------------------------
# ------- clean_features( feature, df):
# ---------- 
# --------------------------------------
# Notes:  BRFSS data is already segmented into numerical values.  
#         However, some data such as "no response" is not useful.  
#         Also, it is useful to shift some values and make them ordinal
#    feature_translations contains a dictionary of changes to feature values
#    ids_to_drop contains a 




def clean_features( feature, df):
    
    match feature:
        case 'GENHLTH':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['GENHLTH'] = df['GENHLTH'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'PHYSHLTH':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['PHYSHLTH'] = df['PHYSHLTH'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'MENTHLTH':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['MENTHLTH'] = df['MENTHLTH'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'PRIMINSR':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['PRIMINSR'] = df['PRIMINSR'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'PERSDOC3':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['PERSDOC3'] = df['PERSDOC3'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'CHECKUP1':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['CHECKUP1'] = df['CHECKUP1'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'EXERANY2':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['EXERANY2'] = df['EXERANY2'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'BPHIGH6':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['BPHIGH6'] = df['BPHIGH6'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'BPMEDS':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['BPMEDS'] = df['BPMEDS'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'CHOLCHK3':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['CHOLCHK3'] = df['CHOLCHK3'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'TOLDHI3':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['TOLDHI3'] = df['TOLDHI3'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'CHOLMED3':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['CHOLMED3'] = df['CHOLMED3'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'CVDCRHD4':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['CVDCRHD4'] = df['CVDCRHD4'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'CVDSTRK3':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['CVDSTRK3'] = df['CVDSTRK3'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'ADDEPEV3':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['ADDEPEV3'] = df['ADDEPEV3'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'CHCKDNY2':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['CHCKDNY2'] = df['CHCKDNY2'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'DIABETE4':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['DIABETE4'] = df['DIABETE4'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'MARITAL':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['MARITAL'] = df['MARITAL'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'EDUCA':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['EDUCA'] = df['EDUCA'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'RENTHOM1':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['RENTHOM1'] = df['RENTHOM1'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'EMPLOY1':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['EMPLOY1'] = df['EMPLOY1'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'INCOME3':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['INCOME3'] = df['INCOME3'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'WEIGHT2':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['WEIGHT2'] = df['WEIGHT2'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'DEAF':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['DEAF'] = df['DEAF'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'BLIND':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['BLIND'] = df['BLIND'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'DIFFWALK':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['DIFFWALK'] = df['DIFFWALK'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'FLUSHOT7':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['FLUSHOT7'] = df['FLUSHOT7'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'PREDIAB1':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['PREDIAB1'] = df['PREDIAB1'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'CHKHEMO3':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['CHKHEMO3'] = df['CHKHEMO3'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'EYEEXAM1':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['EYEEXAM1'] = df['EYEEXAM1'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'TOLDCFS':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['TOLDCFS'] = df['TOLDCFS'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'HAVECFS':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['HAVECFS'] = df['HAVECFS'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'TOLDHEPC':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['TOLDHEPC'] = df['TOLDHEPC'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'HAVEHEPB':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['HAVEHEPB'] = df['HAVEHEPB'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'HPVADVC4':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['HPVADVC4'] = df['HPVADVC4'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'SHINGLE2':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['SHINGLE2'] = df['SHINGLE2'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'CIMEMLOS':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['CIMEMLOS'] = df['CIMEMLOS'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'CDDISCUS':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['CDDISCUS'] = df['CDDISCUS'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'MSCODE':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['MSCODE'] = df['MSCODE'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case '_IMPRACE':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['_IMPRACE'] = df['_IMPRACE'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case '_RFHLTH':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['_RFHLTH'] = df['_RFHLTH'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case '_HLTHPLN':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['_HLTHPLN'] = df['_HLTHPLN'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case '_TOTINDA':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['_TOTINDA'] = df['_TOTINDA'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case '_MICHD':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['_MICHD'] = df['_MICHD'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case '_PRACE1':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['_PRACE1'] = df['_PRACE1'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case '_RACE':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['_RACE'] = df['_RACE'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case '_RACEGR3':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['_RACEGR3'] = df['_RACEGR3'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case '_SEX':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['_SEX'] = df['_SEX'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case '_AGEG5YR':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['_AGEG5YR'] = df['_AGEG5YR'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'WTKG3':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['WTKG3'] = df['WTKG3'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case '_BMI5':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['_BMI5'] = df['_BMI5'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case '_BMI5CAT':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['_BMI5CAT'] = df['_BMI5CAT'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case '_EDUCAG':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['_EDUCAG'] = df['_EDUCAG'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case '_INCOMG1':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['_INCOMG1'] = df['_INCOMG1'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case '_SMOKER3':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['_SMOKER3'] = df['_SMOKER3'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case '_RFSMOK3':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['_RFSMOK3'] = df['_RFSMOK3'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case '_CURECI1':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['_CURECI1'] = df['_CURECI1'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case '_DRNKWK1':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['_DRNKWK1'] = df['_DRNKWK1'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case '_RFDRHV7':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['_RFDRHV7'] = df['_RFDRHV7'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'FTJUDA2_':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['FTJUDA2_'] = df['FTJUDA2_'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'FRUTDA2_':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['FRUTDA2_'] = df['FRUTDA2_'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'GRENDA1_':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['GRENDA1_'] = df['GRENDA1_'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'FRNCHDA_':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['FRNCHDA_'] = df['FRNCHDA_'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'POTADA1_':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['POTADA1_'] = df['POTADA1_'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case 'VEGEDA2_':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['VEGEDA2_'] = df['VEGEDA2_'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case '_FRUTSU1':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['_FRUTSU1'] = df['_FRUTSU1'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case '_VEGESU1':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['_VEGESU1'] = df['_VEGESU1'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
        case '_FRTLT1A':
            feature_translations ={1:0, 2:1}
            values_to_drop = []
            df['_FRTLT1A'] = df['_FRTLT1A'].replace(feature_translations)
            df = df[~df['ID'].isin(values_to_drop)]
    return df


# --------------------------------------
# ------- File components
# ---------- 
# --------------------------------------

# --------------------------------------
# ------- File components
# ---------- 
# --------------------------------------

# --------------------------------------
# ------- File components
# ---------- 
# --------------------------------------

# --------------------------------------
# ------- File components
# ---------- 
# --------------------------------------

# --------------------------------------
# ------- File components
# ---------- 
# --------------------------------------

# --------------------------------------
# ------- File components
# ---------- 
# --------------------------------------

# --------------------------------------
# ------- File components
# ---------- 
# --------------------------------------

# --------------------------------------
# ------- File components
# ---------- 
# --------------------------------------

# --------------------------------------
# ------- File components
# ---------- 
# --------------------------------------

# --------------------------------------
# ------- File components
# ---------- 
# --------------------------------------

