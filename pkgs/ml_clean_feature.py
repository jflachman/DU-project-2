# --------------------------------------
# ------- Imports
# --------------------------------------
import pandas as pd
import numpy as np
import fastparquet as fp
import os
import pickle



# --------------------------------------
# ------- clean_all_features
# ---------- clean all features in features_df that exist in clean_configs
# --------------------------------------
# Notes:  BRFSS data is already segmented into numerical values.  
#         However, some data such as "no response" is not useful.  
#         Also, it is useful to shift some values and make them ordinal
#    translate      = a dictionary of changes to feature values (translate 'a' to 'b' is 'a':'b')
#    values_to_drop = list of values to drop
#    scale          = divisor and round values to scale the feature
def clean_all_features(feature_df, clean_configs):
    for column in feature_df:
        if column in clean_configs:
            print(f"\n\nCleaning Feature: {column}")

            # ----------------------------------------------------------------------------
            # NOTE: COMMENT OUT THE FOLLOWING LINES ONCE FEATURE PARAMETERS ARE REFINED
            feature_list = feature_df[column].unique()
            feature_list = np.sort(feature_list)
            if len(feature_list)> 50:
                first_50 = feature_list[:50]
                print(f"  Initial Unique features in [{column}]:  \n********** More than {len(first_50)} features, list is truncated to first 50 **********\n{first_50}")
            else:
                print(f"  Initial Unique features in [{column}]:  {feature_list}")
            # ----------------------------------------------------------------------------
            clean_config = clean_configs[column]
            
            if 'values_to_drop' in clean_config:
                params = clean_config['values_to_drop']     # Expecting a list of values to drop
                if not params:
                    print(f"  {'values_to_drop'}: ********* NO Parameters were specified *********")
                else:
                    print(f"  {'values_to_drop'}: {params}")
                    feature_df = feature_df[~feature_df[column].isin(params)]
            
            if 'translate' in clean_config:
                params = clean_config['translate']          # Expecting a dictionary of translations (from:to values)
                if not params:
                    print(f"  {'translate'}: ********* NO Parameters were specified *********")
                else:
                    print(f"  {'translate'}: {params}")
                    feature_df[column] = feature_df[column].replace(params)
            
            if 'translate2' in clean_config:
                params = clean_config['translate2']          # Expecting a dictionary of translations (from:to values)
                if not params:
                    print(f"  {'translate2'}: ********* NO Parameters were specified *********")
                else:
                    print(f"  {'translate2'}: {params}")
                    feature_df[column] = feature_df[column].replace(params)
            
            if 'scale' in clean_config:
                params = clean_config['scale']              # expecting dictionary with divisor and rounding values
                if not params:
                    print(f"  {'scale'}: ********* NO Parameters were specified *********")
                else:
                    print(f"  {'scale'}: {params}")

            feature_list = feature_df[column].unique()
            feature_list = np.sort(feature_list)
            if len(feature_list)> 50:
                first_50 = feature_list[:50]
                print(f"  FINAL Unique features in [{column}]:  \n********** More than {len(first_50)} features, list is truncated to first 50 **********\n{first_50}")
            else:
                print(f"  FINAL Unique features in [{column}]:  {feature_list}")


            # if len(feature_df[column].unique())> 50:
            #     first_50 = feature_df[column].unique()[:50]
            #     print(f"  FINAL Unique features in [{column}]:  \n********** More than {len(first_50)} features, list is truncated to first 50 **********\n{first_50}")
            # else:
            #     print(f"  FINAL Unique features in [{column}]:  {feature_df[column].unique()}")
        else:
            print(f"Feature DOES NOT exist: {column}")

    return feature_df



# --------------------------------------
# ------- clean_feature
# ---------- cleans features in clean_configs that exist in features_df
# --------------------------------------
def clean_features_list(feature_df, clean_configs):
    for feature in clean_configs:
        if feature in feature_df:
            print(f"\n\n****Cleaning Feature: {feature}")

            # ----------------------------------------------------------------------------
            # NOTE: COMMENT OUT THE FOLLOWING LINES ONCE FEATURE PARAMETERS ARE REFINED
            feature_list = feature_df[feature].unique()
            feature_list = np.sort(feature_list)
            if len(feature_list)> 50:
                first_50 = feature_list[:50]
                print(f"  Initial Unique features in [{feature}]:  \n********** More than {len(first_50)} features, list is truncated to first 50 **********\n{first_50}")
            else:
                print(f"  Initial Unique features in [{feature}]:  {feature_list}")
            # ----------------------------------------------------------------------------
            clean_config = clean_configs[feature]
            
            if 'values_to_drop' in clean_config:
                params = clean_config['values_to_drop']     # Expecting a list of values to drop
                if not params:
                    print(f"  {'values_to_drop'}: ********* NO Parameters were specified *********")
                else:
                    print(f"  {'values_to_drop'}: {params}")
                    feature_df = feature_df[~feature_df[feature].isin(params)]
            
            if 'translate' in clean_config:
                params = clean_config['translate']          # Expecting a dictionary of translations (from:to values)
                if not params:
                    print(f"  {'translate'}: ********* NO Parameters were specified *********")
                else:
                    print(f"  {'translate'}: {params}")
                    feature_df[feature] = feature_df[feature].replace(params)
            
            if 'translate2' in clean_config:
                params = clean_config['translate2']          # Expecting a dictionary of translations (from:to values)
                if not params:
                    print(f"  {'translate2'}: ********* NO Parameters were specified *********")
                else:
                    print(f"  {'translate2'}: {params}")
                    feature_df[feature] = feature_df[feature].replace(params)
            
            if 'scale' in clean_config:
                params = clean_config['scale']              # expecting dictionary with divisor and rounding values
                if not params:
                    print(f"  {'scale'}: ********* NO Parameters were specified *********")
                else:
                    print(f"  {'scale'}: {params}")

            feature_list = feature_df[feature].unique()
            feature_list = np.sort(feature_list)
            if len(feature_list)> 50:
                first_50 = feature_list[:50]
                print(f"  FINAL Unique features in [{feature}]:  \n********** More than {len(first_50)} features, list is truncated to first 50 **********\n{first_50}")
            else:
                print(f"  FINAL Unique features in [{feature}]:  {feature_list}")
        else:
            print(f"Feature DOES NOT exist: {feature}")

    return feature_df

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

