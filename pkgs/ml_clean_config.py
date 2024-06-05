
def clean_configurations():

# Sample of cleaning parameters
#        'GENHLTH':{ 'scale': {'div':100, 'round':2},
#                    'translate': {1:0, 2:1},
#                    'values_to_drop': [7,9] },

    clean_config = {
        # 'GENHLTH': 
        #       
        'GENHLTH':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [4] },
        # 'PHYSHLTH': 
        #       
        'PHYSHLTH':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'MENTHLTH': 
        #       
        'MENTHLTH':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'PRIMINSR': 
        #       
        'PRIMINSR':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'PERSDOC3': 
        #       
        'PERSDOC3':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'CHECKUP1': 
        #       
        'CHECKUP1':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'EXERANY2': 
        #       
        'EXERANY2':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'BPHIGH6': 
        #       
        'BPHIGH6':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'BPMEDS': 
        #       
        'BPMEDS':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'CHOLCHK3': 
        #       
        'CHOLCHK3':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'TOLDHI3': 
        #       
        'TOLDHI3':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'CHOLMED3': 
        #       
        'CHOLMED3':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'CVDCRHD4': 
        #       
        'CVDCRHD4':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'CVDSTRK3': 
        #       
        'CVDSTRK3':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'ADDEPEV3': 
        #       
        'ADDEPEV3':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'CHCKDNY2': 
        #       
        'CHCKDNY2':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'DIABETE4': 
        #       
        'DIABETE4':{ 'scale': {},
                    'translate': {2:0, 3:0, 1:2, 4:1},
                    'values_to_drop': [7, 9] },
        # 'MARITAL': 
        #       
        'MARITAL':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'EDUCA': 
        #       
        'EDUCA':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'RENTHOM1': 
        #       
        'RENTHOM1':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'EMPLOY1': 
        #       
        'EMPLOY1':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'INCOME3': 
        #       
        'INCOME3':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'WEIGHT2': 
        #       
        'WEIGHT2':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'DEAF': 
        #       
        'DEAF':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'BLIND': 
        #       
        'BLIND':{ 'scale': {},
                    'translate': {},
                        'values_to_drop': [] },
        # 'DIFFWALK': 
        #       
        'DIFFWALK':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'FLUSHOT7': 
        #       
        'FLUSHOT7':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'PREDIAB1': 
        #       
        'PREDIAB1':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'CHKHEMO3': 
        #       
        'CHKHEMO3':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'EYEEXAM1': 
        #       
        'EYEEXAM1':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'TOLDCFS': 
        #       
        'TOLDCFS':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'HAVECFS': 
        #       
        'HAVECFS':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'TOLDHEPC': 
        #       
        'TOLDHEPC':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'HAVEHEPB': 
        #       
        'HAVEHEPB':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'HPVADVC4': 
        #       
        'HPVADVC4':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'SHINGLE2': 
        #       
        'SHINGLE2':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'CIMEMLOS': 
        #       
        'CIMEMLOS':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'CDDISCUS': 
        #       
        'CDDISCUS':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'MSCODE': 
        #       
        'MSCODE':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # '_IMPRACE': 
        #       
        '_IMPRACE':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # '_RFHLTH': 
        #       
        '_RFHLTH':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # '_HLTHPLN': 
        #       
        '_HLTHPLN':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # '_TOTINDA': 
        #       
        '_TOTINDA':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # '_MICHD': 
        #       
        '_MICHD':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # '_PRACE1': 
        #       
        '_PRACE1':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # '_RACE': 
        #       
        '_RACE':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # '_RACEGR3': 
        #       
        '_RACEGR3':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # '_SEX': 
        #       
        '_SEX':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # '_AGEG5YR': 
        #       
        '_AGEG5YR':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'WTKG3': 
        #       
        'WTKG3':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # '_BMI5': 
        #       
        '_BMI5':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # '_BMI5CAT': 
        #       
        '_BMI5CAT':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # '_EDUCAG': 
        #       
        '_EDUCAG':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # '_INCOMG1': 
        #       
        '_INCOMG1':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # '_SMOKER3': 
        #       
        '_SMOKER3':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # '_RFSMOK3': 
        #       
        '_RFSMOK3':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # '_CURECI1': 
        #       
        '_CURECI1':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # '_DRNKWK1': 
        #       
        '_DRNKWK1':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # '_RFDRHV7': 
        #       
        '_RFDRHV7':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'FTJUDA2_': 
        #       
        'FTJUDA2_':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'FRUTDA2_': 
        #       
        'FRUTDA2_':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'GRENDA1_': 
        #       
        'GRENDA1_':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'FRNCHDA_': 
        #       
        'FRNCHDA_':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'POTADA1_': 
        #       
        'POTADA1_':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # 'VEGEDA2_': 
        #       
        'VEGEDA2_':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # '_FRUTSU1': 
        #       
        '_FRUTSU1':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # '_VEGESU1': 
        #       
        '_VEGESU1':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        # '_FRTLT1A': 
        #       
        '_FRTLT1A':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] }
    }
    return clean_config