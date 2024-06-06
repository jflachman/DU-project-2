
def clean_configurations():

# Sample of cleaning parameters
#        'GENHLTH':{ 'scale': {'div':100, 'round':2},
#                    'translate': {1:0, 2:1},
#                    'values_to_drop': [7,9] },

    clean_config = {
        # 'GENHLTH': Done
        #       1:0         - Excellent
        #       2:1         - Very Good
        #       3:2         - Good
        #       4:3         - Fair
        #       5:4         - Poor
        #       Removed( 7: Don't know, 9: Refused)
        'GENHLTH':{ 'scale': {},
                    'translate': {1:0, 2:1, 3:2, 4:3, 5:4},
                    'values_to_drop': [7,9] },
        #       
        # 'PHYSHLTH': Done
        #       1-30:   - num days
        #       88:0    - None
        #       Remove( 77: Don't know, 99: Refused)
        'PHYSHLTH':{ 'scale': {},
                    'translate': {88:0},
                    'values_to_drop': [77,99] },
        #       
        # 'MENTHLTH': Done
        #       1-30:   - num days
        #       88:0    - None
        #       Remove( 77: Don't know, 99: Refused)
        'MENTHLTH':{ 'scale': {},
                    'translate': {88:0},
                    'values_to_drop': [77,99] },
        #       
        # 'PRIMINSR': Done
        #       1-10    - Enumeration of different insurance types
        #       88:0    - No Coverage
        #       Remove( 77: Don't know, 99: Refused)
        'PRIMINSR':{ 'scale': {},
                    'translate': {88:0},
                    'values_to_drop': [77,99] },
        #       
        # 'PERSDOC3': Done
        #       Have Personal Health Care Provider?
        'PERSDOC3':{ 'scale': {},
                    'translate': {3:0},
                    'values_to_drop': [7, 9] },
        #       
        # 'CHECKUP1': Done
        #       
        'CHECKUP1':{ 'scale': {},
                    'translate': {8:0},
                    'values_to_drop': [7, 9] },
        #       
        # 'EXERANY2': Done
        #       
        'EXERANY2':{ 'scale': {},
                    'translate': {2:0},
                    'values_to_drop': [7, 9] },
        #       
        # 'BPHIGH6': Done
        #       3:0         - No
        #       2:1         - Yes, but only when pregnant
        #       1:3         - Yes
        #       4:2         - borderline high or pre-hypertensive or elevated blood pressure
        #       Removed (7: Don't know, 9: Refused)
        'BPHIGH6':{ 'scale': {},
                    'translate': {3:0, 1:5},        # 3(no):0, 1(yes):5
                    'translate2': {2:1, 4:2, 5:3},  # 2(yes but only during pregnancy):1, 4(borderline):2, 5(yes):3
                    'values_to_drop': [7,9] },
        #       
        # 'BPMEDS': Done
        #       
        'BPMEDS':{ 'scale': {},
                    'translate': {2:0},
                    'values_to_drop': [7, 9] },
        #       
        # 'CHOLCHK3': Done
        #       
        'CHOLCHK3':{ 'scale': {},
                    'translate': {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 8:6},
                    'values_to_drop': [7, 9] },
        #       
        # 'TOLDHI3': Done
        #       
        'TOLDHI3':{ 'scale': {},
                    'translate': {2:0},
                    'values_to_drop': [7,9] },
        #       
        # 'CHOLMED3': Done
        #       
        'CHOLMED3':{ 'scale': {},
                    'translate': {2:0},
                    'values_to_drop': [7, 9] },
        #       
        # 'CVDCRHD4': Done
        #       
        'CVDCRHD4':{ 'scale': {},
                    'translate': {2:0},
                    'values_to_drop': [7, 9] },
        #       
        # 'CVDSTRK3': Done
        #       
        'CVDSTRK3':{ 'scale': {},
                    'translate': {2:0},
                    'values_to_drop': [7,9] },
        #       
        # 'ADDEPEV3': Done
        #       
        'ADDEPEV3':{ 'scale': {},
                    'translate': {2:0},
                    'values_to_drop': [7, 9] },
        #       
        # 'CHCKDNY2': Done
        #       
        'CHCKDNY2':{ 'scale': {},
                    'translate': {2:0},
                    'values_to_drop': [7, 9] },
        #       
        # 'DIABETE4': Done
        #       1:2     - Yes
        #       2:0     - Yes, but female told only during pregnancy
        #       3:0     - No
        #       4:1     - No, pre-diabetes, or borderline diabetes
        'DIABETE4':{ 'scale': {},
                    'translate': {2:0, 3:0, 1:2, 4:1},
                    'values_to_drop': [7, 9] },
        #       
        # 'MARITAL': Done
        #       
        'MARITAL':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [9] },
        #       
        # 'EDUCA': Done
        #       
        'EDUCA':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [9] },
        #       
        # 'RENTHOM1': Done
        #       
        'RENTHOM1':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [7,9] },
        #       
        # 'EMPLOY1': Done
        #       
        'EMPLOY1':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [9] },
        #       
        # 'INCOME3': Done
        #       Reduce 1-11 to 0-10 - Income levels
        #       Remove (77: Don't know, 99: Refused)
        'INCOME3':{ 'scale': {},
                    'translate': {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9, 11:10},
                    'values_to_drop': [77,99] },
        #       
        # 'WEIGHT2': Fix later
        #       
        'WEIGHT2':{ 'scale': {'div':100, 'round':2},
                    'translate': {},
                    'values_to_drop': [7777,9999] },
        #       
        # 'DEAF': Done
        #       
        'DEAF':{ 'scale': {},
                    'translate': {2:0},
                    'values_to_drop': [7,9] },
        #       
        # 'BLIND': Done
        #       
        'BLIND':{ 'scale': {},
                    'translate': {2:0},
                        'values_to_drop': [7, 9] },
        #       
        # 'DIFFWALK': Done
        #       
        'DIFFWALK':{ 'scale': {},
                    'translate': {2:0},
                    'values_to_drop': [7,9] },
        #       
        # 'FLUSHOT7': Done
        #       
        'FLUSHOT7':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        #       
        # 'PREDIAB1': Consider dropping or making a second factor
        #       
        'PREDIAB1':{ 'scale': {},
                    'translate': {3:0, 2:1},
                    'values_to_drop': [7, 9] },
        #       
        # 'CHKHEMO3': Done
        #       
        'CHKHEMO3':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [88, 98, 99, 77] },
        #       
        # 'EYEEXAM1': Done
        #       
        'EYEEXAM1':{ 'scale': {},
                    'translate': {8:0},
                    'values_to_drop': [7, 9] },
        #       
        # 'TOLDCFS': No Values
        #       
        'TOLDCFS':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        #       
        # 'HAVECFS': No Values
        #       
        'HAVECFS':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        #       
        # 'TOLDHEPC': Done
        #       
        'TOLDHEPC':{ 'scale': {},
                    'translate': {2:0},
                    'values_to_drop': [7, 9] },
        #       
        # 'HAVEHEPB': Done
        #       
        'HAVEHEPB':{ 'scale': {},
                    'translate': {2:0},
                    'values_to_drop': [7,9] },
        #       
        # 'HPVADVC4': Done
        #       
        'HPVADVC4':{ 'scale': {},
                    'translate': {2:0},
                    'values_to_drop': [3,7,9] },
        #       
        # 'SHINGLE2': Done
        #       
        'SHINGLE2':{ 'scale': {},
                    'translate': {2:0},
                    'values_to_drop': [7,9] },
        #       
        # 'CIMEMLOS': Done
        #       
        'CIMEMLOS':{ 'scale': {},
                    'translate': {2:0},
                    'values_to_drop': [7,9] },
        #       
        # 'CDDISCUS': Done
        #       
        'CDDISCUS':{ 'scale': {},
                    'translate': {2:0},
                    'values_to_drop': [7,9] },
        #       
        # 'MSCODE': Done
        #       
        'MSCODE':{ 'scale': {},
                    'translate': {5:4},
                    'values_to_drop': [] },
        #       
        # '_IMPRACE': Done
        #       
        '_IMPRACE':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        #       
        # '_RFHLTH': Done
        #       
        '_RFHLTH':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [9] },
        #       
        # '_HLTHPLN': Done
        #       Have any health insurance 
        #       2:0         - No
        #       1           - Yes
        #       Remove(9: Don't know or missing)
        '_HLTHPLN':{ 'scale': {},
                    'translate': {2:0},
                    'values_to_drop': [9] },
        #       
        # '_TOTINDA': Done
        #       1           - physical activity
        #       change 2:0  - no physical activity
        #       Remove 9    - don't know
        '_TOTINDA':{ 'scale': {},
                    'translate': {2:0},
                    'values_to_drop': [9] },
        #       
        # '_MICHD': Done
        #       2:0         - Does not have MI or CHD
        #       1           - Reported MI or CHD
        '_MICHD':{ 'scale': {},
                    'translate': {2:0},
                    'values_to_drop': [] },
        #       
        # '_PRACE1': Done
        #       
        '_PRACE1':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [77,99] },
        #       
        # '_RACE': Done
        #       
        '_RACE':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [9] },
        #       
        # '_RACEGR3': Done
        #       
        '_RACEGR3':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [9] },
        #       
        # '_SEX': Done
        #       1:0     - Male
        #       2:1     - Female
        '_SEX':{ 'scale': {},
                    'translate': {1:0, 2:1},
                    'values_to_drop': [] },
        # '_AGEG5YR': Done
        #       1-13        - age buckets
        #       Remove(14: Don't know, Refused)
        '_AGEG5YR':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [14] },
        #       
        # 'WTKG3': Done
        #        Computed Weight in Kilograms
        'WTKG3':{ 'scale': {'div':100, 'round':2},
                    'translate': {},
                    'values_to_drop': [] },
        #       
        # '_BMI5': Done
        #       
        '_BMI5':{ 'scale': {'div':100, 'round':2},
                    'translate': {},
                    'values_to_drop': [] },
        #       
        # '_BMI5CAT': Done
        #       1:0         - Underweight
        #       2:1         - Normal Weight
        #       3:2         - Overweight
        #       4:3         - Obese
        '_BMI5CAT':{ 'scale': {},
                    'translate': {1:0, 2:1, 3:2, 4:3},
                    'values_to_drop': [] },
        #       
        # '_EDUCAG': Done
        #       1:0         - Did not graduate HS
        #       2:1         - HS
        #       3:2         - Attended College / Technical school
        #       4:3         - Graduated College / Technical school
        #       Remove(9: don't know, missing)
        '_EDUCAG':{ 'scale': {},
                    'translate': {1:0, 2:1, 3:2, 4:3},
                    'values_to_drop': [9] },
        #       
        # '_INCOMG1': Done
        #       
        '_INCOMG1':{ 'scale': {},
                    'translate': {1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6},
                    'values_to_drop': [9] },
        #       
        # '_SMOKER3': Done
        #       1:3         - Current Smoker - now smokes every day
        #       2:2         - Current Smoker - now smokes some days
        #       3:1         - Former Smoker
        #       4:0         - Never Smoked
        #       Remove(9: Don't know or Refused)
        '_SMOKER3':{ 'scale': {},
                    'translate': {4:0, 1:5},    # 4(never):0, 1(every day):5
                    'translate2': {3:1, 5:3},   # 3(former):1, 2:(some days):2, 5(every day):3
                    'values_to_drop': [9] },
        #       
        #       
        # '_RFSMOK3': Done
        #       1:0         - No
        #       2:1         - Yes
        #       Remove(9, Don't know, Refused)
        '_RFSMOK3':{ 'scale': {},
                    'translate': {1:0, 2:1},
                    'values_to_drop': [9] },
        #       
        # '_CURECI1': Done
        #       
        '_CURECI1':{ 'scale': {},
                    'translate': {1:0, 2:1},
                    'values_to_drop': [9] },
        #       
        # '_DRNKWK1': Done
        #       
        '_DRNKWK1':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [4] },
        #       
        # '_RFDRHV7': Done
        #       Heavy Alcohol Consumption Calculated Variable
        '_RFDRHV7':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [9] },
        #       
        # 'FTJUDA2_': Done
        #       Computed Fruit Juice intake in times per day 
        #       No Change
        'FTJUDA2_':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        #       
        # 'FRUTDA2_': Done
        #       Computed Fruit intake in times per day 
        #       No Change
        'FRUTDA2_':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        #       
        # 'GRENDA1_': Done
        #       Computed Dark Green Vegetable intake in times per day
        #       No Change
        'GRENDA1_':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        #       
        # 'FRNCHDA_': Done
        #       Computed French Fry intake in times per day 
        #       No Change
        'FRNCHDA_':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        #       
        # 'POTADA1_': Done
        #       Computed Potato Servings per day 
        #       No Change
        'POTADA1_':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        #       
        # 'VEGEDA2_': Done
        #       Computed Other Vegetable intake in times per day
        #       No Change
        'VEGEDA2_':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        #       
        # '_FRUTSU1': Done
        #       Total fruits consumed per day
        #       No Change
        '_FRUTSU1':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        #       
        # '_VEGESU1': Done
        #       Total vegetables consumed per day
        #       No Change
        '_VEGESU1':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] },
        #       
        # '_FRTLT1A': 
        #       Consume Fruit 1 or more times per day 
        #       No Change
        '_FRTLT1A':{ 'scale': {},
                    'translate': {},
                    'values_to_drop': [] }
    }
    return clean_config