## BRFSS Data Cleaning

### 2021 Files:
- [BFRSS Files](https://www.cdc.gov/brfss/annual_data/annual_2021.html)
- [2021 BRFSS Overview CDC](https://www.cdc.gov/brfss/annual_data/2021/pdf/Overview_2021-508.pdf)
- [2021 BRFSS Codebook CDC](https://www.cdc.gov/brfss/annual_data/2021/pdf/codebook21_llcp-v2-508.pdf)
- [Calculated Variables in Data Files CDC](https://www.cdc.gov/brfss/annual_data/2021/pdf/2021-calculated-variables-version4-508.pdf)
- [Summary Matrix of Calculated Variables (CV) in the 2021 Data File](https://www.cdc.gov/brfss/annual_data/2021/summary_matrix_21.html)
- [Variable Layout](https://www.cdc.gov/brfss/annual_data/2021/llcp_varlayout_21_onecolumn.html)

Other analysis - Cleaned 2015 files:
- [CDC Diabetes Health Indicators (2015)](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
    - [Diabetes Health Indicators Dataset (2015)](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)


References:
- [National Diabetes Statistics Report](https://www.cdc.gov/diabetes/php/data-research/index.html) - See Risk Factors towards the end of the report
- [Building Risk Prediction Models for Type 2 Diabetes Using Machine Learning Techniques](https://www.cdc.gov/pcd/issues/2019/19_0109.htm)
- [Original uci.edu referenced dataset](https://www.kaggle.com/code/alexteboul/diabetes-health-indicators-dataset-notebook)

### Risk Factors

There are many risk factors for developing type 2 diabetes, including:
- Age: Being over 40 increases your risk.
- Family history: Having a parent, sibling, or other relative with type 1 or type 2 diabetes increases your risk.
- Ethnicity: People of certain races and ethnicities, including African Americans, Hispanics, American Indians, and Asian-Americans, are more likely to develop type 2 diabetes.
- Inactivity: The less active you are, the greater your risk.
- Weight: Being overweight or obese increases your risk. You can estimate your risk by measuring your waist circumference. Men have a higher risk if their waist circumference is more than 40 inches, while women who are not pregnant have a higher risk if their waist circumference is more than 35 inches.
- Blood pressure: High blood pressure can lead to insulin resistance and eventually type 2 diabetes.
- Cholesterol: High cholesterol can raise your risk for diabetes and heart disease.
- Smoking: Smokers are more 30-40% more likely than non-smokers to develop type 2 diabetes. 

### Indicators

Diabetes is a chronic condition that can be diagnosed by a medical professional. While it often has no symptoms, some indicators include:
- Urination: Frequent urination, especially at night
- Thirst: Excessive thirst
- Hunger: Increased hunger, even when eating
- Weight loss: Unintentional weight loss
- Fatigue: Feeling more tired than usual
- Vision: Blurred vision
- Wounds: Cuts and bruises that take longer to heal
- Skin: Itchy skin or genital itching
- Infections: Urinary tract infections (UTIs) or yeast infections
- Sensations: Unusual sensations like tingling, burning, or pricklin







## Other Studies

### Risk Factors
- Risk Factors
    - Smoking
    - Overweight
    - Physical inactivity
    - A1C
    - High Blook Pressure
    - High Cholesterol
- Preventing complications
    - diabetes care
    - physical activity
    - weight management
    - statin treatment
    - A1C, blood pressure, cholesterol, smoking (ABCs)
    - Vaccinations
- Co-existing conditions
    - Kidney disease
    - Vision disability

#### Original uci.edu referenced dataset noted risk factors

- blood pressure (high)
- cholesterol (high)
- smoking
- diabetes
- obesity
- age
- sex
- race
- diet
- exercise
- alcohol consumption
- BMI
- Household Income
- Marital Status
- Sleep
- Time since last checkup
- Education
- Health care coverage
- Mental Health



### Categories

- Target:
    - (Ever told) you had diabetes (DIABETE4)
    - Ever been told by a doctor or other health professional that you have pre-diabetes or borderline diabetes? (PREDIAB2)
    - According to your doctor or other health professional, what type of diabetes do you have? (DIABTYPE)

- Independent Variables:
    - Health Indicators:
        - Computed body mass index categories (_BMI5CAT)
    - Other chronic Health Conditions:
        - Ever Diagnosed with a Stroke (CVDSTRK3)
        - Ever Diagnosed with Angina or Coronary Heart Disease (CVDCRHD4)
        - Respondents that have ever reported having coronary heart disease (CHD) or myocardial infarction (MI) (_MICHD)
    - Physical Activity:
        - Leisure Time Physical Activity Calculated Variable (_TOTINDA)
    - Diet:
        - NONE
    - Alchohol:
        Computed number of drinks of alcohol beverages per week (_DRNKWK2)
    - Smoker:
        - Smoked at Least 100 Cigarettes (SMOKE100)
        - Four-level smoker status: Everyday smoker, Someday smoker, Former smoker, Non-smoker (_SMOKER3)
    - Health Care:
        - Have Personal Health Care Provider? (PERSDOC3)
        - What is Primary Source of Health Insurance? (PRIMINSR)
        - About how long has it been since you last visited a doctor for a routine checkup? (CHECKUP1)
    - General Health & Mental Health:
        - Would you say that in general your health is: (GENHLTH)
        - Number of Days Physical Health Not Good (0..30) (PHYSHLTH)
        - Number of Days Mental Health Not Good (MENTHLTH)
        - Have you experienced confusion or memory loss that is happening more often or is getting worse? (CIMEMLOS)
        - Adults with good or better health (_RFHLTH)
    - Demographics
        - Sex of Respondent: (_SEX)
        - Fourteen-level age category (_AGEG5YR)
        - Highest Level of education completed (EDUCA)
        - Annual Household income (INCOME3)
        - Urban / Rural (MSCODE)
        - Imputed race/ethnicity value (_IMPRACE)
        - Computed Preferred Race (_PRACE2)


Diabetes is a chronic condition that can be diagnosed by a medical professional. While it often has no symptoms, some indicators include:
- Urination: Frequent urination, especially at night
- Thirst: Excessive thirst
- Hunger: Increased hunger, even when eating
- Weight loss: Unintentional weight loss
- Fatigue: Feeling more tired than usual
- Vision: Blurred vision
- Wounds: Cuts and bruises that take longer to heal
- Skin: Itchy skin or genital itching
- Infections: Urinary tract infections (UTIs) or yeast infections
- Sensations: Unusual sensations like tingling, burning, or pricklin


There are many risk factors for developing type 2 diabetes, including:
- Age: Being over 40 increases your risk.
- Family history: Having a parent, sibling, or other relative with type 1 or type 2 diabetes increases your risk.
- Ethnicity: People of certain races and ethnicities, including African Americans, Hispanics, American Indians, and Asian-Americans, are more likely to develop type 2 diabetes.
- Inactivity: The less active you are, the greater your risk.
- Weight: Being overweight or obese increases your risk. You can estimate your risk by measuring your waist circumference. Men have a higher risk if their waist circumference is more than 40 inches, while women who are not pregnant have a higher risk if their waist circumference is more than 35 inches.
- Blood pressure: High blood pressure can lead to insulin resistance and eventually type 2 diabetes.
- Cholesterol: High cholesterol can raise your risk for diabetes and heart disease.
- Smoking: Smokers are more 30-40% more likely than non-smokers to develop type 2 diabetes. 
