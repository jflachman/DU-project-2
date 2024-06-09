# DU-project-2

## Diabetes Prediction from CDC Behavioral Risk Factor Surveillance System (BRFSS) Survey data
- See the [Project Instructions](project-2-overview.md) for more details about the project requirements


## Team
 
 - Jeff Flachman
 - Ava Lee
 - Elia Porter

# Executive Summary

This project aims to analyze the factors contributing to the prevalence of diabetes in the United State and determine if they provide some predictive value in determining a diagnosis of diabetes.

A dataset pulled from the 2015 BRFSS was available on the [**UC Irvine Machine Learning Repository**](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators).  This dataset was cleaned from the [2015 BRFSS survey](https://www.cdc.gov/brfss/annual_data/annual_2015.html) data.  The team also pulled an cleaned data from the [2021 BRFSS survey](https://www.cdc.gov/brfss/annual_data/annual_2021.html)


# Project Overview

Diabetes is the eighth leading cause of death in the United Stages.  But many rank it second behind heart disease as a chronic illness that leads to death.  Diabetes also has a daily implact of those who live with it.

The team was interested in diabetes predictions using data from the The CDC Behavioral Risk Factor Surveillance System (BRFSS).  The BRFSS is an annual phone survey of 300K-400K respondents.  

A dataset pulled from the 2015 BRFSS was available on the [**UC Irvine Machine Learning Repository**](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators).  This dataset had already been cleaned from the [2015 BRFSS survey](https://www.cdc.gov/brfss/annual_data/annual_2015.html) data.  The 2015 dataset was older and already cleaned.  Therefore, the team also pulled and cleaned the [2021 BRFSS survey](https://www.cdc.gov/brfss/annual_data/annual_2021.html).

Diabetes ***risk factors*** listed below were used to select survey results from BRFSS dataset.  The 2015 dataset contained 21 features.  The team selected 35 features from the 2021 dataset.

Two targets were evaluated:  

- **target 1** (0,1,2): 0: no diabetes, 1: pre-diabetes, 2: diabetes
- **target 2** (binary:0,1): 0: no diabetes, 1: diabetes

Classification models were trained and the metrics were computed.  In addition, alternate scaling and sampling techniques were used to handle the imbalance in the datasets.

In all, 63 configurations of binary/012, scaling & sampling method, and models were trained and the metrics were computed for each base dataset (2015 & 2021).

The metrics were evaluated and a few targeted dataset configurations were selected to optimize.

## Potential Datasets Evaluated

The team brainstormed multiple dataset options for this project.  Some of the datasets reviewed are listed in the [datasets](data_sets.md) files.

# Project Details:

## Ideation

## Feature Selection

**Risk Factors**

There are many risk factors for developing type 2 diabetes, including:

- **Age:** Being over 40 increases your risk.
- **Family history**: Having a parent, sibling, or other relative with type 1 or type 2 diabetes increases your risk.
- **Ethnicity**: People of certain races and ethnicities, including African Americans, Hispanics, American Indians, and Asian-Americans, are more likely to develop type 2 diabetes.
- **Inactivity**: The less active you are, the greater your risk.
- **Weight**: Being overweight or obese increases your risk. You can estimate your risk by measuring your waist circumference. Men have a higher risk if their waist circumference is more than 40 inches, while women who are not pregnant have a higher risk if their waist circumference is more than 35 inches.
- **Blood pressure**: High blood pressure can lead to insulin resistance and eventually type 2 diabetes.
- **Cholesterol**: High cholesterol can raise your risk for diabetes and heart disease.
- **Smoking**: Smokers are more 30-40% more likely than non-smokers to develop type 2 diabetes. 

**Diabetes Indicators / Symptoms**

Diabetes is a chronic condition that can be diagnosed by a medical professional. While it often has no symptoms, some indicators include:

- **Urination**: Frequent urination, especially at night
- **Thirst**: Excessive thirst
- **Hunger**: Increased hunger, even when eating
- **Weight loss**: Unintentional weight loss
- **Fatigue**: Feeling more tired than usual
- **Vision**: Blurred vision
- **Wounds**: Cuts and bruises that take longer to heal
- **Skin**: Itchy skin or genital itching
- **Infections**: Urinary tract infections (UTIs) or yeast infections
- **Sensations**: Unusual sensations like tingling, burning, or pricklin

## Data Cleaning

## Data Analysis

## Initial Conclusions


## Optimization / Hyperparameter tuning






## CDC Diabetes data selected


## Project Approach:

- Look at diabetes data from 2015 with 21 different features to see how well it predicts diabetes
- Check to see if the data is imbalanced; and see how different balancing methods work 
- Evaluate the dataset that has prediabetes and see if prediabetes can also be predicted from the features
- ** if time allows we will look at original source CDC data for different years and see if we can repeat that analysis** (bonus)

**See** the more detailed [project plan](project_plan.md)



---
---
---
---

- Project ideation – due by: 05/30/24
    - We looked at abalone, mushroom, bike sharing, and diabetes;
    - We decided to dig deeper in diabetes research looking into what early factors could cause diabetes 
    - Data fetching 
    - 06/03/24
    - Pull data sets in each of our machines and start basis analysis
- Data exploration
- 06/03/24 – rough draft 
- Data transformation & automation
- 06/04 – 06/06/24 
- Creating functions and stringing them all together 
- Data analysis
    - 06/06/24
        - Review what features have the most impact
        - Testing
- 06/08/24
    - Creating documentation
- 06/08 – 06/09
    - Creating the presentation

It was determined that the 2015 data had already been cleaned.  Discussions with the instructor would 







the the We have chose the topic of diabetes.
[CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)


## Project Overview

The Diabetes Health Indicators Dataset contains healthcare statistics and lifestyle survey information about people and their diabetes diagnosis. It includes 35 features, such as demographics, lab test results, and answers to survey questions for each patient. The target variable for classification indicates whether a patient has diabetes, is pre-diabetic, or is healthy.
### Executive Summary


### Selected Dataset

The selected dataset is the 2021 CDC dataset.
We have concentrated our efforts on the following risk factors for diabetes.

***Risk Factors*** 
 
   There are many risk factors for developing type 2 diabetes, including:


There are many risk factors for developing type 2 diabetes, including:
- **Age:** Being over 40 increases your risk.
- **Family history**: Having a parent, sibling, or other relative with type 1 or type 2 diabetes increases your risk.
- **Ethnicity**: People of certain races and ethnicities, including African Americans, Hispanics, American Indians, and Asian-Americans, are more likely to develop type 2 diabetes.
- **Inactivity**: The less active you are, the greater your risk.
- **Weight**: Being overweight or obese increases your risk. You can estimate your risk by measuring your waist circumference. Men have a higher risk if their waist circumference is more than 40 inches, while women who are not pregnant have a higher risk if their waist circumference is more than 35 inches.
- **Blood pressure**: High blood pressure can lead to insulin resistance and eventually type 2 diabetes.
- **Cholesterol**: High cholesterol can raise your risk for diabetes and heart disease.
- **Smoking**: Smokers are more 30-40% more likely than non-smokers to develop type 2 diabetes. 

### Indicators

Diabetes is a chronic condition that can be diagnosed by a medical professional. While it often has no symptoms, some indicators include:
- **Urination**: Frequent urination, especially at night
- **Thirst**: Excessive thirst
- **Hunger**: Increased hunger, even when eating
- **Weight loss**: Unintentional weight loss
- **Fatigue**: Feeling more tired than usual
- **Vision**: Blurred vision
- **Wounds**: Cuts and bruises that take longer to heal
- **Skin**: Itchy skin or genital itching
- **Infections**: Urinary tract infections (UTIs) or yeast infections
- **Sensations**: Unusual sensations like tingling, burning, or pricklin



Please see the following link for more details: [data_cleaning.md](data_cleaning.md)

### Approach

In our efforts, we focused on the following risk factors for diabetes. Below are the selected feature abbreviations.
[diabetes_features.md](diabetes_features.md)

### DataCleaning 

- Jeff will write

### Metrics

![Metrics](https://images.datacamp.com/image/upload/v1701364260/image_d6ced554a1.png)

# What can we write on:


    

    For Analysis you can look at [analysis.md](analysis.md)



- The Plan
    - Run models on basic data
    - 
