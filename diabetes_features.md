# **Diabetes Features**

The features listed in this file are Diabetes Risk Factors and Indicators.

- **Target**: DIABETE4

# **Summary**

| Feature | Label | Description |
|:-----|:-----|:-----|
ot| **GENHLTH** | General Health | Would you say that in general your health is:  |
| **PHYSHLTH** | Number of Days Physical Health Not Good | Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good? |
| **BPHIGH6** | Ever Told Blood Pressure High | Have you ever been told by a doctor, nurse or other health professional that you have high blood pressure? (If ´Yes´ and respondent is female, ask ´Was this only when you were pregnant?´.) |
| **CHOLCHK3** | How Long since Cholesterol Checked | About how long has it been since you last had your cholesterol checked? |
| **TOLDHI3** | Ever Told Cholesterol Is High | Have you ever been told by a doctor, nurse or other health professional that your cholesterol is high? |
| **CVDSTRK3** | Ever Diagnosed with a Stroke | (Ever told) (you had) a stroke. |
| **DIABETE4** | (Ever told) you had diabetes | (Ever told) (you had) diabetes? (If ´Yes´ and respondent is female, ask ´Was this only when you were pregnant?´. If Respondent says pre-diabetes or borderline diabetes, use response code 4.) |
| **EDUCA** | Education Level | What is the highest grade or year of school you completed? |
| **INCOME3** | Income Level | Is your annual household income from all sources: (If respondent refuses at any income level, code ´Refused.´) |
| **DIFFWALK** | Difficulty Walking or Climbing Stairs | Do you have serious difficulty walking or climbing stairs? |
| **TOLDCFS** | Told had Chronic Fatigue Syndrome (CFS) or (Myalgic Encephalomyelitis) ME | Have you ever been told by a doctor or other health professional that you had Chronic Fatigue Syndrome (CFS) or (Myalgic Encephalomyelitis) ME? |
| **_HLTHPLN** | Have any health insurance | Adults who had some form of health insurance |
| **_TOTINDA** | Leisure Time Physical Activity Calculated Variable | Adults who reported doing physical activity or exercise during the past 30 days other than their regular job |
| **_MICHD** | Ever had CHD or MI‌ | Respondents that have ever reported having coronary heart disease (CHD) or myocardial infarction (MI) |
| **_SEX** | Calculated sex variable | Calculated sex variable |
| **_AGEG5YR** | Reported age in five-year age categories calculated variable | Fourteen-level age category |
| **_BMI5** | Computed body mass index | Body Mass Index (BMI) |
| **_SMOKER3** | Computed Smoking Status | Four-level smoker status: Everyday smoker, Someday smoker, Former smoker, Non-smoker |
| **_RFDRHV7** | Heavy Alcohol Consumption Calculated Variable | Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week) |
| **FTJUDA2_** | Computed Fruit Juice intake in times per day | Fruit juice intake in times per day |
| **FRUTDA2_** | Computed Fruit intake in times per day | Fruit intake in times per day |
| **GRENDA1_** | Computed Dark Green Vege | Dark green vege: table intake in times per day |
| **FRNCHDA_** | Computed French Fry intake in times per day | French Fry intake in times per day |
| **POTADA1_** | Computed Potato Servings per day | Potato servings per day |
| **VEGEDA2_** | Computed Other Vege | Other vege: table intake in times per day |
| **_FRUTSU1** | Total fruits consumed per day | Total fruits consumed per day |
| **_VEGESU1** | Total vege | Total vege: tables consumed per day |
| **_FRTLT1A** | Consume Fruit 1 or more times per day | Consume Fruit 1 or more times per day |

<br><br><br>

# **Feature Details**
The following sections describe the features in detail.  The information includes parameters about the features as well as a table that describes the values


<br><br>

## **General Health**

---
<br>

|  Label  |  General Health |
|:-----|:-----|
|  Section Name  |  Health Status Core |
|  Section Number  |  1 |
|  Question Number  |  1 |
|  Column  |  101 |
|  Type of Variable  |  Num |
|  SAS Variable Name  |  GENHLTH |
|  Question Prologue  |   |
|  Question  |  Would you say that in general your health is:  |
<br>

### Feature Data Table

|    | Value   | Value Label          |   Frequency | Percentage   | Weighted Percentage   |
|---:|:--------|:---------------------|------------:|:-------------|:----------------------|
|  2 | 1       | Excellent            |       77741 | 17.72        | 19.24                 |
|  3 | 2       | Very good            |      149112 | 33.99        | 32.39                 |
|  4 | 3       | Good                 |      137938 | 31.44        | 31.91                 |
|  5 | 4       | Fair                 |       54736 | 12.48        | 12.38                 |
|  6 | 5       | Poor                 |       18005 | 4.10         | 3.82                  |
|  7 | 7       | Don’t know/Not Sure  |         788 | 0.18         | 0.18                  |
|  8 | 9       | Refused              |         369 | 0.08         | 0.08                  |
|  9 | BLANK   | Not asked or Missing |           4 | .            | .                     |
---


<br><br>

## **Number of Days Physical Health Not Good**

---
<br>

|  Label  |  Number of Days Physical Health Not Good |
|:-----|:-----|
|  Section Name  |  Healthy Days‌Core |
|  Section Number  |  2 |
|  Question Number  |  1 |
|  Column  |  102-103 |
|  Type of Variable  |  Num |
|  SAS Variable Name  |  PHYSHLTH |
|  Question Prologue  |   |
|  Question  |  Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good? |
<br>

### Feature Data Table

|    | Value   | Value Label          |   Frequency | Percentage   | Weighted Percentage   |
|---:|:--------|:---------------------|------------:|:-------------|:----------------------|
|  2 | 1 - 30  | Number of days       |      141403 | 32.23        | 31.47                 |
|  3 | 88      | nan                  |      287796 | 65.60        | 66.53                 |
|  4 | 77      | Don’t know/Not sure  |        7898 | 1.80         | 1.66                  |
|  5 | 99      | Refused              |        1593 | 0.36         | 0.34                  |
|  6 | BLANK   | Not asked or Missing |           3 | .            | .                     |
---


<br><br>

## **Ever Told Blood Pressure High**

---
<br>

|  Label  |  Ever Told Blood Pressure High |
|:-----|:-----|
|  Section Name  |  Hypertension Awareness Core |
|  Section Number  |  5‌ |
|  Question Number  |  1 |
|  Column  |  114 |
|  Type of Variable  |  Num |
|  SAS Variable Name  |  BPHIGH6 |
|  Question Prologue  |   |
|  Question  |  Have you ever been told by a doctor, nurse or other health professional that you have high blood pressure? (If ´Yes´ and respondent is female, ask ´Was this only when you were pregnant?´.) |
<br>

### Feature Data Table

|    | Value   | Value Label                                                                                      |   Frequency | Percentage   | Weighted Percentage   |
|---:|:--------|:-------------------------------------------------------------------------------------------------|------------:|:-------------|:----------------------|
|  2 | 1       | Yes                                                                                              |      172133 | 39.24        | 32.65                 |
|  3 | 2       | Yes, but female told only during pregnancy—Go to Section 06.01 CHOLCHK3                          |        3474 | 0.79         | 0.88                  |
|  4 | 3       | No—Go to Section 06.01 CHOLCHK3                                                                  |      256603 | 58.49        | 65.12                 |
|  5 | 4       | Told borderline high or pre-hypertensive or elevated blood pressure—Go to Section 06.01 CHOLCHK3 |        4571 | 1.04         | 0.91                  |
|  6 | 7       | Don´t know/Not Sure—Go to Section 06.01 CHOLCHK3                                                 |        1191 | 0.27         | 0.30                  |
|  7 | 9       | Refused—Go to Section 06.01 CHOLCHK3                                                             |         719 | 0.16         | 0.14                  |
|  8 | BLANK   | Not asked or Missing                                                                             |           2 | .            | .                     |
---


<br><br>

## **How Long since Cholesterol Checked**

---
<br>

|  Label  |  How Long since Cholesterol Checked |
|:-----|:-----|
|  Section Name  |  Cholesterol Awareness Core |
|  Section Number  |  6‌ |
|  Question Number  |  1 |
|  Column  |  116 |
|  Type of Variable  |  Num |
|  SAS Variable Name  |  CHOLCHK3 |
|  Question Prologue  |  Cholesterol is a fatty substance found in the blood. |
|  Question  |  About how long has it been since you last had your cholesterol checked? |
<br>

### Feature Data Table

|    | Value   | Value Label                                         |   Frequency | Percentage   | Weighted Percentage   |
|---:|:--------|:----------------------------------------------------|------------:|:-------------|:----------------------|
|  2 | 1       | Never—Go to Section 07.01 CVDINFR4                  |       31369 | 7.15         | 8.98                  |
|  3 | 2       | Within the past year (anytime < one year ago)       |      289789 | 66.06        | 60.48                 |
|  4 | 3       | Within the past 2 years (1 year but < 2 years ago)  |       48223 | 10.99        | 12.04                 |
|  5 | 4       | Within the past 3 years (2 years but < 3 years ago) |       15422 | 3.52         | 4.06                  |
|  6 | 5       | Within the past 4 years (3 years but < 4 years ago) |        5543 | 1.26         | 1.47                  |
|  7 | 6       | Within the past 5 years (4 years but < 5 years ago) |        5025 | 1.15         | 1.32                  |
|  8 | 7       | Don’t know/Not Sure—Go to Section 07.01 CVDINFR4    |       29013 | 6.61         | 8.05                  |
|  9 | 8       | 5 or more years ago                                 |       13540 | 3.09         | 3.44                  |
| 10 | 9       | Refused—Go to Section 07.01 CVDINFR4                |         767 | 0.17         | 0.16                  |
| 11 | BLANK   | Not asked or Missing                                |           2 | .            | .                     |
---


<br><br>

## **Ever Told Cholesterol Is High**

---
<br>

|  Label  |  Ever Told Cholesterol Is High |
|:-----|:-----|
|  Section Name  |  Cholesterol Awareness Core |
|  Section Number  |  6 |
|  Question Number  |  2 |
|  Column  |  117 |
|  Type of Variable  |  Num |
|  SAS Variable Name  |  TOLDHI3 |
|  Question Prologue  |   |
|  Question  |  Have you ever been told by a doctor, nurse or other health professional that your cholesterol is high? |
<br>

### Feature Data Table

|    | Value   | Value Label                                                                     |   Frequency | Percentage   | Weighted Percentage   |
|---:|:--------|:--------------------------------------------------------------------------------|------------:|:-------------|:----------------------|
|  2 | 1       | Yes                                                                             |      149724 | 39.62        | 35.27                 |
|  3 | 2       | No                                                                              |      224989 | 59.54        | 63.98                 |
|  4 | 7       | Don’t know/Not Sure                                                             |        2689 | 0.71         | 0.65                  |
|  5 | 9       | Refused                                                                         |         455 | 0.12         | 0.10                  |
|  6 | BLANK   | Not asked or MissingNotes: Section 06.01, CHOLCHK3, is coded 1, 7,9, or Missing |       60836 | .            | .                     |
---


<br><br>

## **Ever Diagnosed with a Stroke**

---
<br>

|  Label  |  Ever Diagnosed with a Stroke |
|:-----|:-----|
|  Section Name  |  Chronic Health Conditions Core |
|  Section Number  |  7 |
|  Question Number  |  3 |
|  Column  |  121 |
|  Type of Variable  |  Num |
|  SAS Variable Name  |  CVDSTRK3 |
|  Question Prologue  |   |
|  Question  |  (Ever told) (you had) a stroke. |
<br>

### Feature Data Table

|    | Value   | Value Label          |   Frequency | Percentage   | Weighted Percentage   |
|---:|:--------|:---------------------|------------:|:-------------|:----------------------|
|  2 | 1       | Yes                  |       17213 | 3.92         | 3.25                  |
|  3 | 2       | No                   |      420051 | 95.75        | 96.47                 |
|  4 | 7       | Don’t know/Not sure  |        1130 | 0.26         | 0.22                  |
|  5 | 9       | Refused              |         297 | 0.07         | 0.06                  |
|  6 | BLANK   | Not asked or Missing |           2 | .            | .                     |
---


<br><br>

## **(Ever told) you had diabetes**

---
<br>

|  Label  |  (Ever told) you had diabetes |
|:-----|:-----|
|  Section Name  |  Chronic Health Conditions Core |
|  Section Number  |  7 |
|  Question Number  |  11 |
|  Column  |  129 |
|  Type of Variable  |  Num |
|  SAS Variable Name  |  DIABETE4 |
|  Question Prologue  |   |
|  Question  |  (Ever told) (you had) diabetes? (If ´Yes´ and respondent is female, ask ´Was this only when you were pregnant?´. If Respondent says pre-diabetes or borderline diabetes, use response code 4.) |
<br>

### Feature Data Table

|    | Value   | Value Label                                                             |   Frequency | Percentage   | Weighted Percentage   |
|---:|:--------|:------------------------------------------------------------------------|------------:|:-------------|:----------------------|
|  2 | 1       | Yes                                                                     |       57616 | 13.13        | 11.36                 |
|  3 | 2       | Yes, but female told only during pregnancy—Go to Section 08.01 HAVARTH5 |        3808 | 0.87         | 0.99                  |
|  4 | 3       | No—Go to Section 08.01 HAVARTH5                                         |      366342 | 83.51        | 85.04                 |
|  5 | 4       | No, pre-diabetes or borderline diabetes—Go to Section 08.01 HAVARTH5    |        9942 | 2.27         | 2.40                  |
|  6 | 7       | Don’t know/Not Sure—Go to Section 08.01 HAVARTH5                        |         613 | 0.14         | 0.15                  |
|  7 | 9       | Refused—Go to Section 08.01 HAVARTH5                                    |         369 | 0.08         | 0.07                  |
|  8 | BLANK   | Not asked or Missing                                                    |           3 | .            | .                     |
---


<br><br>

## **Education Level**

---
<br>

|  Label  |  Education Level |
|:-----|:-----|
|  Section Name  |  Demographics Core |
|  Section Number  |  9 |
|  Question Number  |  6‌ |
|  Column  |  176 |
|  Type of Variable  |  Num |
|  SAS Variable Name  |  EDUCA |
|  Question Prologue  |   |
|  Question  |  What is the highest grade or year of school you completed? |
<br>

### Feature Data Table

|    | Value   | Value Label                                                  |   Frequency | Percentage   | Weighted Percentage   |
|---:|:--------|:-------------------------------------------------------------|------------:|:-------------|:----------------------|
|  2 | 1       | Never attended school or only kindergarten                   |         606 | 0.14         | 0.32                  |
|  3 | 2       | Grades 1 through 8 (Elementary)                              |        8259 | 1.88         | 4.03                  |
|  4 | 3       | Grades 9 through 11 (Some high school)                       |       17126 | 3.90         | 7.57                  |
|  5 | 4       | Grade 12 or GED (High school graduate)                       |      111545 | 25.43        | 27.36                 |
|  6 | 5       | College 1 year to 3 years (Some college or technical school) |      120102 | 27.38        | 30.17                 |
|  7 | 6       | College 4 years or more (College graduate)                   |      178577 | 40.71        | 29.95                 |
|  8 | 9       | Refused                                                      |        2473 | 0.56         | 0.59                  |
|  9 | BLANK   | Not asked or Missing                                         |           5 | .            | .                     |
---


<br><br>

## **Income Level**

---
<br>

|  Label  |  Income Level |
|:-----|:-----|
|  Section Name  |  Demographics Core |
|  Section Number  |  9 |
|  Question Number  |  16 |
|  Column  |  193-194 |
|  Type of Variable  |  Num |
|  SAS Variable Name  |  INCOME3 |
|  Question Prologue  |   |
|  Question  |  Is your annual household income from all sources: (If respondent refuses at any income level, code ´Refused.´) |
<br>

### Feature Data Table

|    | Value   | Value Label                                   |   Frequency | Percentage   | Weighted Percentage   |
|---:|:--------|:----------------------------------------------|------------:|:-------------|:----------------------|
|  2 | 1       | Less than $10,000                             |       10878 | 2.53         | 3.30                  |
|  3 | 2       | Less than $15,000 ($10,000 to < $15,000)      |       11530 | 2.68         | 2.87                  |
|  4 | 3       | Less than $20,000 ($15,000 to < $20,000)      |       14960 | 3.48         | 3.65                  |
|  5 | 4       | Less than $25,000 ($20,000 to < $25,000)      |       21071 | 4.90         | 4.88                  |
|  6 | 5       | Less than $35,000 ($25,000 to < $35,000)      |       43893 | 10.21        | 9.99                  |
|  7 | 6       | Less than $50,000 ($35,000 to < $50,000)      |       48339 | 11.25        | 10.37                 |
|  8 | 7       | Less than $75,000 ($50,000 to < $75,000)      |       59408 | 13.82        | 12.67                 |
|  9 | 8       | Less than $100,000? ($75,000 to < $100,000)   |       47838 | 11.13        | 10.47                 |
| 10 | 9       | Less than $150,000? ($100,000 to < $150,000)? |       47642 | 11.08        | 11.07                 |
| 11 | 10      | Less than $200,000? ($150,000 to < $200,000)  |       19769 | 4.60         | 5.00                  |
| 12 | 11      | $200,000 or more                              |       18952 | 4.41         | 5.41                  |
| 13 | 77      | Don’t know/Not sure                           |       36138 | 8.41         | 9.58                  |
| 14 | 99      | Refused                                       |       49428 | 11.50        | 10.76                 |
| 15 | BLANK   | Not asked or Missing                          |        8847 | .            | .                     |
---


<br><br>

## **Difficulty Walking or Climbing Stairs**

---
<br>

|  Label  |  Difficulty Walking or Climbing Stairs |
|:-----|:-----|
|  Section Name  |  Disability‌Core |
|  Section Number  |  10 |
|  Question Number  |  4 |
|  Column  |  207 |
|  Type of Variable  |  Num |
|  SAS Variable Name  |  DIFFWALK |
|  Question Prologue  |   |
|  Question  |  Do you have serious difficulty walking or climbing stairs? |
<br>

### Feature Data Table

|    | Value   | Value Label          |   Frequency | Percentage   | Weighted Percentage   |
|---:|:--------|:---------------------|------------:|:-------------|:----------------------|
|  2 | 1       | Yes                  |       67557 | 16.06        | 13.53                 |
|  3 | 2       | No                   |      351306 | 83.51        | 86.12                 |
|  4 | 7       | Don’t know/Not Sure  |        1190 | 0.28         | 0.22                  |
|  5 | 9       | Refused              |         631 | 0.15         | 0.14                  |
|  6 | BLANK   | Not asked or Missing |       18009 | .            | .                     |
---


<br><br>

## **Told had Chronic Fatigue Syndrome (CFS) or (Myalgic Encephalomyelitis) ME**

---
<br>

|  Label  |  Told had Chronic Fatigue Syndrome (CFS) or (Myalgic Encephalomyelitis) ME |
|:-----|:-----|
|  Section Name  |  ME/CFS‌‌ |
|  Module Number  |  3 |
|  Question Number  |  1 |
|  Column  |  276 |
|  Type of Variable  |  Num |
|  SAS Variable Name  |  TOLDCFS |
|  Question Prologue  |   |
|  Question  |  Have you ever been told by a doctor or other health professional that you had Chronic Fatigue Syndrome (CFS) or (Myalgic Encephalomyelitis) ME? |
<br>

### Feature Data Table

|    | Value   | Value Label          |   Frequency | Percentage   | Weighted Percentage   |
|---:|:--------|:---------------------|------------:|:-------------|:----------------------|
|  2 | BLANK   | Not asked or Missing |      438693 | .            | .                     |
---


<br><br>

## **Have any health insurance**

---
<br>

|  Label  |  Have any health insurance |
|:-----|:-----|
|  Section Name  |  Calculated Variables |
|  Module Number  |  3‌ |
|  Question Number  |  1 |
|  Column  |  1902 |
|  Type of Variable  |  Num |
|  SAS Variable Name  |  _HLTHPLN |
|  Question Prologue  |   |
|  Question  |  Adults who had some form of health insurance |
<br>

### Feature Data Table

|    |   Value | Value Label                                                                         |   Frequency |   Percentage |   Weighted Percentage |
|---:|--------:|:------------------------------------------------------------------------------------|------------:|-------------:|----------------------:|
|  2 |       1 | Have some form of insuranceNotes: PRIMINSR=1, 2, 3, 4, 5, 6, 7, 8, 9, 10            |      398081 |        90.74 |                 87.15 |
|  3 |       2 | Do not have some form of health insurance Notes: PRIMINSR=88                        |       23215 |         5.29 |                  8.19 |
|  4 |       9 | Don´t know, refused or missing insurance responseNotes: PRIMINSR=77, 99, or missing |       17397 |         3.97 |                  4.65 |
---


<br><br>

## **Leisure Time Physical Activity Calculated Variable**

---
<br>

|  Label  |  Leisure Time Physical Activity Calculated Variable |
|:-----|:-----|
|  Section Name  |  Calculated Variables‌ |
|  Module Number  |  4 |
|  Question Number  |  1 |
|  Column  |  1904 |
|  Type of Variable  |  Num |
|  SAS Variable Name  |  _TOTINDA |
|  Question Prologue  |   |
|  Question  |  Adults who reported doing physical activity or exercise during the past 30 days other than their regular job |
<br>

### Feature Data Table

|    |   Value | Value Label                                                          |   Frequency |   Percentage |   Weighted Percentage |
|---:|--------:|:---------------------------------------------------------------------|------------:|-------------:|----------------------:|
|  2 |       1 | Had physical activity or exercise Notes: EXERANY2 = 1                |      330738 |        75.39 |                 75.96 |
|  3 |       2 | No physical activity or exercise in last 30 days Notes: EXERANY2 = 2 |      107027 |        24.4  |                 23.87 |
|  4 |       9 | Don’t know/Refused/MissingNotes: EXERANY2 = 7 or 9 or Missing        |         928 |         0.21 |                  0.17 |
---


<br><br>

## **Ever had CHD or MI‌**

---
<br>

|  Label  |  Ever had CHD or MI‌ |
|:-----|:-----|
|  Section Name  |  Calculated Variables |
|  Module Number  |  7 |
|  Question Number  |  1 |
|  Column  |  1908 |
|  Type of Variable  |  Num |
|  SAS Variable Name  |  _MICHD |
|  Question Prologue  |   |
|  Question  |  Respondents that have ever reported having coronary heart disease (CHD) or myocardial infarction (MI) |
<br>

### Feature Data Table

|    | Value   | Value Label                                                                      |   Frequency | Percentage   | Weighted Percentage   |
|---:|:--------|:---------------------------------------------------------------------------------|------------:|:-------------|:----------------------|
|  2 | 1       | Reported having MI or CHD Notes: CVDINFR4=1 OR CVDCRHD4=1                        |       35323 | 8.14         | 6.13                  |
|  3 | 2       | Did not report having MI or CHD Notes: CVDINFR4=2 AND CVDCRHD4=2                 |      398735 | 91.86        | 93.87                 |
|  4 | BLANK   | Not asked or MissingNotes: CVDINFR4=7, 9 OR MISSING OR CVDCRHD4=7, 9, OR MISSING |        4635 | .            | .                     |
---


<br><br>

## **Calculated sex variable**

---
<br>

|  Label  |  Calculated sex variable |
|:-----|:-----|
|  Section Name  |  Calculated Variables |
|  Module Number  |  9 |
|  Question Number  |  11 |
|  Column  |  1982 |
|  Type of Variable  |  Num |
|  SAS Variable Name  |  _SEX |
|  Question Prologue  |   |
|  Question  |  Calculated sex variable |
<br>

### Feature Data Table

|    |   Value | Value Label                                                  |   Frequency |   Percentage |   Weighted Percentage |
|---:|--------:|:-------------------------------------------------------------|------------:|-------------:|----------------------:|
|  2 |       1 | MaleNotes: BIRTHSEX=1 or BIRTHSEX notin (1,2) and SEXVAR=1   |      203760 |        46.45 |                 48.73 |
|  3 |       2 | FemaleNotes: BIRTHSEX=2 or BIRTHSEX notin (1,2) and SEXVAR=2 |      234933 |        53.55 |                 51.27 |
---


<br><br>

## **Reported age in five-year age categories calculated variable**

---
<br>

|  Label  |  Reported age in five-year age categories calculated variable |
|:-----|:-----|
|  Section Name  |  Calculated Variables |
|  Module Number  |  9 |
|  Question Number  |  12 |
|  Column  |  1983-1984 |
|  Type of Variable  |  Num |
|  SAS Variable Name  |  _AGEG5YR |
|  Question Prologue  |   |
|  Question  |  Fourteen-level age category |
<br>

### Feature Data Table

|    |   Value | Value Label                                     |   Frequency |   Percentage |   Weighted Percentage |
|---:|--------:|:------------------------------------------------|------------:|-------------:|----------------------:|
|  2 |       1 | Age 18 to 24Notes: 18 <= AGE <= 24              |       26025 |         5.93 |                 12.06 |
|  3 |       2 | Age 25 to 29Notes: 25 <= AGE <= 29              |       21623 |         4.93 |                  7.7  |
|  4 |       3 | Age 30 to 34Notes: 30 <= AGE <= 34              |       25724 |         5.86 |                  9.68 |
|  5 |       4 | Age 35 to 39Notes: 35 <= AGE <= 39              |       28559 |         6.51 |                  7.71 |
|  6 |       5 | Age 40 to 44Notes: 40 <= AGE <= 44              |       29435 |         6.71 |                  8.49 |
|  7 |       6 | Age 45 to 49Notes: 45 <= AGE <= 49              |       28391 |         6.47 |                  6.43 |
|  8 |       7 | Age 50 to 54Notes: 50 <= AGE <= 54              |       34086 |         7.77 |                  8    |
|  9 |       8 | Age 55 to 59Notes: 55 <= AGE <= 59              |       38009 |         8.66 |                  7.66 |
| 10 |       9 | Age 60 to 64Notes: 60 <= AGE <= 64              |       44298 |        10.1  |                  8.55 |
| 11 |      10 | Age 65 to 69Notes: 65 <= AGE <= 69              |       45762 |        10.43 |                  6.76 |
| 12 |      11 | Age 70 to 74Notes: 70 <= AGE <= 74              |       42920 |         9.78 |                  6.04 |
| 13 |      12 | Age 75 to 79Notes: 75 <= AGE <= 79              |       29747 |         6.78 |                  4.21 |
| 14 |      13 | Age 80 or olderNotes: 80 <= AGE <= 99           |       34507 |         7.87 |                  4.73 |
| 15 |      14 | Don’t know/Refused/Missing Notes: 7 <= AGE <= 9 |        9607 |         2.19 |                  1.99 |
---


<br><br>

## **Computed body mass index**

---
<br>

|  Label  |  Computed body mass index |
|:-----|:-----|
|  Section Name  |  Calculated Variables |
|  Module Number  |  9‌ |
|  Question Number  |  19 |
|  Column  |  2000-2003 |
|  Type of Variable  |  Num |
|  SAS Variable Name  |  _BMI5 |
|  Question Prologue  |   |
|  Question  |  Body Mass Index (BMI) |
<br>

### Feature Data Table

|    | Value    | Value Label                                                              |   Frequency | Percentage   | Weighted Percentage   |
|---:|:---------|:-------------------------------------------------------------------------|------------:|:-------------|:----------------------|
|  2 | 1 - 9999 | 1 or greaterNotes: WTKG3/(HTM4*HTM4) (Has 2 implied decimal places)      |      391841 | 100.00       | 100.00                |
|  3 | BLANK    | Don’t know/Refused/MissingNotes: WTKG3 = 777 or 999 or HTM4 = 777 or 999 |       46852 | .            | .                     |
---


<br><br>

## **Computed Smoking Status**

---
<br>

|  Label  |  Computed Smoking Status |
|:-----|:-----|
|  Section Name  |  Calculated Variables |
|  Module Number  |  11‌ |
|  Question Number  |  1 |
|  Column  |  2009 |
|  Type of Variable  |  Num |
|  SAS Variable Name  |  _SMOKER3 |
|  Question Prologue  |   |
|  Question  |  Four-level smoker status: Everyday smoker, Someday smoker, Former smoker, Non-smoker |
<br>

### Feature Data Table

|    |   Value | Value Label                                                                                   |   Frequency |   Percentage |   Weighted Percentage |
|---:|--------:|:----------------------------------------------------------------------------------------------|------------:|-------------:|----------------------:|
|  2 |       1 | Current smoker - now smokes every day Notes: SMOKE100 = 1 and SMOKEDAY = 1                    |       38913 |         8.87 |                  8.83 |
|  3 |       2 | Current smoker - now smokes some days Notes: SMOKE100 = 1 and SMOKEDAY = 2                    |       14919 |         3.4  |                  3.77 |
|  4 |       3 | Former smokerNotes: SMOKE100 = 1 and SMOKEDAY = 3                                             |      113247 |        25.81 |                 22.17 |
|  5 |       4 | Never smokedNotes: SMOKE100 = 2                                                               |      246644 |        56.22 |                 58.9  |
|  6 |       9 | Don’t know/Refused/MissingNotes: SMOKE100 = 1 and SMOKEDAY = 9 or SMOKE100= 7 or 9 or Missing |       24970 |         5.69 |                  6.32 |
---


<br><br>

## **Heavy Alcohol Consumption Calculated Variable**

---
<br>

|  Label  |  Heavy Alcohol Consumption Calculated Variable |
|:-----|:-----|
|  Section Name  |  Calculated Variables |
|  Module Number  |  12 |
|  Question Number  |  5 |
|  Column  |  2022 |
|  Type of Variable  |  Num |
|  SAS Variable Name  |  _RFDRHV7 |
|  Question Prologue  |   |
|  Question  |  Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week) |
<br>

### Feature Data Table

|    |   Value | Value Label                                                                                                       |   Frequency |   Percentage |   Weighted Percentage |
|---:|--------:|:------------------------------------------------------------------------------------------------------------------|------------:|-------------:|----------------------:|
|  2 |       1 | NoNotes: (SEXVAR=1 or BIRTHSEX=1) and _DRNKWK1 <=14 or (SEXVAR=2 or BIRTHSEX=2) and _DRNKWK1 <=7 or ALCDAY5 = 888 |      379828 |        86.58 |                 85.55 |
|  3 |       2 | YesNotes: (SEXVAR=1 or BIRTHSEX=1) and _DRNKWK1 > 14 or (SEXVAR=2 or BIRTHSEX=2) and _DRNKWK1 > 7                 |       23872 |         5.44 |                  5.46 |
|  4 |       9 | Don’t know/Refused/Missing Notes: _DRNKWK1 = 99900                                                                |       34993 |         7.98 |                  8.99 |
---


<br><br>

## **Computed Fruit Juice intake in times per day**

---
<br>

|  Label  |  Computed Fruit Juice intake in times per day |
|:-----|:-----|
|  Section Name  |  Calculated Variables |
|  Module Number  |  15 |
|  Question Number  |  1 |
|  Column  |  2026-2029 |
|  Type of Variable  |  Num |
|  SAS Variable Name  |  FTJUDA2_ |
|  Question Prologue  |   |
|  Question  |  Fruit juice intake in times per day |
<br>

### Feature Data Table

|    | Value    | Value Label                                |   Frequency | Percentage   | Weighted Percentage   |
|---:|:---------|:-------------------------------------------|------------:|:-------------|:----------------------|
|  2 | 0 - 9999 | Times per day (two implied decimal places) |      394344 | 100.00       | 100.00                |
|  3 | BLANK    | Don’t know/Not Sure Or Refused/Missing     |       44349 | .            | .                     |
---


<br><br>

## **Computed Fruit intake in times per day**

---
<br>

|  Label  |  Computed Fruit intake in times per day |
|:-----|:-----|
|  Section Name  |  Calculated Variables‌‌ |
|  Module Number  |  15 |
|  Question Number  |  2 |
|  Column  |  2030-2033 |
|  Type of Variable  |  Num |
|  SAS Variable Name  |  FRUTDA2_ |
|  Question Prologue  |   |
|  Question  |  Fruit intake in times per day |
<br>

### Feature Data Table

|    | Value    | Value Label                                |   Frequency | Percentage   | Weighted Percentage   |
|---:|:---------|:-------------------------------------------|------------:|:-------------|:----------------------|
|  2 | 0 - 9999 | Times per day (two implied decimal places) |      394742 | 100.00       | 100.00                |
|  3 | BLANK    | Don’t know/Not Sure Or Refused/Missing     |       43951 | .            | .                     |
---


<br><br>

## **Computed Dark Green Vege**

---
<br>

|  Label  |  Computed Dark Green Vege |
|:-----|:-----|
|  Section Name  |  Calculated Variables |
|  Module Number  |  15 |
|  Question Number  |  3 |
|  Column  |  2034-2037 |
|  Type of Variable  |  Num |
|  SAS Variable Name  |  GRENDA1_ |
|  Question Prologue  |   |
|  Question  |  Dark green vege: table intake in times per day |
<br>

### Feature Data Table

|    | Value    | Value Label                                |   Frequency | Percentage   | Weighted Percentage   |
|---:|:---------|:-------------------------------------------|------------:|:-------------|:----------------------|
|  2 | 0 - 9999 | Times per day (two implied decimal places) |      394443 | 100.00       | 100.00                |
|  3 | BLANK    | Don’t know/Not Sure Or Refused/Missing     |       44250 | .            | .                     |
---


<br><br>

## **Computed French Fry intake in times per day**

---
<br>

|  Label  |  Computed French Fry intake in times per day |
|:-----|:-----|
|  Section Name  |  Calculated Variables |
|  Module Number  |  15 |
|  Question Number  |  4 |
|  Column  |  2038-2041 |
|  Type of Variable  |  Num |
|  SAS Variable Name  |  FRNCHDA_ |
|  Question Prologue  |   |
|  Question  |  French Fry intake in times per day |
<br>

### Feature Data Table

|    | Value    | Value Label                                |   Frequency | Percentage   | Weighted Percentage   |
|---:|:---------|:-------------------------------------------|------------:|:-------------|:----------------------|
|  2 | 0 - 9999 | Times per day (two implied decimal places) |      393928 | 100.00       | 100.00                |
|  3 | BLANK    | Don’t know/Not Sure Or Refused/Missing     |       44765 | .            | .                     |
---


<br><br>

## **Computed Potato Servings per day**

---
<br>

|  Label  |  Computed Potato Servings per day |
|:-----|:-----|
|  Section Name  |  Calculated Variables |
|  Module Number  |  15‌‌ |
|  Question Number  |  5 |
|  Column  |  2042-2045 |
|  Type of Variable  |  Num |
|  SAS Variable Name  |  POTADA1_ |
|  Question Prologue  |   |
|  Question  |  Potato servings per day |
<br>

### Feature Data Table

|    | Value    | Value Label                            |   Frequency | Percentage   | Weighted Percentage   |
|---:|:---------|:---------------------------------------|------------:|:-------------|:----------------------|
|  2 | 0 - 9999 | Times per day                          |      390253 | 100.00       | 100.00                |
|  3 | BLANK    | Don’t know/Not Sure Or Refused/Missing |       48440 | .            | .                     |
---


<br><br>

## **Computed Other Vege**

---
<br>

|  Label  |  Computed Other Vege |
|:-----|:-----|
|  Section Name  |  Calculated Variables |
|  Module Number  |  15 |
|  Question Number  |  6 |
|  Column  |  2046-2049 |
|  Type of Variable  |  Num |
|  SAS Variable Name  |  VEGEDA2_ |
|  Question Prologue  |   |
|  Question  |  Other vege: table intake in times per day |
<br>

### Feature Data Table

|    | Value    | Value Label                                |   Frequency | Percentage   | Weighted Percentage   |
|---:|:---------|:-------------------------------------------|------------:|:-------------|:----------------------|
|  2 | 0 - 9999 | Times per day (two implied decimal places) |      390165 | 100.00       | 100.00                |
|  3 | BLANK    | Don’t know/Not Sure Or Refused/Missing     |       48528 | .            | .                     |
---


<br><br>

## **Total fruits consumed per day**

---
<br>

|  Label  |  Total fruits consumed per day |
|:-----|:-----|
|  Section Name  |  Calculated Variables |
|  Module Number  |  15‌‌ |
|  Question Number  |  11 |
|  Column  |  2054-2059 |
|  Type of Variable  |  Num |
|  SAS Variable Name  |  _FRUTSU1 |
|  Question Prologue  |   |
|  Question  |  Total fruits consumed per day |
<br>

### Feature Data Table

|    | Value     | Value Label                                                    |   Frequency | Percentage   | Weighted Percentage   |
|---:|:----------|:---------------------------------------------------------------|------------:|:-------------|:----------------------|
|  2 | 0 - 99998 | Number of Fruits consumed per day (two implied decimal places) |      387606 | 100.00       | 100.00                |
|  3 | BLANK     | Not asked or Missing                                           |       51087 | .            | .                     |
---


<br><br>

## **Total vege**

---
<br>

|  Label  |  Total vege |
|:-----|:-----|
|  Section Name  |  Calculated Variables |
|  Module Number  |  15 |
|  Question Number  |  12 |
|  Column  |  2060-2065 |
|  Type of Variable  |  Num |
|  SAS Variable Name  |  _VEGESU1 |
|  Question Prologue  |   |
|  Question  |  Total vege: tables consumed per day |
<br>

### Feature Data Table

|    | Value     | Value Label                                                        |   Frequency | Percentage   | Weighted Percentage   |
|---:|:----------|:-------------------------------------------------------------------|------------:|:-------------|:----------------------|
|  2 | 0 - 99998 | Number of Vegetables consumed per day (two implied decimal places) |      378566 | 100.00       | 100.00                |
|  3 | BLANK     | Not asked or Missing                                               |       60127 | .            | .                     |
---


<br><br>

## **Consume Fruit 1 or more times per day**

---
<br>

|  Label  |  Consume Fruit 1 or more times per day |
|:-----|:-----|
|  Section Name  |  Calculated Variables |
|  Module Number  |  15 |
|  Question Number  |  13 |
|  Column  |  2066 |
|  Type of Variable  |  Num |
|  SAS Variable Name  |  _FRTLT1A |
|  Question Prologue  |   |
|  Question  |  Consume Fruit 1 or more times per day |
<br>

### Feature Data Table

|    |   Value | Value Label                              |   Frequency |   Percentage |   Weighted Percentage |
|---:|--------:|:-----------------------------------------|------------:|-------------:|----------------------:|
|  2 |       1 | Consumed fruit one or more times per day |      238916 |        54.46 |                 52.55 |
|  3 |       2 | Consumed fruit < one time per day        |      148690 |        33.89 |                 34.75 |
|  4 |       9 | Don´t know, refused or missing values    |       51087 |        11.65 |                 12.69 |
---

