---
created: 2024-05-13T21:23:56 (UTC -06:00)
tags: []
source: https://bootcampspot.instructure.com/courses/5432/pages/16-project-2-overview?module_item_id=1201087
author: 
---

# 16: Project 2: Bootcamp: DU-VIRT-AI-PT-02-2024-U-LOLC-MTTH

## Requirements

### Data Model Implementation (25 points)

-   There is a Jupyter notebook that thoroughly describes the data extraction, cleaning, and transformation process, and the cleaned data is exported as CSV files for the machine learning model. (10 points)
    - **Complete:**  Exported as parquet file instead of CSV because parquet files load faster and are much smaller on github.
    
-   A Python script initializes, trains, and evaluates a model or loads a pretrained model. (10 points)
    - **Complete:**  
        - This data cleaning process is driven by: 
            - [brfss_2021/1_data_cleanding_2021.ipynb](brfss_2021/1_data_cleanding_2021.ipynb) and 
            [brfss_2015/1_reference_data_2015.ipynb](brfss_2015/1_reference_data_2015.ipynb).  
        - The python functions used are in:
            - [pkgs/ml_clean_config.py](pkgs/ml_clean_config.py)
            - [pkgs/ml_clean_features.py](pkgs/ml_clean_features.py)
            - [pkgs/ml_functions.py](pkgs/ml_functions.py)
    
-   The model demonstrates meaningful predictive power at least 75% classification accuracy or 0.80 R-squared. (5 points)
    - **Complete:**  See the Performance Reports:
        - [brfss_2015/3_performance_report.ipynb](brfss_2015/3_performance_report.ipynb)
            - detailed reports: [brfss_2015/reports](brfss_2015/reports)
        - [brfss_2021/3_performance_report.ipynb](brfss_2021/3_performance_report.ipynb)
            - detailed reports: [brfss_2021/reports](brfss_2021/reports)
        

### Data Model Optimization (25 points)

-   The model optimization and evaluation process showing iterative changes made to the model and the resulting changes in model performance is documented in either a CSV/Excel table or in the Python script itself. (15 points)
    - **Complete:**  See the Optimization Reports:
        - [2015_brfss/5_optimization_report.ipynb](2015_brfss/5_optimization_report.ipynb)
        - [2021_brfss/5_optimization_report.ipynb](2021_brfss/5_optimization_report.ipynb)

-   Overall model performance is printed or displayed at the end of the script. (10 points)
    - **Complete:**See the Optimization Reports:
        - [2015_brfss/5_optimization_report.ipynb](2015_brfss/5_optimization_report.ipynb)
        - [2021_brfss/5_optimization_report.ipynb](2021_brfss/5_optimization_report.ipynb)

### GitHub Documentation (25 points)

-   GitHub repository is free of unnecessary files and folders and has an appropriate .gitignore in use. (10 points)
    - **Complete:**  with the exception that there is some archived information under the [data_cleaning](data_cleaning) and [data](data) folders related to exploration and refinement of features (csv and xlsx files)

-   The README is customized as a polished presentation of the content of the project. (15 points)
    - **Complete:**  [README.md](README.md)

### Presentation Requirements (25 points)

Your presentation should cover the following:

-   An executive summary or overview of the project and project goals. (5 points)
    - **Complete:**  Slides 2 & 3
-   An overview of the data collection, cleanup, and exploration processes. Include a description of how you evaluated the trained model(s) using testing data. (5 points)
    - **Complete:**  Slides 4 & 5
-   The approach that your group took in achieving the project goals. (5 points)
    - **Complete:**  Slide 6
-   Any additional questions that surfaced, what your group might research next if more time was available, or share a plan for future development. (3 points)
    - **Complete:**  Slide 11
-   The results and conclusions of the application or analysis. (3 points)
    - **Complete:**  Slides **11** with slides 7, 8, 9 contributing with analysis 
-   Slides effectively demonstrate the project. (2 points)
    - **Complete:**  Best as can be done with only 7 minutes to brief
-   Slides are visually clean and professional. (2 points)
    - **Complete:**  Great Job Elia


### Presentation Guidelines

This section lists the Project 2 presentation guidelines. Each group will prepare a formal, 10-minute presentation (7 minutes for the presentation followed by a 3-minute question-and-answer session) that covers the following points.

-   An executive summary or overview of the project and project goals:
    
    -   Explain how the project relates to the industry you selected.
        - **Complete:**  
-   An overview of the data collection, cleanup, and exploration processes:
    
    -   Describe the source of your data and why you chose it for your project.
        - **Complete:**  
    -   Describe the collection, exploration, and cleanup process.
        - **Complete:**  
-   The approach that your group took to achieve the project goals:
    
    -   Include any relevant code or demonstrations of the application or analysis.
        - **Complete:**  
        
    -   Discuss any unanticipated insights or problems that arose and how you resolved them.
        - **Complete:**  
        
-   The results/conclusions of the application or analysis:
    
    -   Include relevant images or examples to support your work.
        - **Complete:**  
        
    -   If the project goal was not achieved, discuss the issues and how you attempted to resolve them.
        - **Complete:**  
        
-   Next steps:
    
    -   Briefly discuss potential next steps for the project.
        - **Complete:**  


### Presentation Day

- Your group will have a total of 10 minutes—7 minutes for the presentation followed by a 3-minute question-and-answer session. It’s crucial that you find time to rehearse before presentation day.
    - **Complete:**  
- On the day of your presentation, each member of your group is required to submit the URL of your GitHub repository for grading.
    - **Complete:**  