# Threat Assessment Operations (TAO) 
## Hung Nguyen, Supra Coder DDI (Data & Development Immersive) Cohort 13 Capstone

## Table of Contents
- [Overview](#overview)
  - [Project Proposal](#project-proposal)
  - [Crediting Dataset](#crediting-dataset)
  - [Pre-Data Cleaning](#pre-data-cleaning)
  - [Environment Setup](#environment-setup)
- [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Data Cleaning](#data-cleaning)
  - [Data Visualization](#data-visualization)
- [Models Selected](#models-selected)
- [Data Pipeline](#data-pipeline)
  - [Preprocessor](#preprocessor)
  - [Model Fitting](#model-fitting)
  - [Joblib](#joblib)
- [Streamlit](#streamlit)
- [Future Directions](#future-directions)

## Overview

### Project Proposal

Threat Assessment Operations (TAO) is a capstone project that documents exploratory data analysis and implements machine learning classification models to analyze historical terrorist incident data to support training-oriented Terrorist Risk Assessment via interactive user analysis.   

Research Question

    How can we model insights from historical terrorist incidents to enhance training effectiveness and risk awareness in Terrorist Risk Assessment?
    
Minimum Viable Product (MVP)

The Terrorist Risk Assessor (TRA) consists of trained machine learning classification models deployed through an interactive Streamlit application. Streamlit enables rapid web-based deployment, allowing live demonstrations through a simple and accessible user interface. TRA is a training tool. Users can interact with historical or hypothetical incident parameters and see how changes affect predicted attack outcomes. The MVP is intended as a complementary analytical tool, supporting Terrorist Risk Assessment by enhancing risk awareness and analytical reasoning rather than replacing human judgment.

### Crediting Dataset

Dataset: Global Terrorism Database (GTD) 

Authors/Maintainers: Study of Terrorism and Responses to Terrorism (START), University of Maryland  

Data Collected: Historical terrorist incidents worldwide, including attack types, targets, perpetrators, and outcomes


**Additional Resources:**  
- [Kaggle GTD dataset](https://www.kaggle.com/datasets/START-UMD/gtd)  
- [GTD Codebook](https://www.start.umd.edu/gtd/)  
- [UMD START – GTD Portal](https://www.start.umd.edu/gtd/)



### Pre-Data Cleaning 

The dataset contains 181,691 incidents × 135 attributes. I wrote a function, info_dtypes(df) (located in notebooks/eda.ipynb), to identify each column’s data type and summarize the distribution of numerical and categorical columns, addressing the omitted column outputs.


 Output:
 
      | Category          | Type        | Count |
      | ----------------- | ----------- | ----- |
      | **Data Types**    | object      | 58    |
      |                   | float64     | 55    |
      |                   | int64       | 22    |
      | **Columns**       | Numerical   | 77    |
      |                   | Categorical | 58    |
      | **Total Columns** | —           | 135   |

Numerical Columns (Float64 + Int64): 
    
    Sample:
    eventid, iyear, imonth, iday, extended, country, region, latitude, longitude, specificity


Categorical Columns (Objects):

    Sample:
    approxdate, resolution, country_txt, region_txt, provstate, city, location, summary, alternative_txt, attacktype1_txt


NaN/Missing Values:

106 columns contain NaN or missing values. A majority of those listed columns are sub-categories columns which are optional to fill out. This explains why the ouput shows a wide variation in NaN/missing values across the columns, ranging from nearly all entries missing to only a few. 

    Sample:
    gsubname3           181671
    weapsubtype4_txt    181621
                        ...  
    specificity              6
    multiple                 1


## Environment Setup

To replicate the environment exactly, run:

    bash

    1. Create the Conda environment from the YAML file
    conda env create -f ml_global_terror_env.yml

    2. Activate the environment
    conda activate ml_global_terror_env
    

## Exploratory Data Analysis

### Data Cleaning 

Key steps performed on the dataset:

- Checked for Duplicates: None found
- Selected 7 columns out of 135 for target and features
- Renamed 7 columns for clarity
- Created 1 derived column ['ismilitary'] based on targttype1_txt  


![CleanDataset](img/clean_df.png)

Note: The authors/maintainers have already cleaned and formated the dataset; I worked with the pre-processed dataset.


### Data Visualization

![Heatmap](img/heatmap_features.png)



![Timeline](img/timeline.png)


![Region](img/terrorist_incidents_by_region.png)



## Models Selected

To calculate the probability of a terrorist attack's success, I used  classification model to predict and label the success rate based on user input. I also experimented with predictive (regression) model; however, since the target is categorical rather than continuous, the accuracy was extremely low.  

Here's the following classification models used for the MVP:

    - sklearn.linear_model: LogisticRgression
    - sklearn.ensemble: RandomForestClassifier
    - Hyperparameter tuning: RandomForestClassfier


![cm_lr](img/cm_lr.png)


![cm_rfc](img/cm_rf.png)

        # 3. Selecting Features and Target
        X = df[['Year', 'Month', 'Region', 'Attack_Type',
                'Weapon_Type', 'ismilitary', 'Nationality']]
        y = df['success']

Features & Target


    X = df[['Year', 'Month', 'Region', 'Attack_Type',
            'Weapon_Type', 'ismilitary', 'Nationality']]
    y = df['success']




## Data Pipeline

Note: Located in the notebook/eda.ipynb

### Preprocessor

First, I wrote a pipeline_preprocessor() function to produce a re-usable output for multiple classification models. This function performs the following steps:

- Load and clean the dataset
- Select features and target
- Split the data into 80% train and 20% test sets
- Handle numeric and categorical features (imputation, scaling, one-hot encoding)

### Model Fitting

Afterwards, I implemented the fit_log_reg() and fit_rfc_model() functions (located in notebooks/eda.ipynb) to fit the classification models. These functions take the output of pipeline_preprocessor() to:

- Fit the classification model (Logistic Regression or Random Forest)
- Test the features against the target
- Print the model’s accuracy score
- Extract feature importances or model coefficients for interpretation

I also performed hyperparameter tuning for the Random Forest model (bm) to optimize performance, including parameters like the number of estimators, maximum depth, and minimum samples per leaf. This ensures the model achieves better accuracy and generalization.


### Joblib

After fitting and evaluating my three models (log_reg, rfc, and bm), I preserved my trained models and their corresponding metadata using joblib to enable reproducibility and deployment.

- Model Name and Trained Date
- Training Data Description
- Accuracy score/success rate
- Author Information


## Streamlit


Launch the Streamlit application from the src folder:

    cd src/streamlit.py
    streamlit run streamlit.py

![Homepage](img/tra_homepage.png)



## Future Direction
