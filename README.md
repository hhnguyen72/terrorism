# Threat Assessment Operations (TAO) - 
## Hung Nguyen, Supra Coder Data & Development Immersive Cohort 13 Capstone

## Table of Contents
- [Overview](#overview)
  - [Project Proposal](#project_proposal)
  - [Pre-Data Cleaning](#pre-data-cleaning)
- [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Data Cleaning](#data-cleaning)
  - [Data Visualization](#data-visualization)
- [Models Selected](#models-selected)
- [Data Pipeline](#data-pipeline)
  - [Features & Target](#features-&-target)
  -  

## Overview

### Project Proposal
    State your question. What is it that you are curious about? What are you looking for in the data
    State your MVP. MVP is your Minimum Viable Product. What's the minimum that you hope to accomplish? Then, feel free to expand on MVP+ and MVP++.


### Pre-Data Cleaning 

Data Shape (rows, columns)

    (181691, 135)

Normally, DataFrame.info() lists the data type for each column. However, because this dataset contains 135 columns, the detailed per-column output is omitted here.
I have written a function to address the omitted output issue: this function extracts, identifies, and prints each column's dtype, as well as the numerical/categorical columns distribution.

Function:

        def info_dtypes(df):
          num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
          cat_cols = df.select_dtypes(include=['object']).columns.tolist()
      
          num_cols_count = df[num_cols].dtypes.count()
          cat_cols_count = df[cat_cols].dtypes.count()
      
          print(f"Dtypes: \n{df.dtypes.value_counts()}\n")
          print(f"Numerical Columns: {num_cols_count}")
          print(f"Categorical Columns: {cat_cols_count}")
          return num_cols, cat_cols
        num_cols, cat_cols = info_dtypes(df)

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

    eventid, iyear, imonth, iday, extended, country, region, latitude, longitude, specificity
    vicinity, crit1, crit2, crit3, doubtterr, alternative, multiple, success, suicide, attacktype1
    attacktype2, attacktype3, targtype1, targsubtype1, natlty1, targtype2, targsubtype2, natlty2, targtype3, targsubtype3
    natlty3, guncertain1, guncertain2, guncertain3, individual, nperps, nperpcap, claimed, claimmode, claim2
    claimmode2, claim3, claimmode3, compclaim, weaptype1, weapsubtype1, weaptype2, weapsubtype2, weaptype3, weapsubtype3
    weaptype4, weapsubtype4, nkill, nkillus, nkillter, nwound, nwoundus, nwoundte, property, propextent
    propvalue, ishostkid, nhostkid, nhostkidus, nhours, ndays, ransom, ransomamt, ransomamtus, ransompaid
    ransompaidus, hostkidoutcome, nreleased, INT_LOG, INT_IDEO, INT_MISC, INT_ANY


Categorical Columns (Objects):

    approxdate, resolution, country_txt, region_txt, provstate, city, location, summary, alternative_txt, attacktype1_txt
    attacktype2_txt, attacktype3_txt, targtype1_txt, targsubtype1_txt, corp1, target1, natlty1_txt, targtype2_txt, targsubtype2_txt, corp2
    target2, natlty2_txt, targtype3_txt, targsubtype3_txt, corp3, target3, natlty3_txt, gname, gsubname, gname2
    gsubname2, gname3, gsubname3, motive, claimmode_txt, claimmode2_txt, claimmode3_txt, weaptype1_txt, weapsubtype1_txt, weaptype2_txt
    weapsubtype2_txt, weaptype3_txt, weapsubtype3_txt, weaptype4_txt, weapsubtype4_txt, weapdetail, propextent_txt, propcomment, divert, kidhijcountry
    ransomnote, hostkidoutcome_txt, addnotes, scite1, scite2, scite3, dbsource, related

NaN/Missing Values:

In this dataset, 106 columns contain NaN or missing values. A majority of those listed columns are sub-categories columns which are optional to fill out. 
This explains why the ouput shows a wide variation in NaN/missing values across the columns, ranging from nearly all entries missing to only a few. 
Remember: there is 181691 total entities.

Sample:

    | Column Name       | Missing Count |
    |------------------|----------------|
    | gsubname3         | 181,671       |
    | weapsubtype4_txt  | 181,621       |
    | weapsubtype4      | 181,621       |
    | ...               | ...           |
    | specificity       | 6             |
    | multiple          | 1             |
    | doubtterr         | 1             |




## Exploratory Data Analysis

### Data Cleaning 

Selecting and renaming which columns to focus.
Function:

    def data_man(df):
        # Renaming Columns to use for ML model
        rename_cols = df.rename(columns={'iyear': 'Year', 
                                         'imonth': 'Month', 
                                         'region_txt': 'Region', 
                                         'attacktype1_txt': 'Attack_Type', 
                                         'weaptype1_txt': 'Weapon_Type', 
                                         'natlty1_txt': 'Nationality'})
        df = rename_cols
    df = data_man(df)





### Data Visualization


## Models Selected



## Data Pipeline

