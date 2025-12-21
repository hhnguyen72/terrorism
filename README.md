# Threat Assessment Operations (TAO) - 
## Hung Nguyen, Supra Coder DDI (Data & Development Immersive) Cohort 13 Capstone

## Table of Contents
- [Overview](#overview)
  - [Project Proposal](#project_proposal)
  - [Crediting Dataset](#
  - [Pre-Data Cleaning](#pre-data-cleaning)
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

For my DDI capstone, 
Question/Scope/Direction:

    How can we model insights from historical terrorist incidents to enhance training and risk awareness in a Terrorist Risk Assessment?

    
Minimum Viable Product (MVP)


### Pre-Data Cleaning 

Data Shape (rows, columns):

    (181691, 135)

Normally, DataFrame.info() lists and prints the data type for each column. However, because this dataset contains 135 columns, the detailed per-column output is omitted here. I have written a function to address the omitted output issue: this function extracts, identifies, and prints each column's dtype, as well as the numerical/categorical columns distribution.

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


Features & Target

    X = df[['Year', 'Month', 'Region', 'Attack_Type',
            'Weapon_Type', 'ismilitary', 'Nationality']]
    y = df['success']



## Data Pipeline


### Preprocessor


    def pipeline_preprocessor():
        #1. Load Dataset
        df = pd.read_csv('../data/globalterrorismdb_0718dist.csv')
        df = df.copy()
    
        #2. Data Cleaning
        df = df.rename(columns={'iyear': 'Year', 'imonth': 'Month','region_txt': 'Region', 
                                'attacktype1_txt': 'Attack_Type', 'weaptype1_txt': 'Weapon_Type', 'natlty1_txt': 'Nationality'})
        
        # Mapping
        df['ismilitary'] = df['targtype1_txt'].apply(lambda x: 1 if x == 'Military' else 0)
    
        # 3. Selecting Features and Target
        X = df[['Year', 'Month', 'Region', 'Attack_Type',
                'Weapon_Type', 'ismilitary', 'Nationality']]
        y = df['success']
    
        # 4. Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)
    
        # 5. Identify numeric vs categorical columns
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
        # 6. Transform pipelines
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", MinMaxScaler())
            ])
        
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy='constant', fill_value='Unknown')),
            ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
            ])
    
        # 7. Column transformer
        preprocessor = ColumnTransformer(transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
            ])
        
        return preprocessor, X_train, X_test, y_train, y_test
    
    preprocessor, X_train, X_test, y_train, y_test = pipeline_preprocessor()

### Model Fitting


    def fit_log_reg(preprocessor, X_train, X_test, y_train, y_test):
        log_reg_model = Pipeline([
            ('preprocess', preprocessor),
            ('model', LogisticRegression(solver='saga'))
            ])
    
        # 9. Fit model
        log_reg_model.fit(X_train, y_train)
        log_score = log_reg_model.score(X_test, y_test)
        y_pred_lr = log_reg_model.predict(X_test)
    
        print(f"Logisitic Regression Score: {log_score:.4f}")
        print(f"Logisitic Regression Accurary: {log_score *100:.2f}")
        return log_reg_model, log_score, y_pred_lr
    
    log_reg_model, log_score, y_pred_lr = fit_log_reg(preprocessor, X_train, X_test, y_train, y_test)


### Joblib

After fitting and evaluating my three models, I preserved my trained models and their corresponding metadata using joblib to enable reproducibility and deployment.

    # Save the model to a file
    (lg/rfc/bm)_metadata = {
        "model_name": "(lg/rfc/bm)_terrorist_success_rate",
        "trained_date": "2025-12-12",
        "training_data_description": (
            "Predicting a terrorist attack's success rate based on "
            "Year, Month, Region, Attack_Type, Weapon_Type, "
            "ismilitary, and Nationality"
        ),
        "accuracy": 0.89,
        "author": "Hung Nguyen"
    }
    
    # Save the model and metadata to a file
    joblib.dump({'model': (lg/rfc/bm)_model, 'coefficients': (lg/rfc/bm)_coeff, 'metadata': (lg/rfc/bm)_metadata}, '../model/(lg/rfc/bm)_terrorist_success_rate.joblib')
    print("(lg/rfc/bm) saved successfully with coeff and metadata.")


## Streamlit



## Future Direction
