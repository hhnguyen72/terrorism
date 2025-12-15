import pandas as pd 
import numpy as np
import streamlit as st
import joblib

def loading_df():
    data_path = r"D:\DDI\terrorism\data\globalterrorismdb_0718dist.csv"
    df = pd.read_csv(data_path)

    # Rename columns for ML use
    df = df.rename(columns={
        'iyear': 'Year',
        'imonth': 'Month',
        'country_txt': 'Country',
        'region_txt': 'Region',
        'attacktype1_txt': 'Attack_Type',
        'weaptype1_txt': 'Weapon_Type',
        'natlty1_txt': 'Nationality'
    })

    # Military target mapping (do this before selecting columns)
    df['ismilitary'] = (df['targtype1_txt'] == 'Military').astype(int)

    # Select only relevant columns
    df = df[['Year', 'Month', 'Country', 'Region', 'Attack_Type',
             'Weapon_Type', 'ismilitary', 'Nationality', 'success']]
    return df

def input_data_sl(df):
    countries = df['Country'].unique()
    regions = df['Region'].unique()
    attack_types = df['Attack_Type'].unique()
    weapon_types = df['Weapon_Type'].unique()
    nationalities = df['Nationality'].unique()

    # 
    year = st.slider("Select the year", 1970, 2017, 1)
    month = st.slider("Select the month", 0, 12, 1)
    country = st.selectbox("Select the Country", options = countries)
    region = st.selectbox("Select the Region", options = regions)
    atk_type = st.selectbox("Select the Attack Method", options = attack_types)
    weapon_types = st.selectbox("Select the Weapon", options = weapon_types)
    ismilitary = st.slider("Is the target military", 0, 1, 1)
    nationality = st.selectbox("Select the target's nationality", options = nationalities)


    input_data_df = pd.DataFrame([{
        'Year': year,
        'Month': month,
        'Country': country,
        'Region' : region,
        'Attack_Types': atk_type,
        'Weapon_Type' : weapon_types,
        'ismilitary' : ismilitary,
        'Nationality' : nationality
    }])
    return input_data_df