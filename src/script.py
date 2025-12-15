import pandas as pd 
import numpy as np

def input_to_model(df):
    """
    Generate a random attack scenario from the dataset for model input.
    df: your loaded dataframe
    """
    
    # Get unique options for categorical columns
    countries = df['Country'].unique()
    regions = df['Region'].unique()
    attack_types = df['Attack_Type'].unique()
    weapon_types = df['Weapon_Type'].unique()
    nationalities = df['Nationality'].unique()
    
    # Generate random input
    input_data = pd.DataFrame([{
        'Year': np.random.randint(1971, 2018),  # 2017 inclusive
        'Month': np.random.randint(1, 13),      # 1–12
        'Country': np.random.choice(countries),
        'Region': np.random.choice(regions),
        'Attack_Type': np.random.choice(attack_types),
        'Weapon_Type': np.random.choice(weapon_types),
        'ismilitary': np.random.choice([0, 1]),
        'Nationality': np.random.choice(nationalities)
    }])
    return input_data

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


