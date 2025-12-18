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
        'region_txt': 'Region',
        'attacktype1_txt': 'Attack_Type',
        'weaptype1_txt': 'Weapon_Type',
        'natlty1_txt': 'Nationality'
    })

    # Military target mapping (do this before selecting columns)
    df['ismilitary'] = (df['targtype1_txt'] == 'Military').astype(int)

    # Select only relevant columns
    df = df[['Year', 'Month', 'Region', 'Attack_Type',
             'Weapon_Type', 'ismilitary', 'Nationality', 'success']]
    return df


def input_data_sl(df):
    regions = df['Region'].unique()
    attack_types = df['Attack_Type'].unique()
    weapon_types = df['Weapon_Type'].unique()
    nationalities = df['Nationality'].unique()
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]

    # 
    cols = st.columns(3)

    with cols[0]:
        st.subheader("Temporal Context")
        year = st.slider("Year of Incident", 1970, 2017, 1970)
        month_index = st.selectbox("Month of Incident", options=list(range(12)), format_func=lambda x: months[x])

    # Spatial and Tactical
    with cols[1]:
        st.subheader("Operational Characteristics")
        region = st.selectbox("Region of Incident", options=regions)
        atk_type = st.selectbox("Attack Method", options=attack_types)
        weapon_type = st.selectbox("Weapon Used", options=weapon_types)

    # Target Details
    with cols[2]:
        st.subheader("Target Profile")
        st.markdown("<br>", unsafe_allow_html=True)
        is_military = st.checkbox("Target is Military Personnel", value=True)
        nationality = st.selectbox("Target Nationality", options=nationalities)

    # Build DataFrame
    input_data_df = pd.DataFrame([{
        'Year': year,
        'Month': month_index + 1,
        'Region': region,
        'Attack_Type': atk_type,
        'Weapon_Type': weapon_type,
        'ismilitary': int(is_military),
        'Nationality': nationality
    }])
    return input_data_df


def predict_w_model(streamlit_input, model_path):
# Load model
    saved_data = joblib.load(model_path)
    model = saved_data['model']
    metadata = saved_data['metadata']
    coefficients = saved_data['coefficients']

    # Make prediction
    prediction = model.predict(streamlit_input)
    confidence = model.predict_proba(streamlit_input)
    confidence_score = confidence[0][prediction[0]]

    # Human-readable interpretation
    interpretation = "Likely to succeed" if prediction[0] == 1 else "Likely to fail"

    # Color-blind friendly color
    color = "#2ca02c" if prediction[0] == 1 else "#d62728"  # green/red alternative
    symbol = "✅" if prediction[0] == 1 else "❌"

    # Display in Streamlit
    st.markdown(f"<span style='color:{color}; font-weight:bold'>{symbol} {interpretation} ({confidence_score*100:.2f}% Confidence)</span>", unsafe_allow_html=True)

    return {
        "model_name": metadata.get('model_name', model_path),
        "prediction": interpretation,
        "confidence": confidence_score,
        "coefficients": coefficients
    }
