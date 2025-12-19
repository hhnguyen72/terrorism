import numpy as np
import pandas as pd
import streamlit as st
import joblib
# Remember streamlit run .py

# loading script
from streamlit_script import loading_df, input_data_sl, predict_w_model
df = loading_df()

models_path = {
    "Best Model - Nova" : '../model/bm_terrorist_success_rate.joblib',
    "Random Forrest Classification - Vanguard " : '../model/rfc_terrorist_success_rate.joblib',
    "Logistic Regression - Scout" : '../model/lg_terrorist_success_rate.joblib'
}

st.set_page_config(
    page_title="TRA - Terrorist Risk Assessor",
    layout="wide",
    initial_sidebar_state="expanded"
)
tab1, tab2 = st.tabs(["Model", "Dataset"])

with tab1:
    st.title("TRA – Terrorist Risk Assessor")
    st.write("Predicts the likelihood of terrorist attack success using structured historical indicators")
    st.markdown("<br>", unsafe_allow_html=True)

    streamlit_input = input_data_sl(df)
    st.markdown("<br>", unsafe_allow_html=True)

    selected_model_name = st.selectbox("Deploy a model for prediction:", list(models_path.keys()))
    selected_model_path = models_path[selected_model_name]
    model_result = predict_w_model(streamlit_input, selected_model_path)

    st.write("Model Coefficients:")
    st.dataframe(
        model_result['coefficients'],
        use_container_width=True
    )


with tab2:
    st.title("Dataset")

    cols = st.columns(3)
    with cols[0]:
        st.subheader("Source & References")
        st.markdown(
                "**Authors:** The Study of Terrorism and Responses to Terrorism (START)  \n"
                "**Dataset:** [Global Terrorism Database (GTD)](https://www.kaggle.com/datasets/START-UMD/gtd)  \n"
                 "**Additional Resources:**  \n"
                "[GTD Codebook](https://www.start.umd.edu/sites/default/files/2024-10/Codebook.pdf)  \n"
                "[UMD START – GTD Portal](https://www.start.umd.edu/data-tools/GTD#Top)"
        )

    with cols[1]:
        st.subheader("Cleaned Dataset")
        st.markdown(
                    f"- Numerical Columns: [Year, Month, ismilitary, success]  \n"
                    f"- Categorical Columns: [Region, Attack_Type, Weapon_Type, Nationality]  \n"
                    f"- Columns, Rows After Cleaning: {df.shape}"
        )
        st.markdown("<br>", unsafe_allow_html=True)

    with cols[2]:
        st.subheader("Dataset Scope & Limits")
        st.markdown(
                    "- Historical terrorism data stops at 2017  \n"
        "- Reporting bias may lead to underreported incidents  \n"
        "- Suitable for training, pattern analysis, and baseline risk modeling for global terrorism"
        )
    
    st.dataframe(df)