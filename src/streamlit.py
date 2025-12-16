import numpy as np
import pandas as pd
import streamlit as st
import joblib
# Remember streamlit run .py

# loading script
from streamlit_script import loading_df, input_data_sl, predict_w_model
df = loading_df()
models_path = {
    "Best Model" : '../model/bm_terrorist_success_rate.joblib',
    "Random Forrest Classification" : '../model/rfc_terrorist_success_rate.joblib',
    "Logistic Regression" : '../model/lg_terrorist_success_rate.joblib'
}

tab1, tab2 = st.tabs(["Model", "Dataset Used"])

with tab1:
    st.title("Analyzing Global Terrorist Incidents")
    st.subheader("Exploratory analysis and ML-based prediction")

    streamlit_input = input_data_sl(df)
    selected_model_name = st.selectbox("Choose a model for prediction:", list(models_path.keys()))
    selected_model_path = models_path[selected_model_name]
    model_result = predict_w_model(streamlit_input, selected_model_path)

    # Display prediction
    st.subheader("Prediction Result:")
    st.write(f"Prediction: {model_result['prediction']} ({model_result['confidence']*100:.2f}% Confidence)")
    st.dataframe(
        model_result['coefficients'],
        use_container_width=True
    )


with tab2:
    st.title("Dataset")
    st.subheader("Description for Dataset")
    st.write(f"Rows, Columns: {df.shape}")

    st.dataframe(df)
