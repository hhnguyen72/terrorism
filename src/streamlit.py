import numpy as np
import pandas as pd
import streamlit as st
import joblib
# Remember streamlit run .py

# loading df
from script import loading_df, input_data_sl
df = loading_df()
# month_map = {
#     0: "Unknown",
#     1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
#     5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
#     9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
# }

@st.cache_resource
def load_models():
    return {
        "Logistic Regression": joblib.load("model/lg_terrorist_success_rate.joblib"),
        "Random Forest": joblib.load("model/rfc_terrorist_success_rate.joblib"),
        "Balanced Model": joblib.load("model/bm_terrorist_success_rate.joblib")
    }
models = load_models()

# Tabs for organizing content
tab1, tab2, tab3 = st.tabs(["Context", "Dataset", "Deploy Model"])

with tab1:
    st.image("../img/recorded_mil_atk.png")
    st.image("../img/terrorist_incidents_by_region.png")


with tab2:
    st.subheader("Global Terrorism (1970 - 2017) Clean Up Dataset")
    st.write(df.shape)
    st.dataframe(df)

with tab3:
    streamlit_input = input_data_sl(df)

