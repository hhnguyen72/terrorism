import numpy as np
import pandas as pd
import streamlit as st
import joblib

# loading df
from script import loading_df
df = loading_df()
# 

# Tabs for organizing content
tab1, tab2 = st.tabs(["Overview", "Charts"])

# Entire Dataframe.
st.subheader("TTest")
st.dataframe(df)

