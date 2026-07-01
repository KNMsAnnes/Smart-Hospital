import streamlit as st
import pandas as pd
import pickle

# ===============================
# LOAD MACHINE LEARNING MODEL
# ===============================

with open("hospital_model.pkl", "rb") as f:
    bundle = pickle.load(f)
    st.success("Model loaded successfully!")
    st.write(bundle.keys())

x
