import streamlit as st
from src.data_utils import load_merged_data


@st.cache_data
def load_data():
    return load_merged_data()
