import streamlit as st
from data_loader import load_data

df = load_data()

if df is None:
    st.error("Data files not found in data/raw/.")
    st.stop()

st.title("Dataset Insights — Summary")

col1, col2, col3 = st.columns(3)
col1.metric("Total Samples", f"{len(df):,}")
col2.metric("Classes (prdtypecode)", f"{df['prdtypecode'].nunique()}")
missing_desc = df["description"].isna().mean() * 100
col3.metric("Missing Descriptions", f"{missing_desc:.2f}%")

st.subheader("Class Distribution")
class_counts = df["prdtypecode"].value_counts().sort_values(ascending=False)
st.bar_chart(class_counts)

with st.expander("View Class Counts Table"):
    df_counts = class_counts.reset_index()
    df_counts.columns = ["prdtypecode", "count"]
    st.dataframe(df_counts)

st.subheader("Description Availability")
desc_status = df["description"].notna().value_counts().rename({True: "Present", False: "Missing"})
st.bar_chart(desc_status)
