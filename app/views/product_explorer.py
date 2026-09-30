import random
import pandas as pd
from PIL import Image
import streamlit as st
from data_loader import load_data
from src.data_utils import get_image_path

df = load_data()

if df is None:
    st.error("Data files not found in data/raw/.")
    st.stop()

st.title("Product Explorer")

col_filter, col_btn = st.columns([3, 1])
with col_filter:
    classes = sorted(df["prdtypecode"].unique().tolist())
    selected_class = st.selectbox("Filter by prdtypecode:", ["All"] + classes)

subset = df if selected_class == "All" else df[df["prdtypecode"] == selected_class]

with col_btn:
    st.write("")
    st.write("")
    pick_random = st.button("Random Sample")

if "current_idx" not in st.session_state or pick_random:
    st.session_state["current_idx"] = random.choice(subset.index.tolist())

if st.session_state["current_idx"] not in subset.index:
    st.session_state["current_idx"] = subset.index[0]

row = df.loc[st.session_state["current_idx"]]

col_text, col_img = st.columns([1, 1])

with col_text:
    st.text(f"Index: {st.session_state['current_idx']}")
    st.text(f"prdtypecode: {row['prdtypecode']}")
    st.text(f"productid: {row['productid']}")
    st.text(f"imageid: {row['imageid']}")

    st.subheader("Designation")
    st.write(row["designation"])

    st.subheader("Description")
    if pd.isna(row["description"]) or str(row["description"]).strip() == "":
        st.text("None")
    else:
        st.write(row["description"])

with col_img:
    st.subheader("Image")
    img_path = get_image_path(row["imageid"], row["productid"])
    try:
        st.image(Image.open(img_path))
    except Exception:
        st.text("Image not found")
