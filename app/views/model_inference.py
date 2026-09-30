import streamlit as st

st.title("Model Inference")
st.text_input("Designation")
st.text_area("Description")
st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
st.button("Predict")
