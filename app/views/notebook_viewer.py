import glob
import os
import streamlit as st
import streamlit.components.v1 as components
from notebook_renderer import PROJECT_ROOT, load_notebook_html

st.title("Notebook Viewer")

notebooks_dir = os.path.join(PROJECT_ROOT, "notebooks")
nb_files = sorted(glob.glob(os.path.join(notebooks_dir, "*.ipynb")))

if not nb_files:
    st.info("No `.ipynb` notebook files found in `notebooks/` directory.")
    st.stop()

nb_options = {os.path.basename(f): f for f in nb_files}

selected_name = st.selectbox("Select Notebook:", list(nb_options.keys()))
selected_path = nb_options[selected_name]

html_content = load_notebook_html(selected_path)

if html_content:
    components.html(html_content, height=1000, scrolling=True)
else:
    st.text(f"Unable to render `{selected_name}`.")
