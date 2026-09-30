import os
import nbformat
from nbconvert import HTMLExporter
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)

@st.cache_data
def load_notebook_html(nb_path: str) -> str:
    if not os.path.isabs(nb_path):
        nb_path = os.path.join(PROJECT_ROOT, nb_path)

    if not os.path.exists(nb_path):
        return ""

    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        exporter = HTMLExporter()
        exporter.template_name = "classic"
        html_data, _ = exporter.from_notebook_node(nb)
        return html_data
    except Exception as e:
        return f"<div style='color:red;'>Error rendering notebook: {e}</div>"
