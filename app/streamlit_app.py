import streamlit as st

st.set_page_config(
    page_title="Rakuten Product Classification",
    layout="wide",
)

pages = {
    "Dataset Insights": [
        st.Page("views/dataset_insights.py", title="Summary Dashboard", default=True),
        st.Page("views/product_explorer.py", title="Product Explorer"),
    ],
    "Notebooks": [
        st.Page("views/notebook_viewer.py", title="Notebook Viewer"),
    ],
    "Modeling": [
        st.Page("views/model_inference.py", title="Model Inference"),
    ],
}

pg = st.navigation(pages)
pg.run()
