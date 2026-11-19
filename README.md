# Rakuten Multimodal Product Classification

Multimodal classification challenge predicting product category codes (`prdtypecode`) from catalog text features (designation, description) and product catalog images.

## Project Structure

```text
rakuten-multimodal-classification/
├── data/                               # Data directory (ignored in .gitignore)
│   └── raw/                            # Untouched raw challenge files
│       ├── X_train_update.csv          # Training text features (designation, description)
│       ├── Y_train_CVw08PX.csv         # Training labels (prdtypecode)
│       ├── X_test_update.csv           # Test text features
│       └── images/                     # Raw product catalog images (~2.2 GB)
│           ├── image_train/
│           └── image_test/
│
├── notebooks/                          # Sequenced by project milestones
│   ├── 01_dataviz_and_audit.ipynb      # Milestone 1: Data visualization & audit
│   ├── 02_preprocessing_text.ipynb     # Fixed text cleaning
│   ├── 02_preprocessing_image.ipynb    # Fixed image preparation
│   └── 02_preprocessing.ipynb          # Shared train/validation split
│
├── src/                                # Reusable Python modules
│   ├── __init__.py
│   ├── data_utils.py                   # Data loading & image path utilities
│   ├── text_utils.py                   # Text cleaning & normalization
│   ├── vision_utils.py                 # Dimensionality reduction & image plotting
│   └── metrics.py                      # Weighted-F1 & evaluation metrics
│
├── app/                                # Streamlit web application
│   ├── streamlit_app.py                # Main app entrypoint & navigation
│   ├── data_loader.py                  # Cached data loader
│   ├── notebook_renderer.py            # Notebook rendering helper
│   └── views/
│       ├── dataset_insights.py         # Summary dashboard & class distribution
│       ├── product_explorer.py         # Interactive product & image viewer
│       ├── notebook_viewer.py          # Interactive notebook viewer
│       └── model_inference.py          # Live model prediction view
│
├── docs/                               # Challenge documentation & specifications
│   ├── challenge.md
│   └── (EN) Projects_methodology_report.pdf
│
├── reports/                            # Project reports & figures
│   └── figures/
│
├── .gitignore                          # Excludes /data, checkpoints, and cache
├── pyproject.toml                      # Package configuration (enables pip install -e .)
├── requirements.txt                    # Pinned library dependencies
└── README.md                           # Project documentation
```

## Setup & Installation

1. **Clone repository:**
   ```bash
   git clone https://github.com/wahajaslm/liora-capstone.git
   cd capstone_raukaten
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies & local package:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Get the trained models (needed for the Model Inference page):**

   The models are not in git. They are trained on the confidential challenge
   data, so they are only shared inside the team.

   - CamemBERT and ConvNeXt: ask the team for `rakuten_models.zip`, then unzip
     it in the repository root:
     ```bash
     unzip rakuten_models.zip
     ```
     This creates `models/camembert/` and
     `models/outputs/03_7_convnext_tiny_optimized_best.pt`.
   - LinearSVC: build it from the data (a few minutes):
     ```bash
     python scripts/export_svm.py
     ```
     This creates `models/svm_tfidf_calibrated.joblib`.

5. **Launch Streamlit Demo:**
   ```bash
   streamlit run app/streamlit_app.py
   ```
