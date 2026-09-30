import os
import pandas as pd
from PIL import Image

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
IMAGES_DIR = os.path.join(DATA_RAW_DIR, "images")


def get_data_paths():
    """File paths for raw challenge datasets."""
    return {
        "x_train": os.path.join(DATA_RAW_DIR, "X_train_update.csv"),
        "y_train": os.path.join(DATA_RAW_DIR, "Y_train_CVw08PX.csv"),
        "x_test": os.path.join(DATA_RAW_DIR, "X_test_update.csv"),
        "image_train": os.path.join(IMAGES_DIR, "image_train"),
        "image_test": os.path.join(IMAGES_DIR, "image_test"),
    }


def load_raw_data():
    """Load raw X_train and Y_train dataframes."""
    paths = get_data_paths()
    df_x = pd.read_csv(paths["x_train"], index_col=0)
    df_y = pd.read_csv(paths["y_train"], index_col=0)
    return df_x, df_y


def load_merged_data():
    """Load and merge X_train with target prdtypecode."""
    df_x, df_y = load_raw_data()
    df_x["prdtypecode"] = df_y["prdtypecode"]
    return df_x


def get_image_path(imageid, productid, split="image_train"):
    """Return the path for a product catalog image."""
    paths = get_data_paths()
    return os.path.join(paths[split], f"image_{imageid}_product_{productid}.jpg")


def load_image(imageid, productid, split="image_train"):
    """Load a product catalog image as a PIL Image."""
    return Image.open(get_image_path(imageid, productid, split))
