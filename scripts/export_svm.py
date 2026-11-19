"""Fit and export the calibrated LinearSVC + TF-IDF pipeline for inference."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "text_train_processed.csv"
OUT_PATH = PROJECT_ROOT / "models" / "svm_tfidf_calibrated.joblib"


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    train = data[data["split"] == "train"]
    validation = data[data["split"] == "validation"]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, binary=True)),
        ("svm", CalibratedClassifierCV(LinearSVC(C=1, random_state=42), cv=3)),
    ])
    pipeline.fit(train["text_combined"], train["prdtypecode"])

    preds = pipeline.predict(validation["text_combined"])
    weighted_f1 = f1_score(validation["prdtypecode"], preds, average="weighted")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, OUT_PATH, compress=3)
    size_mb = OUT_PATH.stat().st_size / (1024 * 1024)
    print(f"Saved {OUT_PATH} ({size_mb:.1f} MB) | Held-out weighted F1: {weighted_f1:.4f}")


if __name__ == "__main__":
    main()
