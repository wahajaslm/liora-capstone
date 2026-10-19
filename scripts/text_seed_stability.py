"""Measure TF-IDF text-model sensitivity to the train/validation split seed.

Only the random seed used by ``train_test_split`` changes between runs. The
vectorizer, classifier, validation size, and class stratification stay fixed.
"""

import html
import re
import unicodedata
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data/raw/X_train_update.csv"
LABELS_PATH = PROJECT_ROOT / "data/raw/Y_train_CVw08PX.csv"
SPLIT_SEEDS = [0, 1, 2, 3, 4, 42]

# same multilingual vocabulary cleaning used in the preprocessing notebook.
FRENCH_STOPWORDS = [
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou",
    "en", "dans", "avec", "pour", "sur", "par", "au", "aux",
    "ce", "ces", "cette",
]
GERMAN_STOPWORDS = [
    "der", "die", "das", "und", "von", "mit", "zu", "ist",
    "im", "den", "dem", "ein", "eine", "für", "auf",
]
STOPWORDS = FRENCH_STOPWORDS + GERMAN_STOPWORDS + sorted(ENGLISH_STOP_WORDS)


def strip_html(text):
    """Decode HTML entities and remove HTML tags while retaining their text."""
    if not isinstance(text, str):
        return ""
    decoded = html.unescape(html.unescape(text))
    if not re.search(r"</?[a-zA-Z][^>]*>", decoded):
        return decoded
    soup = BeautifulSoup(decoded, "html.parser")
    return soup.get_text(separator=" ")


def normalize_accents(text):
    """Convert accented Unicode characters to their ASCII equivalents."""
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()


def clean_text(text, remove_accents=False):
    """Remove formatting noise while preserving product information."""
    if not isinstance(text, str):
        return ""
    text = strip_html(text)
    if remove_accents:
        text = normalize_accents(text)
    return " ".join(text.lower().split())


def combine_text(designation, description):
    """Join designation and description without an unnecessary separator."""
    return " ".join(part for part in (designation, description) if part)


def prepare_text(data):
    """apply the same text-preprocessing workflow used in the notebook."""
    prepared = data.copy()
    prepared["description_available"] = prepared["description"].notna()
    prepared["designation_clean"] = (
        prepared["designation"].fillna("").map(clean_text)
    )
    prepared["description_clean"] = (
        prepared["description"].fillna("").map(clean_text)
    )
    prepared["text_combined"] = [
        combine_text(designation, description)
        for designation, description in zip(
            prepared["designation_clean"],
            prepared["description_clean"],
        )
    ]
    return prepared


def load_experiment_data():
    """Load the raw files and create both versions of the model text."""
    data = pd.read_csv(RAW_PATH, index_col=0)
    labels = pd.read_csv(LABELS_PATH, index_col=0)

    data["prdtypecode"] = labels["prdtypecode"]
    data["text_raw"] = (
        data["designation"].fillna("")
        + " "
        + data["description"].fillna("")
    )
    data = prepare_text(data)

    if data["prdtypecode"].isna().any():
        raise ValueError("Some raw products could not be matched to a label.")
    return data


def train_model(data, train_rows, validation_rows, text_column):
    """Fit on training text and transform validation text without refitting."""
    vectorizer = TfidfVectorizer(
        max_features=50_000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words=STOPWORDS,
    )
    x_train = vectorizer.fit_transform(data.loc[train_rows, text_column])
    x_validation = vectorizer.transform(data.loc[validation_rows, text_column])

    model = LinearSVC(random_state=42)
    model.fit(x_train, data.loc[train_rows, "prdtypecode"])
    return model, x_validation


def evaluate_model(data, validation_rows, model, x_validation):
    """Predict validation labels and calculate weighted and macro F1."""
    predictions = model.predict(x_validation)
    labels = data.loc[validation_rows, "prdtypecode"]

    return {
        "weighted_f1": f1_score(labels, predictions, average="weighted"),
        "macro_f1": f1_score(labels, predictions, average="macro"),
    }


def run_experiment(seeds):
    data = load_experiment_data()
    results = []

    for seed in seeds:
        print(f"Starting split seed {seed}...", flush=True)
        train_rows, validation_rows = train_test_split(
            data.index,
            test_size=0.20,
            random_state=seed,
            stratify=data["prdtypecode"],
        )

        for text_version, text_column in (
            ("raw", "text_raw"),
            ("cleaned", "text_combined"),
        ):
            print(f"  Training {text_version} text...", flush=True)
            model, x_validation = train_model(
                data,
                train_rows,
                validation_rows,
                text_column,
            )
            scores = evaluate_model(
                data,
                validation_rows,
                model,
                x_validation,
            )
            results.append({"seed": seed, "text": text_version, **scores})
            print(
                f"seed={seed:>3} text={text_version:<7} "
                f"weighted_f1={scores['weighted_f1']:.6f} "
                f"macro_f1={scores['macro_f1']:.6f}",
                flush=True,
            )

    return pd.DataFrame(results)


def print_summary(results):
    print("\nScore variation across split seeds")
    print(
        results.groupby("text")[["weighted_f1", "macro_f1"]]
        .agg(["mean", "std", "min", "max"])
        .round(6)
    )

    paired = results.pivot(index="seed", columns="text", values="weighted_f1")
    paired["cleaned_minus_raw"] = paired["cleaned"] - paired["raw"]
    print("\nRaw-versus-cleaned weighted F1 for each seed")
    print(paired.round(6))
    print(
        "\nMean cleaned-minus-raw difference: "
        f"{paired['cleaned_minus_raw'].mean():+.6f}"
    )


if __name__ == "__main__":
    experiment_results = run_experiment(SPLIT_SEEDS)
    print_summary(experiment_results)
