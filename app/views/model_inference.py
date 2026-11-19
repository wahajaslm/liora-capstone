"""Multimodal text and image classification with equal-weight late fusion."""
import logging
from pathlib import Path
from time import perf_counter

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from data_loader import load_data
from src.categories import CATEGORY_NAMES, category_label
from src.data_utils import get_image_path
from src.inference import ImagePredictor, SVMPredictor, TextPredictor, fuse_scores
from src.text_utils import clean_text, combine_text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "text_train_processed.csv"
# Saved held-out scores of the three models, the same files the report fuses.
SAVED_SCORES = {
    "svm": PROJECT_ROOT / "data" / "processed" / "text_validation_probabilities.csv",
    "camembert": PROJECT_ROOT / "data" / "processed" / "camembert_validation_probabilities.csv",
    "image": PROJECT_ROOT / "models" / "outputs" / "03_7_convnext_tiny_optimized_validation_predictions.csv",
}
EXAMPLES = {
    "image_helps": "Text wrong, image right",
    "text_helps": "Image wrong, text right",
    "mistake": "Fusion wrong",
}
CLASS_CODES = sorted(CATEGORY_NAMES)

MODEL_BUILDERS = {
    "svm": SVMPredictor,
    "camembert": TextPredictor,
    "image": ImagePredictor,
}
RESULT_KEYS = ("inference_svm", "inference_camembert", "inference_image")

# One fixed colour per model, from a colour-blind-checked palette.
MODEL_COLORS = {"LinearSVC": "#2a78d6", "CamemBERT": "#eb6834", "ConvNeXt": "#1baf7a", "Fusion": "#eda100"}


@st.cache_resource(show_spinner=False)
def load_predictor(modality: str):
    """Cache each model separately so unused modalities do not load."""
    return MODEL_BUILDERS[modality]()


@st.cache_data(show_spinner=False)
def load_heldout_catalog() -> pd.DataFrame | None:
    """Load the 16,984 held-out products with raw titles, descriptions, and image paths."""
    if not PROCESSED_PATH.exists():
        return None
    try:
        raw = load_data()
        proc = pd.read_csv(PROCESSED_PATH, usecols=["row_id", "split"])
    except Exception:
        return None
    val_ids = proc.loc[proc["split"] == "validation", "row_id"].tolist()
    val_ids = [idx for idx in val_ids if idx in raw.index]
    if not val_ids:
        return None
    subset = raw.loc[val_ids, ["designation", "description", "imageid", "productid", "prdtypecode"]].copy()
    subset.insert(0, "row_id", subset.index)
    return subset.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_example_rows() -> dict[str, list[int]] | None:
    """Group held-out row IDs by which models got them right, from saved scores."""
    if not all(path.exists() for path in SAVED_SCORES.values()):
        return None
    tables = {}
    for name, path in SAVED_SCORES.items():
        table = pd.read_csv(path, index_col="row_id")
        if name == "image":
            truth = table["true_prdtypecode"]
            table = table[[c for c in table if c.startswith("prob_")]].rename(columns=lambda c: c[5:])
        tables[name] = table[[str(code) for code in CLASS_CODES]]
    rows = tables["svm"].index
    codes = np.array(CLASS_CODES)
    preds = {name: codes[table.loc[rows].to_numpy().argmax(1)] for name, table in tables.items()}
    fused = sum(table.loc[rows].to_numpy() for table in tables.values()) / len(tables)
    preds["fusion"] = codes[fused.argmax(1)]
    true = truth.loc[rows].to_numpy()
    right = {name: pred == true for name, pred in preds.items()}
    masks = {
        "image_helps": ~right["svm"] & ~right["camembert"] & right["image"] & right["fusion"],
        "text_helps": right["svm"] & right["camembert"] & ~right["image"] & right["fusion"],
        "mistake": ~right["fusion"],
    }
    return {case: rows[mask].tolist() for case, mask in masks.items()}


def load_example_callback(catalog: pd.DataFrame, rows: list[int]) -> None:
    match = catalog[catalog["row_id"].isin(rows)]
    if not match.empty:
        fill_from_row(match.sample(n=1).iloc[0])


def top_chart(scores, picks: list[int], true_code: int | None, color: str) -> alt.Chart:
    """One model's scores for a fixed list of categories, in a fixed order."""
    data = pd.DataFrame({
        "Category": [
            f"{CATEGORY_NAMES[CLASS_CODES[i]]} (true)" if CLASS_CODES[i] == true_code else CATEGORY_NAMES[CLASS_CODES[i]]
            for i in picks
        ],
        "Score": [float(scores[i]) for i in picks],
    })
    base = alt.Chart(data).encode(
        y=alt.Y("Category:N", sort=None, title=None, axis=alt.Axis(labelFontSize=12, labelLimit=130)),
        x=alt.X("Score:Q", title=None, scale=alt.Scale(domain=[0, 1.3]), axis=None),
    )
    bars = base.mark_bar(cornerRadiusEnd=4, color=color).encode(
        tooltip=["Category", alt.Tooltip("Score:Q", format=".0%")],
    )
    labels = base.mark_text(align="left", dx=4, fontSize=12).encode(text=alt.Text("Score:Q", format=".0%"))
    return (bars + labels).properties(height=alt.Step(26))


def clear_text_results() -> None:
    st.session_state.pop("inference_svm", None)
    st.session_state.pop("inference_camembert", None)


def on_text_edit() -> None:
    # Hand-edited text no longer matches the sample, so hide its true label.
    clear_text_results()
    if st.session_state["inf_sample"] is not None:
        st.session_state["inf_sample"]["edited"] = True


def on_image_upload() -> None:
    st.session_state.pop("inference_image", None)
    st.session_state["inf_sample"] = None


def clear_all_results() -> None:
    for key in RESULT_KEYS:
        st.session_state.pop(key, None)


def load_sample_callback(catalog: pd.DataFrame, category_choice: str) -> None:
    if category_choice == "All categories":
        pool = catalog
    else:
        code = int(category_choice.rsplit("(", 1)[-1].rstrip(")"))
        pool = catalog[catalog["prdtypecode"] == code]
    if pool.empty:
        return
    fill_from_row(pool.sample(n=1).iloc[0])


def load_row_callback(catalog: pd.DataFrame) -> None:
    row_id = st.session_state["inf_row_id"]
    match = catalog[catalog["row_id"] == row_id]
    if row_id is None or match.empty:
        st.session_state["inf_row_error"] = f"Row {row_id} is not in the held-out set."
        return
    fill_from_row(match.iloc[0])


def fill_from_row(row: pd.Series) -> None:
    st.session_state["inf_row_error"] = None
    st.session_state["inf_title"] = "" if pd.isna(row["designation"]) else str(row["designation"])
    st.session_state["inf_desc"] = "" if pd.isna(row["description"]) else str(row["description"])
    st.session_state["inf_sample"] = {
        "row_id": int(row["row_id"]),
        "prdtypecode": int(row["prdtypecode"]),
        "image_path": get_image_path(row["imageid"], row["productid"]),
        "edited": False,
    }
    st.session_state["inf_uploader_key"] += 1
    clear_all_results()


def reset_inputs_callback() -> None:
    st.session_state["inf_row_error"] = None
    st.session_state["inf_title"] = ""
    st.session_state["inf_desc"] = ""
    st.session_state["inf_sample"] = None
    st.session_state["inf_uploader_key"] += 1
    clear_all_results()


def classify(modality: str, label: str, *inputs) -> dict:
    """Run and cache a single model's prediction in session state."""
    state_key = f"inference_{modality}"
    if state_key not in st.session_state:
        try:
            with st.spinner(f"Running {label}… (first run also loads the model)"):
                predictor = load_predictor(modality)
                started = perf_counter()
                scores = predictor.predict(*inputs)
                st.session_state[state_key] = {
                    "scores": scores,
                    "seconds": perf_counter() - started,
                    "error": None,
                }
        except Exception as exc:
            logging.getLogger(__name__).exception("%s inference failed", modality)
            error = (
                str(exc)
                if isinstance(exc, (ValueError, FileNotFoundError))
                else f"{label} prediction failed. Check the model files and server log."
            )
            st.session_state[state_key] = {"scores": None, "error": error}
    return st.session_state[state_key]




st.session_state.setdefault("inf_title", "")
st.session_state.setdefault("inf_desc", "")
st.session_state.setdefault("inf_sample", None)
st.session_state.setdefault("inf_uploader_key", 0)
st.session_state.setdefault("inf_row_error", None)

st.title("What product is this?")
st.caption("Pick a product and three models guess its category, out of 27. Two models read the text, one looks at the image. Fusion averages the three guesses into one final answer.")

# Step 1: pick a product, either from the held-out set or your own.
catalog = load_heldout_catalog()
with st.container(border=True):
    st.subheader("1. Pick a product")
    if catalog is not None:
        category_options = ["All categories"] + [category_label(code) for code in CLASS_CODES]
        pick_column, example_column = st.columns([3, 2], gap="large")
        with pick_column:
            with st.container(horizontal=True, vertical_alignment="bottom"):
                selected_category = st.selectbox(
                    "Category",
                    category_options,
                    key="inf_sample_filter",
                    width=220,
                )
                st.button(
                    "Random product",
                    type="primary",
                    on_click=load_sample_callback,
                    args=(catalog, selected_category),
                )
            with st.container(horizontal=True, vertical_alignment="bottom"):
                st.number_input(
                    "Or a row ID",
                    min_value=0,
                    step=1,
                    value=None,
                    placeholder="e.g. 1234",
                    key="inf_row_id",
                    width=220,
                )
                st.button(
                    "Load row",
                    on_click=load_row_callback,
                    args=(catalog,),
                )
                st.button(
                    "Clear",
                    type="tertiary",
                    on_click=reset_inputs_callback,
                )
        with example_column:
            example_rows = load_example_rows()
            if example_rows:
                st.markdown("**Show an example where:**")
                for case, label in EXAMPLES.items():
                    st.button(
                        label,
                        key=f"example_{case}",
                        help=f"{len(example_rows[case]):,} held-out products like this",
                        on_click=load_example_callback,
                        args=(catalog, example_rows[case]),
                        width="stretch",
                    )
        st.caption(f"Products come from the {len(catalog):,} held-out products the models never saw in training. You can also upload your own image or type your own text below.")
        if st.session_state["inf_row_error"]:
            st.warning(st.session_state["inf_row_error"])

    image_column, text_column = st.columns([1, 1.6], gap="medium")
    image = None

    with image_column:
        uploaded = st.file_uploader(
            "Product image",
            type=["jpg", "jpeg", "png", "webp"],
            key=f"inf_uploader_{st.session_state['inf_uploader_key']}",
            on_change=on_image_upload,
        )
        sample = st.session_state["inf_sample"]
        image_source = uploaded if uploaded is not None else (sample["image_path"] if sample else None)
        if image_source is not None:
            try:
                with Image.open(image_source) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                image_caption = uploaded.name if uploaded is not None else f"Held-out row {sample['row_id']}"
                st.image(image, caption=image_caption, width=320)
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
                st.error(f"Cannot read this image: {exc}")

    with text_column:
        title = st.text_input(
            "Product title",
            key="inf_title",
            placeholder="e.g. Piscine gonflable familiale",
            on_change=on_text_edit,
        )
        description = st.text_area(
            "Description (optional)",
            key="inf_desc",
            height=180,
            placeholder="Add the product description in French…",
            on_change=on_text_edit,
        )
        with st.container(horizontal=True, vertical_alignment="center"):
            st.button("Classify", type="primary")
            st.caption("Press Enter or click Classify after typing.")

has_text = bool(combine_text(clean_text(title), clean_text(description)))
svm_result = camembert_result = image_result = None

if has_text:
    svm_result = classify("svm", "LinearSVC", title, description)
    camembert_result = classify("camembert", "CamemBERT", title, description)
else:
    clear_text_results()

if image is not None:
    image_result = classify("image", "ConvNeXt", image)
else:
    st.session_state.pop("inference_image", None)

svm_scores = svm_result["scores"] if svm_result else None
camembert_scores = camembert_result["scores"] if camembert_result else None
image_scores = image_result["scores"] if image_result else None

fusion_scores = None
if image_scores is not None and (svm_scores is not None or camembert_scores is not None):
    available = [s for s in (svm_scores, camembert_scores, image_scores) if s is not None]
    fusion_scores = fuse_scores(*available)

sample = st.session_state["inf_sample"]
true_code = sample["prdtypecode"] if sample is not None and not sample["edited"] else None

# Step 2: the final answer, then one small chart per model on the same categories.
models = [
    ("LinearSVC", "Reads the text: which words appear", svm_scores, svm_result, "text"),
    ("CamemBERT", "Reads the text: French sentences", camembert_scores, camembert_result, "text"),
    ("ConvNeXt", "Looks at the image", image_scores, image_result, "an image"),
    ("Fusion", "Average of the three models", fusion_scores, None, "text and an image"),
]

with st.container(border=True):
    st.subheader("2. Result")
    for name, _, _, result, _ in models:
        if result and result["error"]:
            st.warning(f"{name}: {result['error']}")

    with st.container(horizontal=True, vertical_alignment="center", gap="large"):
        if fusion_scores is None:
            st.metric("Final answer", "Waiting", width="content")
            st.caption("Add both text and an image for the final answer.")
        else:
            pred_code = CLASS_CODES[int(fusion_scores.argmax())]
            st.metric("Final answer", CATEGORY_NAMES[pred_code], help=f"Rakuten code {pred_code}", width="content")
            st.metric("Confidence", f"{fusion_scores.max():.0%}", width="content")
            if true_code is not None:
                if pred_code == true_code:
                    st.badge("Correct", color="green")
                else:
                    st.badge("Wrong", color="red")
        if true_code is not None:
            st.caption(f"True category: **{category_label(true_code)}**")

    available = [values for _, _, values, _, _ in models if values is not None]
    if available:
        # Same rows in every chart: the top 5 overall, plus each model's own first choice.
        ranking = fusion_scores if fusion_scores is not None else np.mean(available, axis=0)
        picks = [int(i) for i in ranking.argsort()[::-1][:5]]
        picks += [int(v.argmax()) for v in available if int(v.argmax()) not in picks]
        picks.sort(key=lambda i: -ranking[i])

        st.caption("Each chart shows how sure one model is about the same categories. Read across to compare.")
        for column, (name, uses, values, _, needs) in zip(st.columns(4, gap="small"), models):
            with column, st.container(border=True, height="stretch"):
                with st.container(horizontal=True, vertical_alignment="center"):
                    st.markdown(f"**{name}**")
                    st.space("stretch")
                    if values is not None and true_code is not None:
                        hit = CLASS_CODES[int(values.argmax())] == true_code
                        st.badge("Correct" if hit else "Wrong", color="green" if hit else "red")
                st.caption(uses)
                if values is None:
                    st.caption(f"Needs {needs}.")
                else:
                    st.altair_chart(top_chart(values, picks, true_code, MODEL_COLORS[name]), width="stretch")
    else:
        st.caption("Add text or an image to see the scores.")

with st.expander("How does fusion work?"):
    st.write("LinearSVC, CamemBERT and ConvNeXt classify independently. Each gives a score for all 27 categories. Fusion takes the plain average of the three and picks the highest.")
    st.write("This is the final ensemble in the report: 0.8941 weighted F1 on the 16,984 held-out products.")

st.caption("Scores are model estimates. Category names are the team's reading of Rakuten's numeric codes.")
