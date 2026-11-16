"""Automatic, independent text and image classification with late fusion."""
import logging
from time import perf_counter

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from src.categories import CATEGORY_NAMES, category_label
from src.text_utils import clean_text, combine_text


@st.cache_resource(show_spinner=False)
def load_predictor(modality):
    # Cache each model separately: image uploads never need the text model to load.
    from src.inference import ImagePredictor, TextPredictor
    return TextPredictor() if modality == "text" else ImagePredictor()


def clear_result(modality):
    # Editing text invalidates only text; changing the upload invalidates only image.
    st.session_state.pop(f"inference_{modality}", None)


def classify(modality, *inputs):
    # Keep predictions/errors in this user's session, not in the shared model cache.
    # Unrelated page reruns reuse the result instead of running the model again.
    state_key = f"inference_{modality}"
    if state_key not in st.session_state:
        try:
            with st.spinner(f"Classifying {modality}… The first run also loads the model."):
                predictor = load_predictor(modality)
                started   = perf_counter()
                scores    = predictor.predict(*inputs)
                st.session_state[state_key] = {"scores": scores, "seconds": perf_counter() - started, "error": None}
        except Exception as exc:
            logging.getLogger(__name__).exception("%s inference failed", modality)
            error = str(exc) if isinstance(exc, ValueError) else f"{modality.title()} prediction failed. Check the model files and server log."
            st.session_state[state_key] = {"scores": None, "error": error}
    return st.session_state[state_key]


# Product input: Streamlit reruns the page when an upload or committed field changes.
st.title   ("What product is this?")
st.caption ("Upload a product image or enter French text · Predictions update automatically")
st.info    ("Minimal version: CamemBERT + ConvNeXt, fused with equal weights. The report's additional SVM model is not included.")

image_column, text_column = st.columns([1, 1.4])
image = None

with image_column:
    uploaded = st.file_uploader("Product image", type=["jpg", "jpeg", "png", "webp"], on_change=clear_result, args=("image",))
    if uploaded is not None:
        try:
            # EXIF rotation makes phone photos appear in the intended orientation.
            with Image.open(uploaded) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            st.image(image, caption=uploaded.name, use_container_width=True)
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            st.error(f"Cannot read this image: {exc}")

with text_column:
    title       = st.text_input("Product title", placeholder="e.g. Piscine gonflable familiale", on_change=clear_result, args=("text",))
    description = st.text_area("Description (optional)", height=150, placeholder="Add the product description in French…", on_change=clear_result, args=("text",))
    st.caption("Text updates after you confirm or leave the field. Either field can be used on its own.")

# Run only the modality that has input and no current result. Clearing an input
# removes its prediction, so the fusion can never retain results from an old product.
text_result = image_result = None
if combine_text(clean_text(title), clean_text(description)):
    text_result = classify("text", title, description)
else:
    clear_result("text")

if image is not None:
    image_result = classify("image", image)
else:
    clear_result("image")

# Fusion is inexpensive: recompute it from the two current results on each rerun.
text_scores   = text_result["scores"] if text_result else None
image_scores  = image_result["scores"] if image_result else None
fusion_scores = None
if text_scores is not None and image_scores is not None:
    from src.inference import fuse_scores
    fusion_scores = fuse_scores(text_scores, image_scores)

st.subheader("Compare predictions")
panels = [("Text", "CamemBERT", text_scores, text_result), ("Image", "ConvNeXt · optimized", image_scores, image_result),
          ("Fusion", "50% text + 50% image", fusion_scores, None)]
class_codes = sorted(CATEGORY_NAMES)

for column, (heading, model, values, result) in zip(st.columns(3), panels):
    with column, st.container(border=True):
        st.subheader(heading)
        st.caption(model)
        if result and result["error"]:
            st.warning(result["error"])
        elif values is None:
            st.write("Requires valid text and image predictions." if heading == "Fusion" else f"Add {heading.lower()} to classify.")
        else:
            # Top-three rows use category codes as well as the team's readable labels.
            top = values.argsort()[::-1][:3]
            st.markdown(f"**{category_label(class_codes[top[0]])}**")
            rows = pd.DataFrame({"Category": [category_label(class_codes[i]) for i in top], "Score": [f"{values[i]:.1%}" for i in top]})
            st.dataframe(rows, hide_index=True, use_container_width=True)
            if result:
                st.caption(f"Inference: {result['seconds']:.1f} s (excluding model loading).")

with st.expander("How does fusion work?"):
    st.write("Text and image are classified independently. Once both predictions are available, fusion averages their scores for each of the 27 categories.")
    st.write("ConvNeXt uses smart white-border cropping and horizontal-flip averaging, as in notebook 03_7. This is not the report's three-model ensemble, which also includes a calibrated SVM.")

st.caption("Scores are model estimates, not guaranteed correctness. The true class is unknown for custom uploads. Category names are the team's interpretation of Rakuten codes.")
