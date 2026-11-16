# Minimal inference demo

Run from the repository root using the project's installed environment:

```sh
.venv/bin/python -m streamlit run app/streamlit_app.py
```

Open **Modeling → Model Inference**, upload a JPG/PNG/WebP, enter a French product title and/or description. Each modality is classified automatically
and independently. Text updates when the field is confirmed or loses focus. Fusion appears
when both predictions are valid. Editing one modality reuses the other prediction; clearing
an input removes its result and the fusion. Three columns show top-three predictions.

The demo loads the local CamemBERT directory and
`models/outputs/03_7_convnext_tiny_optimized_best.pt` independently on first use and caches each model per server process.
It does not download weights, train models, or persist uploaded images.
CPU inference is used for portability; the first request includes model loading.
Hosting must include these artifacts and enough RAM for both models.

Text cleaning and the 128-token limit match `scripts/train_camembert.py` and
text preprocessing. Image cropping, resize, normalization and horizontal-flip
averaging match notebook 03_7. ConvNeXt outputs are reordered by checkpoint class
mapping to the ascending category order used during CamemBERT training.

**Scope:** fusion is the equal mean of CamemBERT and ConvNeXt probabilities.
It is not the report's three-model ensemble and must not be presented with that
ensemble's evaluation score. The fitted calibrated SVM and TF-IDF vectorizer
are not saved in the repository; exporting and integrating them is a follow-up.

For a presentation, prepare a matching product image and French title before
opening the app. Person A presents text predictions, person B presents image
predictions and fusion. New inputs clear previous results. Model errors are
logged on the server and shown as a readable message in the app.

## Current local model issue

The CamemBERT safetensors file was checked directly and contains NaN weights.
The app therefore shows a clear warning for text and fusion while still computing
the real ConvNeXt prediction. Replace `models/camembert` with a valid trained
export and restart Streamlit to enable text and fusion. No model weights were
modified by this implementation.
