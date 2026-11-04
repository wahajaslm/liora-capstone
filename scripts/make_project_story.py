"""Build the visual project story: one page per step, with concepts, plots and examiner questions.

Run from the project root, after the notebooks have been run:

    python scripts/make_project_story.py

Output: reports/project_story/project_story.pdf (and the HTML used to build it).
"""

import base64
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from weasyprint import HTML

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "reports/project_story"
FIG_DIR = OUT_DIR / "figures"

INK, MUTED, LINE, ACCENT, ORANGE, RED = "#1f2933", "#6b7280", "#d1d5db", "#2b5c8f", "#e07b39", "#c0504d"

# Results (weighted F1 on the 16984 validation products)
CAMEMBERT_F1 = 0.8765
FUSION_BEST_F1 = 0.8937


# ---------- figure helpers ----------

def style(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=INK)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def bar_h(names, values, title, xlabel, name, fmt="{:.4f}", xlim=None, highlight=None, size=(6, 2.6)):
    fig, ax = plt.subplots(figsize=size)
    colors = [ORANGE if highlight and n == highlight else ACCENT for n in names]
    bars = ax.barh(names[::-1], values[::-1], color=colors[::-1], height=0.6)
    ax.bar_label(bars, labels=[fmt.format(v) for v in values[::-1]], padding=3, fontsize=8.5, color=INK)
    if xlim:
        ax.set_xlim(*xlim)
    ax.grid(axis="x", color=LINE, linewidth=0.7)
    style(ax, title, xlabel)
    return save(fig, name)


def notebook_image(notebook, cell, name, index=0):
    nb = json.load(open(PROJECT_ROOT / "notebooks" / notebook))
    images = [o["data"]["image/png"] for o in nb["cells"][cell].get("outputs", []) if "image/png" in o.get("data", {})]
    path = FIG_DIR / f"{name}.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(images[index]))
    return path


def make_figures():
    figs = {}
    data = pd.read_csv(PROJECT_ROOT / "data/processed/text_train_processed.csv",
                       usecols=["prdtypecode", "description_available", "split"])

    # Category distribution
    share = data["prdtypecode"].value_counts(normalize=True).sort_values(ascending=False) * 100
    fig, ax = plt.subplots(figsize=(6.2, 2.4))
    ax.bar([str(c) for c in share.index], share.values, color=ACCENT, width=0.7)
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    ax.grid(axis="y", color=LINE, linewidth=0.7)
    style(ax, "Share of products per category (%)", ylabel="% of products")
    figs["classes"] = save(fig, "classes")

    # Split
    figs["baselines"] = bar_h(["Majority class", "Image (frozen ResNet18)", "Text (TF-IDF + SVM)"],
                              [0.026, 0.4817, 0.8274], "First baselines", "Weighted F1", "baselines",
                              xlim=(0, 1))

    # Seed check
    seeds = pd.DataFrame({"seed": [0, 1, 2, 3, 4],
                          "normal": [0.8260, 0.8213, 0.8242, 0.8258, 0.8284],
                          "binary": [0.8311, 0.8254, 0.8285, 0.8255, 0.8316]})
    fig, ax = plt.subplots(figsize=(6, 2.6))
    for _, r in seeds.iterrows():
        ax.plot([r["normal"], r["binary"]], [r["seed"]] * 2, color=LINE, linewidth=2, zorder=1)
    ax.scatter(seeds["normal"], seeds["seed"], color=MUTED, s=40, zorder=2, label="Normal TF-IDF")
    ax.scatter(seeds["binary"], seeds["seed"], color=ACCENT, s=40, zorder=2, label="Binary TF-IDF")
    ax.axvspan(seeds["normal"].min(), seeds["normal"].max(), color=ORANGE, alpha=0.12, zorder=0)
    ax.text(seeds["normal"].min() + 0.0002, 4.45, "spread between seeds: 0.71 points", fontsize=8, color=ORANGE)
    ax.set_yticks(seeds["seed"], [f"seed {s}" for s in seeds["seed"]])
    ax.set_ylim(-0.6, 4.8)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    style(ax, "Binary vs normal TF-IDF on 5 different splits", "Weighted F1")
    figs["seeds"] = save(fig, "seeds")

    # Feature limit
    fig, ax = plt.subplots(figsize=(6, 2.6))
    labels, vals = ["20000", "50000", "100000", "all (460326)"], [0.8027, 0.8164, 0.8231, 0.8284]
    ax.plot(labels, vals, color=ACCENT, marker="o", linewidth=2)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.4f}", (i, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5)
    ax.set_ylim(0.795, 0.835)
    ax.grid(axis="y", color=LINE, linewidth=0.7)
    style(ax, "Number of TF-IDF features kept (3-fold CV on train)", "Features kept", "CV weighted F1")
    figs["features"] = save(fig, "features")

    # Grid search heatmap
    grid = np.array([[0.7999, 0.8223, 0.8235, 0.8231], [0.8031, 0.8263, 0.8284, 0.8282]])
    fig, ax = plt.subplots(figsize=(5.2, 1.9))
    ax.imshow(grid, cmap="Blues", vmin=0.79, vmax=0.835, aspect="auto")
    for (i, j), v in np.ndenumerate(grid):
        ax.text(j, i, f"{v:.4f}", ha="center", va="center", fontsize=9,
                color="white" if v > 0.82 else INK, fontweight="bold" if v == grid.max() else "normal")
    ax.set_xticks(range(4), ["C = 0.1", "C = 0.5", "C = 1", "C = 2"])
    ax.set_yticks(range(2), ["normal", "binary"])
    style(ax, "Grid search: CV weighted F1 for each setting")
    ax.spines[["left", "bottom"]].set_visible(False)
    figs["grid"] = save(fig, "grid")

    figs["models"] = bar_h(["LinearSVC", "Logistic Regression", "Complement Naive Bayes"],
                           [0.8284, 0.8084, 0.7600], "Other text models (3-fold CV on train)",
                           "CV weighted F1", "models", xlim=(0.7, 0.86), highlight="LinearSVC")

    # CamemBERT loss curves
    log_path = PROJECT_ROOT / "data/processed/camembert_training_log.csv"
    log = pd.read_csv(log_path)
    first = (log["epoch"] == 2) & (log["step"] == log.loc[log["epoch"] == 2, "step"].min())
    log.loc[first, "train_loss"] *= 500 / (log.loc[first, "step"] - 4246)  # see notebook 05
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    ax.plot(log["step"], log["train_loss"], color=ACCENT, marker="o", markersize=3, linewidth=2, label="Train loss")
    if "validation_loss" in log:
        ax.plot(log["step"], log["validation_loss"], color=ORANGE, marker="o", markersize=3, linewidth=2,
                label="Validation loss")
    ax.axvline(4246, color=LINE, linestyle="--")
    ax.text(4300, ax.get_ylim()[1] * 0.9, "end of epoch 1", fontsize=8, color=MUTED)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", color=LINE, linewidth=0.7)
    style(ax, "CamemBERT training", "Training step (batch of 16)", "Loss")
    figs["loss"] = save(fig, "loss")

    # Learning rate schedule
    steps = np.arange(8492)
    warm = int(0.06 * 8492)
    lr = np.where(steps < warm, 5e-5 * steps / warm, 5e-5 * (8492 - steps) / (8492 - warm))
    fig, ax = plt.subplots(figsize=(3.2, 1.8))
    ax.plot(steps, lr * 1e5, color=ACCENT, linewidth=2)
    ax.set_yticks([0, 5], ["0", "5e-5"])
    style(ax, "Learning rate", "Step")
    figs["lr"] = save(fig, "lr")

    figs["image_models"] = bar_h(["Frozen ResNet18", "ResNet18, last block trained", "ResNet18, full network",
                                  "ConvNeXt-tiny"], [0.4817, 0.6238, 0.6281, 0.6970],
                                 "Image models", "Weighted F1", "image_models", xlim=(0, 0.8),
                                 highlight="ConvNeXt-tiny")

    # Image fine-tuning: train vs validation loss (teammate's run)
    epochs = [1, 2, 3, 4, 5]
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.plot(epochs, [0.5281, 0.4325, 0.3594, 0.3020, 0.2634], color=ACCENT, marker="o", label="Train loss")
    ax.plot(epochs, [1.4390, 1.4848, 1.6338, 1.6444, 1.6761], color=ORANGE, marker="o", label="Validation loss")
    ax.legend(frameon=False, fontsize=7.5)
    style(ax, "Full ResNet18 fine-tuning", "Epoch", "Loss")
    figs["image_overfit"] = save(fig, "image_overfit")

    names = ["SVM baseline", "SVM optimised", "CamemBERT", "Fusion\n(SVM + image)", "Fusion\n(all three)"]
    vals = [0.8274, 0.8436, CAMEMBERT_F1, 0.8630, FUSION_BEST_F1]
    fig, ax = plt.subplots(figsize=(7, 2.3))
    ax.plot(range(len(names)), vals, color=ACCENT, marker="o", linewidth=2)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.4f}", (i, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8.5)
    ax.set_xticks(range(len(names)), names, fontsize=8)
    ax.set_ylim(0.815, 0.905)
    ax.grid(axis="y", color=LINE, linewidth=0.7)
    style(ax, None, None, "Weighted F1")
    figs["progress"] = save(fig, "progress")

    # Reused notebook figures
    figs["errors"] = notebook_image("03_text_baseline.ipynb", 31, "errors")
    figs["camembert_gain"] = notebook_image("05_text_camembert.ipynb", 9, "camembert_gain")
    figs["fusion"] = notebook_image("06_fusion.ipynb", 12, "fusion")
    figs["words"] = notebook_image("07_text_interpretability.ipynb", 7, "words")
    return figs


# ---------- page content ----------

CSS = """
@page landscape { size: A4 landscape; margin: 10mm; }
.landscape { page: landscape; }
@page { size: A4; margin: 14mm 14mm 14mm 14mm;
        @bottom-right { content: counter(page); font: 8pt Helvetica, Arial, sans-serif; color: #9ca3af; } }
body { font-family: Helvetica, Arial, sans-serif; color: #1f2933; font-size: 9.6pt; line-height: 1.42; }
.page { page-break-after: always; }
.page:last-child { page-break-after: auto; }
.step { font-size: 8pt; letter-spacing: 0.08em; color: #d0573a; font-weight: bold; margin: 0 0 2mm 0; }
h1 { font-size: 20pt; margin: 0 0 1mm 0; }
h2 { font-size: 16pt; margin: 0 0 3mm 0; }
h3 { font-size: 9pt; letter-spacing: 0.06em; color: #6b7280; margin: 4mm 0 1.5mm 0; text-transform: uppercase; }
.main { font-size: 11.5pt; font-weight: bold; color: #2b5c8f; border-left: 3px solid #2b5c8f;
        padding: 1mm 0 1mm 3mm; margin: 0 0 3mm 0; }
.cols { display: table; width: 100%; table-layout: fixed; border-spacing: 0; }
.col { display: table-cell; vertical-align: top; padding-right: 5mm; }
.col:last-child { padding-right: 0; }
.wide { margin: 2mm 0; }
ul { margin: 0; padding-left: 4.5mm; }
li { margin-bottom: 1mm; }
img { width: 100%; }
.concept { background: #f3f4f6; border-radius: 2mm; padding: 3mm 4mm; margin: 2mm 0; }
.concept b { color: #1f2933; }
.qa { border-top: 1px solid #d1d5db; padding-top: 2mm; margin-top: 2mm; }
.q { font-style: italic; margin: 1.5mm 0 0.5mm 0; }
.a { color: #4b5563; margin: 0 0 1.5mm 0; }
.flow { display: flex; align-items: center; gap: 2mm; font-size: 8.5pt; margin: 2mm 0; }
.box { border: 1px solid #9ca3af; border-radius: 1.5mm; padding: 1.5mm 2mm; text-align: center; background: white; }
.arrow { color: #9ca3af; }
.note { color: #6b7280; font-size: 8.5pt; }
table { border-collapse: collapse; font-size: 8.5pt; width: 100%; }
td, th { border-bottom: 1px solid #e5e7eb; padding: 1mm 2mm; text-align: left; }
th { color: #6b7280; font-weight: bold; }
.fold { display: inline-block; width: 13mm; height: 4mm; margin: 0.4mm; border-radius: 0.8mm; }
.tr { background: #bfd3e6; } .te { background: #e07b39; }
"""


def img(figs, key, width="100%", max_height=None):
    limit = f"max-height:{max_height};object-fit:contain;" if max_height else ""
    return f'<img src="{figs[key].as_uri()}" style="width:{width};{limit}">'


def qa(pairs):
    return '<div class="qa"><h3>Examiner questions</h3>' + "".join(
        f'<p class="q">{q}</p><p class="a">{a}</p>' for q, a in pairs) + "</div>"


def flow(items):
    parts = []
    for i, item in enumerate(items):
        if i:
            parts.append('<span class="arrow">&gt;</span>')
        parts.append(f'<div class="box">{item}</div>')
    return '<div class="flow">' + "".join(parts) + "</div>"


def folds():
    rows = []
    for k in range(3):
        cells = "".join(f'<span class="fold {"te" if i == k else "tr"}"></span>' for i in range(3))
        rows.append(f"<div>{cells} <span class='note'>round {k + 1}</span></div>")
    return "".join(rows)


def build_pages(figs):
    timeline = PROJECT_ROOT / "reports/figures/project_timeline_lanes.png"
    pages = []

    pages.append(f"""
<div class="page">
  <p class="step">PROJECT STORY</p>
  <h1>Rakuten product classification</h1>
  <p class="note">Classify 84916 products into 27 categories from their title, description and image.
  All scores are weighted F1 on the same 16984 validation products.</p>
  <div class="main">Best model: a Random Forest that combines CamemBERT, a TF-IDF SVM and an image model,
  with {FUSION_BEST_F1:.4f} weighted F1. Text carries most of the signal; images help on books.</div>
  <h3>The final pipeline</h3>
  {flow(["Title + description", "TF-IDF + SVM<br>27 probabilities"])}
  {flow(["Title + description", "CamemBERT<br>27 probabilities"])}
  {flow(["Product image", "ResNet18<br>27 probabilities"])}
  {flow(["81 probabilities", "Random Forest", f"Category (F1 {FUSION_BEST_F1:.4f})"])}
  <h3>How the score improved</h3>
  {img(figs, "progress")}
  <h3>Contents</h3>
  <table>
    <tr><td>1. Data audit</td><td>2. Preprocessing and shared split</td><td>3. Baselines</td></tr>
    <tr><td>4. Text optimisation</td><td>5. CamemBERT</td><td>6. Image models</td></tr>
    <tr><td>7. Fusion</td><td>8. Interpretability</td><td>9. Conclusions</td></tr>
  </table>
</div>
<div class="page landscape">
  <img src="{timeline.as_uri()}" style="width:100%;max-height:185mm;object-fit:contain">
</div>""")

    pages.append(f"""
<div class="page">
  <p class="step">STEP 1 &nbsp; 1 OCT</p>
  <h2>Data audit</h2>
  <div class="main">The text is rich but messy, and the categories are unbalanced. We score with weighted F1
  and also report macro F1.</div>
  <div class="cols">
    <div class="col">
      <h3>What we found</h3>
      <ul>
        <li>84916 products, 27 categories, each with a title, an optional description and one image.</li>
        <li>35% of products have no description. 18% of descriptions contain HTML.</li>
        <li>The largest category (2583, pool) has 12% of products, the smallest (1180) 0.9%: 13 times less.</li>
        <li>Category codes have no names. We named them by reading product titles.</li>
      </ul>
      <div class="concept"><b>Concept: weighted vs macro F1.</b> F1 combines precision (how many predicted
      products are right) and recall (how many true products are found). <b>Weighted F1</b> averages the 27
      category scores weighted by their size: it is the challenge metric. <b>Macro F1</b> gives every category
      the same weight, so it shows if small categories are ignored.</div>
    </div>
    <div class="col">{img(figs, "classes")}</div>
  </div>
  {qa([("Why not accuracy?", "With unbalanced categories, a model can ignore small ones and still look good. "
        "F1 per category, averaged, makes this visible."),
       ("Is the imbalance a problem for training?", "Not much here: the smallest category still has 764 products. "
        "We check it with macro F1 and per-category F1 instead of re-sampling.")])}
</div>""")

    pages.append(f"""
<div class="page">
  <p class="step">STEP 2 &nbsp; 8 OCT</p>
  <h2>Preprocessing and a shared split</h2>
  <div class="main">Every model is scored on the same 16984 validation products. Without this, text, image and
  fusion scores could not be compared or combined.</div>
  <div class="cols">
    <div class="col">
      <h3>Text</h3>
      <ul><li>HTML removed, text lowercased.</li><li>Title and description joined into one text.</li></ul>
      <h3>Images</h3>
      <ul><li>Duplicate images found by comparing a hash (fingerprint) of each file.</li>
          <li>5620 duplicate or conflicting images removed from <b>train only</b>.</li></ul>
      <h3>Split</h3>
      <ul><li>One stratified 80/20 split: 67932 train, 16984 validation, saved to a file.</li>
          <li>Stratified means each category has the same share in both parts.</li></ul>
    </div>
    <div class="col">
      <div class="concept"><b>Concept: why validation must stay untouched.</b> The validation set plays the role
      of new, unseen products. If we clean it, remove rows from it, or tune settings on it, its score becomes too
      optimistic. So all cleaning that removes products is done on train only.</div>
      {flow(["84916 products", "Train 67932<br>(cleaned)", "Validation 16984<br>(kept complete)"])}
      <p class="note">The mentor found that an early version of the image cleaning also removed 551 validation
      products. This was fixed: the image model now uses the same 16984 products as text.</p>
    </div>
  </div>
  {qa([("Why keep duplicate images in validation?", "At test time every product needs a prediction, duplicate or "
        "not. Validation must look like the real test."),
       ("Did the text cleaning help?", "No: 0.8276 on raw text against 0.8274 on clean text. We measured it instead "
        "of assuming it. TF-IDF already gives little weight to rare HTML tags.")])}
</div>""")

    pages.append(f"""
<div class="page">
  <p class="step">STEP 3.1 &nbsp; 15 OCT</p>
  <h2>Baselines</h2>
  <div class="main">Text is far stronger than images. A simple TF-IDF + SVM already reaches 0.83, the image model
  0.48.</div>
  <div class="cols">
    <div class="col">
      <div class="concept"><b>Concept: TF-IDF.</b> Each product becomes a long list of numbers, one per word or
      word pair. A word gets a high value if it appears in this product but is rare overall (<i>piscine</i>),
      and a low value if it appears everywhere (<i>pour</i>).</div>
      <div class="concept"><b>Concept: linear SVM.</b> For each category, the model learns one weight per word.
      A product's score for a category is the sum of its word values times the weights. The highest score wins.</div>
    </div>
    <div class="col">
      {img(figs, "baselines")}
      {img(figs, "seeds")}
      <p class="note">Binary TF-IDF (count a word once, however often it is repeated) looked +0.4 better on one
      split. Over 5 splits the gain is 0.33, smaller than the 0.71 the score moves between splits.</p>
    </div>
  </div>
  <div class="wide">{img(figs, "errors", max_height="62mm")}</div>
  {qa([("Is 0.83 better than the official benchmark (0.8113)?", "Not directly comparable: the benchmark is a CNN "
        "on titles only, scored on Rakuten's hidden test set. Our reference is our own baseline on our validation."),
       ("Where does the SVM fail?", "36% of errors have a margin under 0.20: the model hesitated between two "
        "categories. 7 of the 10 most confused pairs are book categories.")])}
</div>""")

    pages.append(f"""
<div class="page">
  <p class="step">STEP 3.2 &nbsp; 29 OCT</p>
  <h2>Optimisation of the text model</h2>
  <div class="main">Using all words instead of 50000 and tuning with cross-validation on train: 0.8313 to 0.8436.</div>
  <div class="cols">
    <div class="col">
      <div class="concept"><b>Concept: 3-fold cross-validation.</b> Train is cut into 3 parts. The model trains
      on 2 and is tested on the third, 3 times. The average is used to choose settings, so validation is never
      used for choices.<br><br>{folds()}<span class="note">blue: train, orange: test</span></div>
      <div class="concept"><b>Concept: C in the SVM.</b> C sets how strictly the SVM fits the training data.
      Small C gives a simpler model (strong regularisation), large C fits every training product, including noise.</div>
    </div>
    <div class="col">
      {img(figs, "features")}
      <p class="note">The 50000 limit kept only the most frequent words and dropped 89% of them, including rare
      but useful ones such as brand or model names.</p>
    </div>
  </div>
  <div class="cols">
    <div class="col">{img(figs, "grid")}</div>
    <div class="col">{img(figs, "models")}</div>
  </div>
  {qa([("Was validation used to choose settings?", "No. Feature limit, binary and C were chosen with 3-fold CV on "
        "train. Validation was scored once at the end."),
       ("Why keep binary TF-IDF now?", "With all features its gain (0.50 points) is larger than the CV spread "
        "(0.30). We set this rule before looking at the result."),
       ("Why LinearSVC and not Logistic Regression?", "2 points better in CV, and still trains in seconds.")])}
</div>""")

    pages.append(f"""
<div class="page">
  <p class="step">STEP 3.3 &nbsp; 5 NOV</p>
  <h2>Deep learning for text: CamemBERT</h2>
  <div class="main">CamemBERT reads words in context and reaches {CAMEMBERT_F1:.4f}, +3.4 points over the SVM.
  The largest gains are on the book categories.</div>
  <div class="cols">
    <div class="col">
      <div class="concept"><b>Concept: transformer and fine-tuning.</b> CamemBERT is a transformer pre-trained on
      a large amount of French text, so it already knows French. Each word is read together with the words around
      it, so <i>lot de 5 tomes</i> and <i>lot de jardin</i> are understood differently. Fine-tuning adds a small
      layer for our 27 categories and trains the whole model a little more on our products.</div>
      <table>
        <tr><th>Setting</th><th>Value</th></tr>
        <tr><td>Input</td><td>title + description, 128 tokens</td></tr>
        <tr><td>Epochs, batch size</td><td>2, 16</td></tr>
        <tr><td>Optimiser</td><td>AdamW, learning rate 0.00005</td></tr>
        <tr><td>Regularisation</td><td>weight decay 0.01, dropout 0.1</td></tr>
        <tr><td>Stability</td><td>warmup, linear decay, gradient clipping 1.0</td></tr>
      </table>
      {img(figs, "lr", "60%")}
    </div>
    <div class="col">
      {img(figs, "loss")}
      <p class="note">Validation loss keeps falling in both epochs (0.501, then 0.427): no overfitting. In epoch 2 train loss drops a little below it, as each product is seen a second time.</p>
      {img(figs, "camembert_gain", max_height="95mm")}
    </div>
  </div>
  {qa([("Why no cross-validation for CamemBERT?", "One training run takes about 2 hours on our laptop. We used the "
        "standard settings and tuned nothing, so validation only measures the result."),
       ("Why is it better than TF-IDF?", "TF-IDF loses word order and context. Book titles share the same words "
        "(tome, édition) but not the same context.")])}
</div>""")

    pages.append(f"""
<div class="page">
  <p class="step">STEP 3.3 &nbsp; 5 NOV</p>
  <h2>Image models</h2>
  <div class="main">Images alone are weak (0.48 frozen, 0.70 with ConvNeXt), but their errors differ from text.</div>
  <div class="cols">
    <div class="col">
      <div class="concept"><b>Concept: frozen vs fine-tuned network.</b> A pre-trained image network already
      recognises shapes and objects. <b>Frozen</b>: only the last layer is trained, fast but limited.
      <b>Fine-tuned</b>: deeper layers are also trained, so the network adapts to product photos.</div>
      {img(figs, "image_models")}
    </div>
    <div class="col">
      {img(figs, "image_overfit", "85%")}
      <p class="note">When the whole network is trained, train loss keeps falling but validation loss rises after
      epoch 1: the model starts to memorise the training images.</p>
    </div>
  </div>
  {qa([("Why was train loss higher than validation loss for the frozen model?", "Train loss is averaged over the "
        "epoch while the model is still learning, and validation loss is measured at the end. With one epoch this "
        "gap is expected."),
       ("Why use a model that only scores 0.48 to 0.70?", "Because it fails on different products than text. "
        "Combined, it fixes text errors, mainly on books.")])}
</div>""")

    pages.append(f"""
<div class="page">
  <p class="step">STEP 3.3 &nbsp; 5 NOV</p>
  <h2>Fusion: combining text and image</h2>
  <div class="main">A Random Forest that learns how to combine the models beats any single model:
  {FUSION_BEST_F1:.4f}.</div>
  {flow(["SVM<br>27 probs", "CamemBERT<br>27 probs", "Image<br>27 probs", "Random Forest<br>81 inputs", "Category"])}
  <div class="cols">
    <div class="col">
      <div class="concept"><b>Concept: stacking.</b> Each model gives 27 probabilities per product. A second model,
      here a Random Forest, takes these 81 numbers as input and learns when to trust which model, for example
      "for books, listen to the image more".</div>
      <div class="concept"><b>Concept: Random Forest and Gradient Boosting.</b> A Random Forest trains 300 decision
      trees on random samples and lets them vote (bagging). Gradient Boosting trains trees one after another,
      each fixing the previous errors (boosting).</div>
    </div>
    <div class="col">
      <div class="concept"><b>Concept: calibration.</b> An SVM gives scores, not probabilities. Calibration learns
      on the training data how to turn these scores into probabilities, so they can be combined with the other
      models.</div>
    </div>
  </div>
  <div class="wide">{img(figs, "fusion", max_height="80mm")}</div>
  {qa([("Is the fusion score optimistic?", "No. The Random Forest is scored with 5-fold cross-validation on the "
        "validation products: every product is predicted by a model that never saw it."),
       ("Why keep the SVM if CamemBERT is better?", "They make different errors: CamemBERT reads context, the SVM "
        "catches rare exact words. With the SVM: 0.8937, without: 0.8837.")])}
</div>""")

    pages.append(f"""
<div class="page">
  <p class="step">STEP 3.3 &nbsp; 5 NOV</p>
  <h2>Interpretability</h2>
  <div class="main">The text model mostly uses sensible product words, but it also learned seller names and
  writing style.</div>
  <div class="cols">
    <div class="col">
      <div class="concept"><b>Concept: reading the SVM weights.</b> A linear SVM has one weight per word and
      category. The highest weights show which words push a product toward a category.</div>
    </div>
    <div class="col">
      <ul>
        <li>Clear words: <i>piscine</i> for pool, <i>chien</i> for pets, <i>warhammer</i> for wargames.</li>
        <li>Seller names: <i>sweetheart93</i>, <i>germenly</i>. This works on this data but may not on new sellers.</li>
        <li>Writing style: new books use summary words (<i>cet ouvrage</i>), used books often have English or
        German titles.</li>
        <li>Short titles with no book word (<i>la promenade au parc</i>) give the model nothing to decide with.</li>
      </ul>
    </div>
  </div>
  <div class="wide">{img(figs, "words")}</div>
  {qa([("Can the model be trusted on new sellers?", "Partly. Part of its score comes from seller names. It should be "
        "monitored on new sellers and retrained regularly.")])}
</div>""")

    pages.append(f"""
<div class="page">
  <p class="step">CONCLUSIONS</p>
  <h2>What we learned</h2>
  <div class="cols">
    <div class="col">
      <h3>Scientific</h3>
      <ul>
        <li>Text carries most of the signal. Images alone are weak but help where text is weak.</li>
        <li>Context matters: CamemBERT beats TF-IDF by 3.4 points, mostly on books.</li>
        <li>A learned combination beats a fixed weight: {FUSION_BEST_F1:.4f} against 0.8501.</li>
        <li>Several "obvious" ideas did not help when measured: text cleaning, and binary TF-IDF with 50000 features.</li>
      </ul>
      <h3>Business</h3>
      <ul>
        <li>About 9 in 10 products are put in the right category automatically.</li>
        <li>Low-confidence products, mainly books, should go to a human or ask the seller to confirm.</li>
        <li>Requiring a description would help: short titles are the main source of errors.</li>
      </ul>
    </div>
    <div class="col">
      <h3>Limits</h3>
      <ul>
        <li>Scores are on our validation split, not on Rakuten's hidden test set.</li>
        <li>CamemBERT settings were not tuned, because each run takes 2 hours.</li>
        <li>Category names are our guesses; Rakuten does not publish them.</li>
        <li>Part of the text model relies on seller names.</li>
      </ul>
      <h3>Next steps</h3>
      <ul>
        <li>Fuse the ConvNeXt image model (0.70) instead of ResNet18.</li>
        <li>Train CamemBERT for a third epoch and on longer texts.</li>
        <li>Streamlit app with confidence levels for each prediction.</li>
      </ul>
    </div>
  </div>
</div>""")
    return pages


def main():
    figs = make_figures()
    html = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{''.join(build_pages(figs))}</body></html>"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "project_story.html").write_text(html, encoding="utf-8")
    HTML(string=html, base_url=str(OUT_DIR)).write_pdf(OUT_DIR / "project_story.pdf")
    print(f"Saved {OUT_DIR / 'project_story.pdf'}")


if __name__ == "__main__":
    main()
