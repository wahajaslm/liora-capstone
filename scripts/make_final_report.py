"""Build the final report: draw its figures from the saved results, then write the PDF.

Run from the project root, after the notebooks have been run:

    python scripts/make_final_report.py

Input:  reports/final/final_report.html (text of the report)
Output: reports/final/figures/*.png and reports/final/final_report.pdf
"""

import base64
import json
import os
from pathlib import Path

# Ensure matplotlib uses a writable temporary config directory in sandbox
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib_config"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.metrics import confusion_matrix, f1_score
from weasyprint import HTML

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports/final"
FIG_DIR = REPORT_DIR / "figures"
OUTPUTS = PROJECT_ROOT / "models/outputs"

INK, MUTED, LINE = "#1f2933", "#6b7280", "#d1d5db"
IMAGE_COLOR, TEXT_COLOR, FUSION_COLOR, RED = "#0d9488", "#4338ca", "#d97706", "#c0504d"


def style(ax, title, xlabel):
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(axis="x", color=LINE, linewidth=0.7)
    ax.set_axisbelow(True)


def save(fig, name):
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    return path


def image_f1(name):
    return json.load(open(OUTPUTS / f"{name}_validation_metrics.json"))["weighted_f1"]


def majority_f1():
    """Weighted F1 when every validation product is given the largest category."""
    data = pd.read_csv(PROJECT_ROOT / "data/processed/text_train_processed.csv", usecols=["prdtypecode", "split"])
    y = data.loc[data["split"] == "validation", "prdtypecode"]
    return f1_score(y, [y.mode()[0]] * len(y), average="weighted")


def benchmark_figure(fusion):
    res_base = pd.read_csv(OUTPUTS / "03_1_resnet18_baseline_validation_predictions.csv")
    f1_res_base = f1_score(res_base["true_prdtypecode"], res_base["predicted_prdtypecode"], average="weighted")

    rows = [
        ("Zero-Rule Majority Baseline", majority_f1(), "#94a3b8"),
        ("ResNet18, frozen (03)", f1_res_base, IMAGE_COLOR),
        ("ResNet18, full fine-tuning (03_3)", image_f1("03_3_resnet18_full"), IMAGE_COLOR),
        ("ResNet18, augmentation (03_4)", image_f1("03_4_resnet18_regularized"), IMAGE_COLOR),
        ("ViT-B/16 (03_6)", image_f1("03_6_vit_b16"), IMAGE_COLOR),
        ("ResNet50 (03_5)", image_f1("03_5_resnet50"), IMAGE_COLOR),
        ("ConvNeXt-Tiny (03_5)", image_f1("03_5_convnext_tiny"), IMAGE_COLOR),
        ("ConvNeXt-Tiny, optimised (03_7)", image_f1("03_7_convnext_tiny_optimized"), IMAGE_COLOR),
    ]
    f1 = fusion.set_index("Model")["Weighted F1"]
    rows += [
        ("TF-IDF + SVM (03_2)", f1["SVM only (TF-IDF)"], TEXT_COLOR),
        ("CamemBERT (03_3)", f1["CamemBERT only"], TEXT_COLOR),
        ("Weighted soft voting (SVM + image) (04)", f1["Weighted soft voting (SVM + image)"], FUSION_COLOR),
        ("Weighted soft voting (CamemBERT + image) (04)", f1["Weighted soft voting (CamemBERT + image)"], FUSION_COLOR),
        ("Weighted soft voting (CamemBERT + SVM + image) (04)", f1["Weighted soft voting (CamemBERT + SVM + image)"], FUSION_COLOR),
        ("Soft voting, equal weights (CamemBERT + SVM + image) (04)", f1["Soft voting, equal weights (CamemBERT + SVM + image)"], FUSION_COLOR),
    ]
    labels, values, colors = zip(*rows[::-1])

    fig, ax = plt.subplots(figsize=(9, 5.8))
    bars = ax.barh(labels, values, color=colors, height=0.65)
    ax.bar_label(bars, labels=[f"{v:.4f}" for v in values], padding=4, fontsize=8, color=INK)
    ax.set_xlim(0.0, 1.02)
    handles = [
        Patch(color=color, label=name)
        for color, name in [
            (IMAGE_COLOR, "Image"),
            (TEXT_COLOR, "Text"),
            (FUSION_COLOR, "Fusion"),
            ("#94a3b8", "Baseline"),
        ]
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.85),
        frameon=True,
        facecolor="#ffffff",
        edgecolor=LINE,
        fontsize=8.5,
    )
    style(ax, "Weighted F1 on the 16984 validation products", "Weighted F1")
    return save(fig, "fig1_benchmark_progression")


def class_figure():
    from src.categories import CATEGORY_NAMES

    raw_y_path = PROJECT_ROOT / "data/raw/Y_train_CVw08PX.csv"
    if raw_y_path.exists():
        raw_y = pd.read_csv(raw_y_path, index_col=0)
        counts = raw_y["prdtypecode"].value_counts().sort_values()
    else:
        counts = pd.read_csv(
            PROJECT_ROOT / "data/processed/text_train_processed.csv",
            usecols=["prdtypecode"],
        )["prdtypecode"].value_counts().sort_values()

    labels = [f"{CATEGORY_NAMES[c]} ({c})" for c in counts.index]
    share = counts / counts.sum() * 100

    fig, ax = plt.subplots(figsize=(9, 6.5))
    bars = ax.barh(labels, counts.values, color=[RED if n < 1000 else TEXT_COLOR for n in counts.values], height=0.7)
    ax.bar_label(bars, labels=[f"{n} ({s:.1f}%)" for n, s in zip(counts.values, share.values)],
                 padding=3, fontsize=7.5, color=INK)
    ax.set_xlim(0, counts.max() * 1.18)
    ax.tick_params(axis="y", labelsize=8)
    style(ax, f"Products per category (largest / smallest = {counts.max() / counts.min():.1f})",
          "Number of products (red: fewer than 1000)")
    return save(fig, "fig2_class_distribution")


def validation_probabilities():
    """True labels and the three models' probabilities for the 16984 validation products."""
    data = pd.read_csv(PROJECT_ROOT / "data/processed/text_train_processed.csv", usecols=["row_id", "prdtypecode", "split"])
    val = data[data["split"] == "validation"]
    classes = np.sort(data["prdtypecode"].unique())
    columns = [str(c) for c in classes]

    def load(path, prefix=""):
        frame = pd.read_csv(path)
        frame.columns = [c.strip() for c in frame.columns]
        return val[["row_id"]].merge(frame, on="row_id")[[prefix + c for c in columns]].to_numpy()

    P_cam = load(PROJECT_ROOT / "data/processed/camembert_validation_probabilities.csv")
    P_svm = load(PROJECT_ROOT / "data/processed/text_validation_probabilities.csv")
    P_img = load(OUTPUTS / "03_7_convnext_tiny_optimized_validation_predictions.csv", "prob_")
    return val["prdtypecode"].to_numpy(), classes, P_cam, P_svm, P_img


def per_category_bars(gain, title, xlabel, name):
    from src.categories import CATEGORY_NAMES

    gain = gain.sort_values()
    fig, ax = plt.subplots(figsize=(9, 6.5))
    bars = ax.barh([f"{CATEGORY_NAMES[c]} ({c})" for c in gain.index], gain.values,
                   color=[RED if g < 0 else IMAGE_COLOR for g in gain.values], height=0.7)
    ax.bar_label(bars, labels=[f"{g:+.1f}" for g in gain.values], padding=3, fontsize=7.5, color=INK)
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.tick_params(axis="y", labelsize=8)
    style(ax, title, xlabel)
    return save(fig, name)


def fusion_figures():
    """Per-category gain over CamemBERT, image contribution and confusion matrix of equal soft voting."""
    from src.categories import CATEGORY_NAMES

    y, classes, P_cam, P_svm, P_img = validation_probabilities()
    vote = classes[(P_cam + P_svm + P_img).argmax(axis=1)]
    vote_without_image = classes[(P_cam + P_svm).argmax(axis=1)]
    camembert = classes[P_cam.argmax(axis=1)]
    per_class = lambda pred: pd.Series(f1_score(y, pred, average=None, labels=classes) * 100, index=classes)

    per_category_bars(per_class(vote) - per_class(camembert),
                      "F1 gain of equal soft voting over CamemBERT alone, per category",
                      "Gain in F1 points", "fig3_multimodal_synergy_delta")
    per_category_bars(per_class(vote) - per_class(vote_without_image),
                      "What the image adds to equal soft voting, per category",
                      "F1 with image minus F1 without image (points)", "fig6_image_gain")

    cm = confusion_matrix(y, vote, labels=classes)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots(figsize=(10, 8.5), constrained_layout=True)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Recall (Normalized)", rotation=-90, va="bottom", fontsize=9)
    labels = [f"{CATEGORY_NAMES[c]} ({c})" for c in classes]
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7.5)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Predicted Category", fontsize=10, fontweight="bold")
    ax.set_ylabel("True Category", fontsize=10, fontweight="bold")
    ax.set_title("Normalized Confusion Matrix — Equal Soft Voting", fontsize=11, fontweight="bold", pad=12)
    return save(fig, "fig10_confusion_matrix")


def notebook_figure(notebook_relpath, marker, name):
    """Copy the plot of the notebook cell whose code contains `marker` to FIG_DIR.

    Finding the cell by its code keeps this working when cells are added or removed.
    If the notebook has no saved plot (e.g. not run), the existing figure file is kept.
    """
    nb = json.load(open(PROJECT_ROOT / notebook_relpath))
    for cell in nb["cells"]:
        if cell["cell_type"] != "code" or marker not in "".join(cell["source"]):
            continue
        for output in cell.get("outputs", []):
            if "image/png" in output.get("data", {}):
                path = FIG_DIR / f"{name}.png"
                path.write_bytes(base64.b64decode(output["data"]["image/png"]))
                print(f"Saved {path}")
                return path
    print(f"Kept existing {name}.png: no saved plot for '{marker}' in {notebook_relpath}")
    return None


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fusion = pd.read_csv(OUTPUTS / "04_fusion_results.csv")

    benchmark_figure(fusion)
    class_figure()
    fusion_figures()

    # Interpretability figures from notebook 05
    interpretability = "notebooks/05_interpretability.ipynb"
    notebook_figure(interpretability, "book_codes = [10, 2705, 2403, 2280]", "fig4_book_words")
    notebook_figure(interpretability, "def word_importance", "fig11_camembert_words")

    # Grad-CAM figures from notebook 03_8 (kept as files if the notebook has no saved plots)
    gradcam = "notebooks/03_image_fineTuning/03_8_image_convnext_gradcam.ipynb"
    notebook_figure(gradcam, "SUCCESS_ROW_IDS", "fig7_gradcam_correct")
    notebook_figure(gradcam, "ERROR_ROW_IDS", "fig8_gradcam_errors")
    notebook_figure(gradcam, "CROP_ROW_ID =", "fig9_gradcam_cropping")

    print("Compiling PDF with WeasyPrint...")
    HTML(filename=str(REPORT_DIR / "final_report.html")).write_pdf(REPORT_DIR / "final_report.pdf")
    print(f"Successfully generated {REPORT_DIR / 'final_report.pdf'}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    main()
