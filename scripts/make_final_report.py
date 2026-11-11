"""Build the final report: draw its figures from the saved results, then write the PDF.

Run from the project root, after the notebooks have been run:

    python scripts/make_final_report.py

Input:  reports/final/final_report.html (text of the report)
Output: reports/final/figures/*.png and reports/final/final_report.pdf
"""

import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from sklearn.metrics import f1_score
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
    return path


def image_f1(name):
    return json.load(open(OUTPUTS / f"{name}_validation_metrics.json"))["weighted_f1"]


def benchmark_figure(fusion):
    resnet_baseline = pd.read_csv(PROJECT_ROOT / "data/processed/image_validation_predictions.csv")
    rows = [
        ("ResNet18, frozen (03)", f1_score(resnet_baseline["true_prdtypecode"],
                                           resnet_baseline["predicted_prdtypecode"], average="weighted"), IMAGE_COLOR),
        ("ResNet18, full fine-tuning (03_3)", image_f1("03_3_resnet18_full"), IMAGE_COLOR),
        ("ResNet18, augmentation (03_4)", image_f1("03_4_resnet18_regularized"), IMAGE_COLOR),
        ("ViT-B/16 (03_6)", image_f1("03_6_vit_b16"), IMAGE_COLOR),
        ("ResNet50 (03_5)", image_f1("03_5_resnet50"), IMAGE_COLOR),
        ("ConvNeXt-Tiny (03_5)", image_f1("03_5_convnext_tiny"), IMAGE_COLOR),
        ("ConvNeXt-Tiny, optimised (03_7)", image_f1("03_7_convnext_tiny_optimized"), IMAGE_COLOR),
    ]
    f1 = fusion.set_index("Model")["Weighted F1"]
    rows += [
        ("TF-IDF + SVM", f1["SVM only (TF-IDF)"], TEXT_COLOR),
        ("CamemBERT", f1["CamemBERT only"], TEXT_COLOR),
        ("Weighted soft voting (SVM + image)", f1["Weighted soft voting (SVM + image)"], FUSION_COLOR),
        ("Soft voting, equal weights (all 3)", f1["Soft voting, equal weights (CamemBERT + SVM + image)"], FUSION_COLOR),
        ("Random Forest (all 3)", f1["Random Forest (CamemBERT + SVM + image)"], FUSION_COLOR),
    ]
    labels, values, colors = zip(*rows[::-1])

    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars = ax.barh(labels, values, color=colors, height=0.65)
    ax.bar_label(bars, labels=[f"{v:.4f}" for v in values], padding=3, fontsize=8, color=INK)
    ax.set_xlim(0.4, 0.95)
    handles = [Patch(color=color, label=name)
               for color, name in [(IMAGE_COLOR, "Image"), (TEXT_COLOR, "Text"), (FUSION_COLOR, "Fusion")]]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="upper right")
    style(ax, "Weighted F1 on the 16984 validation products", "Weighted F1")
    return save(fig, "fig1_benchmark_progression")


def class_figure():
    from src.categories import CATEGORY_NAMES

    counts = pd.read_csv(PROJECT_ROOT / "data/processed/text_train_processed.csv",
                         usecols=["prdtypecode"])["prdtypecode"].value_counts().sort_values()
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


def gain_figure():
    per_class = pd.read_csv(OUTPUTS / "04_fusion_per_class.csv").sort_values("Gain")
    labels = [f"{n} ({c})" for n, c in zip(per_class["Category"], per_class["Code"])]
    gain = per_class["Gain"] * 100

    fig, ax = plt.subplots(figsize=(9, 6.5))
    bars = ax.barh(labels, gain, color=[RED if g < 0 else IMAGE_COLOR for g in gain], height=0.7)
    ax.bar_label(bars, labels=[f"{g:+.1f}" for g in gain], padding=3, fontsize=7.5, color=INK)
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.tick_params(axis="y", labelsize=8)
    style(ax, "F1 gain of the Random Forest fusion over CamemBERT alone, per category",
          "Gain in F1 points")
    return save(fig, "fig3_multimodal_synergy_delta")


def notebook_figure(notebook, cell, name):
    """Copy the first plot of a notebook cell, so the report shows the same figure."""
    nb = json.load(open(PROJECT_ROOT / "notebooks" / notebook))
    images = [o["data"]["image/png"] for o in nb["cells"][cell].get("outputs", []) if "image/png" in o.get("data", {})]
    path = FIG_DIR / f"{name}.png"
    path.write_bytes(base64.b64decode(images[0]))
    return path


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fusion = pd.read_csv(OUTPUTS / "04_fusion_results.csv")

    benchmark_figure(fusion)
    class_figure()
    gain_figure()
    notebook_figure("05_interpretability.ipynb", 7, "fig4_book_words")
    notebook_figure("05_interpretability.ipynb", 17, "fig5_fusion_importance")
    notebook_figure("05_interpretability.ipynb", 18, "fig6_image_gain")

    HTML(filename=str(REPORT_DIR / "final_report.html")).write_pdf(REPORT_DIR / "final_report.pdf")
    print(f"Saved {REPORT_DIR / 'final_report.pdf'}")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(PROJECT_ROOT))
    main()
