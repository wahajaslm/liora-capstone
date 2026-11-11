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


def benchmark_figure(fusion):
    res_base = pd.read_csv(OUTPUTS / "03_1_resnet18_baseline_validation_predictions.csv")
    f1_res_base = f1_score(res_base["true_prdtypecode"], res_base["predicted_prdtypecode"], average="weighted")

    rows = [
        ("Zero-Rule Majority Baseline (03)", 0.0258, "#94a3b8"),
        ("ResNet18, frozen (03)", f1_res_base, IMAGE_COLOR),
        ("ResNet18, layer4 fine-tuning (03_2)", image_f1("03_2_resnet18_finetuned"), IMAGE_COLOR),
        ("ResNet18, augmentation (03_4)", image_f1("03_4_resnet18_regularized"), IMAGE_COLOR),
        ("ResNet18, full fine-tuning (03_3)", image_f1("03_3_resnet18_full"), IMAGE_COLOR),
        ("ViT-B/16 (03_6)", image_f1("03_6_vit_b16"), IMAGE_COLOR),
        ("ResNet50 (03_5)", image_f1("03_5_resnet50"), IMAGE_COLOR),
        ("ConvNeXt-Tiny (03_5)", image_f1("03_5_convnext_tiny"), IMAGE_COLOR),
        ("ConvNeXt-Tiny, optimised (03_7)", image_f1("03_7_convnext_tiny_optimized"), IMAGE_COLOR),
    ]
    f1 = fusion.set_index("Model")["Weighted F1"]
    rows += [
        ("Calibrated LinearSVC (03_2 / 04)", f1["SVM only (TF-IDF)"], TEXT_COLOR),
        ("Weighted soft voting (SVM + image) (04)", f1["Weighted soft voting (SVM + image)"], FUSION_COLOR),
        ("CamemBERT (03_3)", f1["CamemBERT only"], TEXT_COLOR),
        ("Weighted soft voting (CamemBERT + image) (04)", f1["Weighted soft voting (CamemBERT + image)"], FUSION_COLOR),
        ("Weighted soft voting (all 3) (04)", f1["Weighted soft voting (CamemBERT + SVM + image)"], FUSION_COLOR),
        ("Soft voting, equal weights (all 3) (04)", f1["Soft voting, equal weights (CamemBERT + SVM + image)"], FUSION_COLOR),
        ("Random Forest (all 3) (04)", f1["Random Forest (CamemBERT + SVM + image)"], FUSION_COLOR),
    ]
    labels, values, colors = zip(*rows[::-1])

    fig, ax = plt.subplots(figsize=(9, 6.4))
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


def confusion_matrix_figure():
    from src.categories import CATEGORY_NAMES

    processed_path = PROJECT_ROOT / "data/processed/text_train_processed.csv"
    if not processed_path.exists():
        print("Skipping CM figure (text_train_processed.csv not found)")
        return None

    data = pd.read_csv(processed_path)
    val = data[data["split"] == "validation"]
    classes = np.sort(data["prdtypecode"].unique())
    classes_str = [str(c) for c in classes]

    cam_df = pd.read_csv(PROJECT_ROOT / "data/processed/camembert_validation_probabilities.csv")
    svm_df = pd.read_csv(PROJECT_ROOT / "data/processed/text_validation_probabilities.csv")
    img_df = pd.read_csv(OUTPUTS / "03_7_convnext_tiny_optimized_validation_predictions.csv")

    P_cam = val[["row_id"]].merge(cam_df, on="row_id")[classes_str].to_numpy()
    P_svm = val[["row_id"]].merge(svm_df, on="row_id")[classes_str].to_numpy()
    P_img = val[["row_id"]].merge(img_df, on="row_id")[[f"prob_{c}" for c in classes]].to_numpy()
    y_true = val["prdtypecode"].to_numpy()

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    X_all = np.hstack([P_cam, P_svm, P_img])
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = np.zeros_like(y_true)

    for train_idx, test_idx in folds.split(X_all, y_true):
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=42)
        rf.fit(X_all[train_idx], y_true[train_idx])
        y_pred[test_idx] = rf.predict(X_all[test_idx])

    cm = confusion_matrix(y_true, y_pred, labels=classes)
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
    ax.set_title("Normalized Confusion Matrix — Random Forest Fusion (5-Fold CV OOF)", fontsize=11, fontweight="bold", pad=12)

    return save(fig, "fig10_confusion_matrix")


def notebook_figure(notebook_relpath, cell, output_idx, name):
    """Copy a plot of a notebook cell to FIG_DIR."""
    nb_path = PROJECT_ROOT / notebook_relpath
    if not nb_path.exists():
        print(f"Warning: {nb_path} does not exist.")
        return None
    nb = json.load(open(nb_path))
    outputs = nb["cells"][cell].get("outputs", [])
    if output_idx < len(outputs) and "image/png" in outputs[output_idx].get("data", {}):
        image_data = outputs[output_idx]["data"]["image/png"]
        path = FIG_DIR / f"{name}.png"
        path.write_bytes(base64.b64decode(image_data))
        print(f"Saved {path}")
        return path
    else:
        # Fallback to finding first image in cell
        for o in outputs:
            if "image/png" in o.get("data", {}):
                path = FIG_DIR / f"{name}.png"
                path.write_bytes(base64.b64decode(o["data"]["image/png"]))
                print(f"Saved {path}")
                return path
    print(f"Warning: No image found in {notebook_relpath} cell {cell}")
    return None


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fusion = pd.read_csv(OUTPUTS / "04_fusion_results.csv")

    benchmark_figure(fusion)
    class_figure()
    gain_figure()

    # Extract Interpretability figures from notebook 05
    notebook_figure("notebooks/05_interpretability.ipynb", 7, 0, "fig4_book_words")
    notebook_figure("notebooks/05_interpretability.ipynb", 17, 0, "fig5_fusion_importance")
    notebook_figure("notebooks/05_interpretability.ipynb", 18, 1, "fig6_image_gain")

    # Extract Grad-CAM figures from notebook 03_8
    notebook_figure("notebooks/03_image_fineTuning/03_8_image_convnext_gradcam.ipynb", 10, 0, "fig7_gradcam_correct")
    notebook_figure("notebooks/03_image_fineTuning/03_8_image_convnext_gradcam.ipynb", 13, 0, "fig8_gradcam_errors")
    notebook_figure("notebooks/03_image_fineTuning/03_8_image_convnext_gradcam.ipynb", 16, 0, "fig9_gradcam_cropping")

    # Generate Confusion Matrix figure if not already present
    if not (FIG_DIR / "fig10_confusion_matrix.png").exists():
        confusion_matrix_figure()

    print("Compiling PDF with WeasyPrint...")
    HTML(filename=str(REPORT_DIR / "final_report.html")).write_pdf(REPORT_DIR / "final_report.pdf")
    print(f"Successfully generated {REPORT_DIR / 'final_report.pdf'}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    main()
