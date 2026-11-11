"""Draw the project timeline figure for the reports.

Run from the project root:

    python scripts/make_timeline.py

Numbers are written in RESULTS below. Update them if a model is re-run.
"""

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "reports/figures"

# Weighted F1 on the 16984 validation products
RESULTS = {
    "Majority class": 0.026,
    "SVM baseline": 0.8274,
    "SVM binary": 0.8313,
    "SVM optimised": 0.8436,
    "CamemBERT": 0.8765,
    "Fusion (RF)": 0.8988,
}
# F1 on Used Books (10), the hardest category
USED_BOOKS = {"SVM": 0.566, "RF SVM + image": 0.663, "CamemBERT": 0.689}

INK = "#1f2933"
MUTED = "#6b7280"
LINE = "#d1d5db"
ACCENT = "#2b5c8f"
CARD = "#f8fafc"

STAGES = [
    {
        "title": "1. Data audit",
        "problem": "We did not know the data quality or how balanced the categories were.",
        "when": "Step 1, 1 Oct",
        "did": "Explored 84916 products in 27 categories, with title, description and image.",
        "result": "35% have no description. 18% of descriptions contain HTML. "
                  "Unbalanced: largest category 12%, smallest 0.9%.",
        "qa": [
            ("Why weighted F1?",
             "It is the challenge metric. Macro F1 is also reported so small categories are not hidden."),
        ],
    },
    {
        "title": "2. Preprocessing",
        "problem": "HTML noise, missing descriptions, and duplicate images that could leak between train and validation.",
        "when": "Step 2, 8 Oct",
        "did": "Removed HTML, lowercased, joined title and description. "
               "One stratified 80/20 split saved to a file. Duplicate images found by file hash.",
        "result": "67932 train, 16984 validation. 5620 duplicate or conflicting images removed from train only.",
        "qa": [
            ("Why keep duplicates in validation?",
             "At test time every product needs a prediction. Validation stays complete and the same for every model."),
        ],
    },
    {
        "title": "3. Baselines",
        "problem": "No reference score yet. Which signal is stronger, text or image?",
        "when": "Step 3.1, 15 Oct",
        "did": "Majority class, TF-IDF with a linear SVM, and a frozen ResNet18 on images.",
        "result": "Weighted F1: majority 0.026, SVM 0.827, image 0.482.",
        "qa": [
            ("Did text cleaning help?",
             "No. 0.8276 on raw text, 0.8274 on clean text. Tested, not assumed."),
            ("Is binary TF-IDF better (+0.4)?",
             "Not clearly. Over 5 split seeds the gain is 0.33, the noise is 0.71."),
        ],
    },
    {
        "title": "4. Optimisation",
        "problem": "Baseline settings were picked by hand, and binary TF-IDF was picked by looking at validation.",
        "when": "Step 3.2, 29 Oct",
        "did": "Grid search with 3-fold CV on train (binary, C). All 460326 features instead of 50000. "
               "Compared with Naive Bayes and Logistic Regression.",
        "result": "SVM: 0.8313 to 0.8436.",
        "qa": [
            ("Was validation used to choose settings?",
             "No. Settings chosen with CV on train. Validation scored once."),
            ("Why keep LinearSVC?",
             "Best CV score: 0.828, against 0.808 for Logistic Regression and 0.760 for Naive Bayes."),
        ],
    },
    {
        "title": "5. CamemBERT",
        "problem": "Book categories share the same words. TF-IDF cannot separate them.",
        "when": "Step 3.3, 5 Nov",
        "did": "Fine-tuned a French transformer: 2 epochs, AdamW, warmup and linear decay, "
               "weight decay and dropout.",
        "result": "Weighted F1 0.8765 (+3.4 over SVM). Book categories +10 to +12 points.",
        "qa": [
            ("Why better than TF-IDF?",
             "It reads words in context and in order. Book titles share words but not context."),
            ("Why no cross-validation?",
             "One run takes 2 hours on a laptop. Standard settings, nothing tuned. Validation only measures."),
        ],
    },
    {
        "title": "6. Fusion",
        "problem": "Text and image are good at different categories. One fixed weight cannot use that.",
        "when": "Step 3.3, 5 Nov",
        "did": "Combined text and image probabilities: a fixed weight, then Random Forest "
               "and Gradient Boosting trained on the probabilities.",
        "result": "Fixed weight 0.8640. Random Forest on CamemBERT + SVM + ConvNeXt image 0.8988.",
        "qa": [
            ("Is the fusion score optimistic?",
             "No. 5-fold CV on validation. The weight is chosen inside each fold."),
            ("Why add an image model at 0.48?",
             "Its errors differ from text. It helps most on books."),
        ],
    },
    {
        "title": "7. Interpretability",
        "problem": "A good score does not prove the model learned the right signals.",
        "when": "Step 3.3, 5 Nov",
        "did": "Read the SVM word weights for each category. Explained one wrong prediction word by word.",
        "result": "Clear words (piscine, chien, warhammer), but also seller names (sweetheart93) "
                  "and writing style.",
        "qa": [
            ("Will it work for new sellers?",
             "Partly. Seller names are a risk, so the model needs monitoring and retraining."),
        ],
    },
]


def wrapped(text, width):
    return "\n".join(textwrap.wrap(text, width))


def draw_card(fig, x, y, w, h, stage):
    ax = fig.add_axes([x, y, w, h])
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.02",
                                facecolor=CARD, edgecolor=LINE, linewidth=1, transform=ax.transAxes))
    width = 34
    k = 0.5 / h  # keep line spacing constant whatever the card height
    cursor = 1 - 0.035 * k

    def block(label, text, color=INK, style="normal", weight="normal", gap=0.012):
        nonlocal cursor
        if label:
            ax.text(0.05, cursor, label, fontsize=8.5, color=MUTED, fontweight="bold", va="top",
                    transform=ax.transAxes)
            cursor -= 0.032 * k
        body = wrapped(text, width)
        ax.text(0.05, cursor, body, fontsize=9.6, color=color, va="top", style=style,
                fontweight=weight, linespacing=1.3, transform=ax.transAxes)
        cursor -= (0.0295 * (body.count("\n") + 1) + gap) * k

    block("PROBLEM", stage["problem"], gap=0.018)
    block("WHAT WE DID", stage["did"], gap=0.018)
    block("RESULT", stage["result"], color=ACCENT, weight="bold", gap=0.02)
    ax.plot([0.05, 0.95], [cursor + 0.008 * k, cursor + 0.008 * k], color=LINE, linewidth=1, transform=ax.transAxes)
    cursor -= 0.015 * k
    for question, answer in stage["qa"]:
        block(None, f"Q: {question}", color=INK, style="italic", gap=0.004)
        block(None, f"A: {answer}", color=MUTED, gap=0.02)


def main():
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(24, 14), facecolor="white")

    fig.text(0.02, 0.965, "Rakuten product classification: how the model was built",
             fontsize=18, fontweight="bold", color=INK)
    fig.text(0.02, 0.94, "Each step: what we did, the result, and the question an examiner would ask. "
             "All scores are weighted F1 on the same 16984 validation products.",
             fontsize=10.5, color=MUTED)

    n = len(STAGES)
    left, right = 0.02, 0.98
    gap = 0.008
    card_w = (right - left - gap * (n - 1)) / n
    timeline_y = 0.885

    timeline = fig.add_axes([0, 0, 1, 1], facecolor="none")
    timeline.set_axis_off()
    timeline.set_xlim(0, 1)
    timeline.set_ylim(0, 1)
    centers = [left + card_w / 2 + i * (card_w + gap) for i in range(n)]
    timeline.plot([centers[0], centers[-1]], [timeline_y, timeline_y], color=LINE, linewidth=2, zorder=1)
    for cx, stage in zip(centers, STAGES):
        timeline.scatter([cx], [timeline_y], s=110, color=ACCENT, zorder=2, edgecolor="white", linewidth=2)
        timeline.text(cx, timeline_y + 0.022, stage["title"], ha="center", fontsize=11.5,
                      fontweight="bold", color=INK)
        timeline.text(cx, timeline_y - 0.03, stage["when"], ha="center", fontsize=9, color=MUTED)

    for i, stage in enumerate(STAGES):
        draw_card(fig, left + i * (card_w + gap), 0.425, card_w, 0.42, stage)

    # Progress chart
    ax = fig.add_axes([0.05, 0.06, 0.55, 0.28])
    names = list(RESULTS)[1:]
    values = [RESULTS[k] for k in names]
    ax.plot(range(len(names)), values, color=ACCENT, linewidth=2.2, marker="o", markersize=8,
            markeredgecolor="white", markeredgewidth=1.5)
    for i, v in enumerate(values):
        ax.annotate(f"{v:.4f}", (i, v), textcoords="offset points", xytext=(0, 10), ha="center",
                    fontsize=9.5, color=INK)
    ax.set_xticks(range(len(names)), names, fontsize=9.5, color=INK)
    ax.set_ylim(0.81, 0.91)
    ax.set_ylabel("Weighted F1", fontsize=9.5, color=MUTED)
    ax.set_title("Progress of the best model (majority class baseline: 0.026, off the chart)",
                 loc="left", fontsize=11, fontweight="bold", color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(LINE)
    ax.tick_params(colors=MUTED)
    ax.grid(axis="y", color=LINE, linewidth=0.8, alpha=0.7)

    # Hardest category chart
    ax2 = fig.add_axes([0.68, 0.06, 0.29, 0.28])
    labels = list(USED_BOOKS)
    vals = [USED_BOOKS[k] for k in labels]
    bars = ax2.barh(labels[::-1], vals[::-1], color=ACCENT, height=0.55)
    ax2.bar_label(bars, labels=[f"{v:.2f}" for v in vals[::-1]], padding=4, fontsize=9.5, color=INK)
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("F1", fontsize=9.5, color=MUTED)
    ax2.set_title("Hardest category: Used Books (10)", loc="left", fontsize=11, fontweight="bold", color=INK)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.spines[["left", "bottom"]].set_color(LINE)
    ax2.tick_params(colors=MUTED, labelsize=9.5)
    ax2.grid(axis="x", color=LINE, linewidth=0.8, alpha=0.7)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "project_timeline.png", dpi=170)
    fig.savefig(OUTPUT_DIR / "project_timeline.pdf")
    print(f"Saved {OUTPUT_DIR / 'project_timeline.png'}")


if __name__ == "__main__":
    main()
