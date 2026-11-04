"""Draw the project timeline with one lane each for project, text, image and fusion.

Run from the project root:

    python scripts/make_timeline_lanes.py

Dates come from the git history and the mentor's schedule.
"""

import textwrap
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "reports/figures"

INK = "#1f2933"
MUTED = "#6b7280"
FAINT = "#9ca3af"
LINE = "#d1d5db"
BAND = "#f3f4f6"
DATE = "#d0573a"

START, END = date(2026, 9, 21), date(2026, 12, 3)
TODAY = date(2026, 11, 4)

PHASES = [
    ("STEP 1", "data audit", date(2026, 9, 24), date(2026, 10, 1)),
    ("STEP 2", "preprocessing", date(2026, 10, 1), date(2026, 10, 8)),
    ("STEP 3.1", "baselines", date(2026, 10, 8), date(2026, 10, 15)),
    ("STEP 3.2", "optimisation", date(2026, 10, 15), date(2026, 10, 29)),
    ("STEP 3.3", "advanced models", date(2026, 10, 29), date(2026, 11, 5)),
    ("STEP 4", "final report", date(2026, 11, 5), date(2026, 11, 13)),
    ("STEP 5", "app and defense", date(2026, 11, 13), date(2026, 11, 24)),
]

LANES = {
    "PROJECT": [
        (date(2026, 9, 24), "Framing", "First meeting with the mentor. Scope and deadlines set."),
        (date(2026, 10, 5), "Data audit", "84916 products, 27 categories. 35% have no description. "
                                          "Categories range from 12% to 0.9%."),
        (date(2026, 10, 10), "Shared split", "One stratified 80/20 split saved to a file, "
                                             "so every model is scored on the same products."),
        (date(2026, 10, 23), "Mentor feedback", "8 points: same validation set for all models, "
                                                "check seed noise, per-category F1, fusion plan."),
        (date(2026, 11, 5), "Rendering 2", "Modelling report with interpretability and conclusions."),
        (date(2026, 11, 13), "Final report", "Reports 1 and 2 combined, code on GitHub.", "future"),
        (date(2026, 11, 24), "Defense", "20 min presentation with a Streamlit demo.", "future"),
    ],
    "TEXT": [
        (date(2026, 10, 10), "Text cleaning", "HTML removed, lowercased, title and description joined."),
        (date(2026, 10, 13), "First baseline", "TF-IDF + linear SVM: 0.8274. Cleaning gave no gain "
                                               "(raw text 0.8276), so we tested instead of assuming."),
        (date(2026, 10, 25), "Seed check", "Binary TF-IDF gains 0.33 points, but scores move 0.71 "
                                           "between seeds. Not a clear win."),
        (date(2026, 10, 26), "Feature limit", "The 50000 limit dropped 89% of the words. "
                                              "All features: +1.2 points (CV on train)."),
        (date(2026, 10, 28), "Grid search", "3-fold CV on train picks binary TF-IDF and C = 1. "
                                            "SVM beats Logistic Regression and Naive Bayes: 0.8436."),
        (date(2026, 11, 2), "Word weights", "Mostly clear product words, but also seller names. "
                                            "Short titles without a book word fail."),
        (date(2026, 11, 3), "CamemBERT", "French transformer, 2 epochs: 0.8764. Largest gains "
                                         "on books (+9 to +15 points)."),
        (date(2026, 11, 4), "CamemBERT re-run", "Logs train and validation loss to check for overfitting."),
    ],
    "IMAGE": [
        (date(2026, 10, 7), "Duplicate audit", "Identical images found by comparing file hashes."),
        (date(2026, 10, 10), "Image preprocessing", "Duplicate and conflicting images removed from train."),
        (date(2026, 10, 21), "Frozen ResNet18", "Pre-trained network, only the last layer trained, "
                                                "1 epoch: 0.48."),
        (date(2026, 10, 26), "Validation fix", "Validation keeps all 16984 products, the same as text."),
        (date(2026, 11, 2), "ResNet18 fine-tuning", "Trained on Colab. Last block unfrozen: 0.62. "
                                                    "Full network: 0.63, but validation loss rises after epoch 2."),
        (date(2026, 11, 2), "ConvNeXt-tiny", "A newer image network: 0.70, the best image model so far."),
    ],
    "FUSION": [
        (date(2026, 10, 25), "Text vs image", "Image is weaker on every category, "
                                              "but closest to text on books."),
        (date(2026, 10, 29), "Simple fusion", "70% text, 30% image: 0.8501. "
                                              "Books gain the most."),
        (date(2026, 11, 3), "Random Forest", "Learns how to combine the probabilities. "
                                             "SVM + image: 0.8630."),
        (date(2026, 11, 3), "Best model", "Random Forest on CamemBERT + SVM + image: 0.8919."),
    ],
}

LABEL_DAYS = 8.2  # width of one label, in days, used to avoid overlaps
LEVEL_HEIGHT = 0.058
LANE_LEVELS = {"PROJECT": 2, "TEXT": 4, "IMAGE": 3, "FUSION": 3}
LANE_GAP = 0.03


def x_of(d):
    left, right = 0.08, 0.985
    return left + (right - left) * (d - START).days / (END - START).days


def assign_levels(items, n_levels):
    last_day = [None] * n_levels
    levels = []
    for item in items:
        day = item[0].toordinal()
        free = [k for k in range(n_levels) if last_day[k] is None or day - last_day[k] >= LABEL_DAYS]
        level = free[0] if free else min(range(n_levels), key=lambda k: last_day[k])
        last_day[level] = day
        levels.append(level)
    return levels


def main():
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig = plt.figure(figsize=(26, 17.5), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.text(0.02, 0.965, "Rakuten product classification: project timeline", fontsize=20,
            fontweight="bold", color=INK)
    ax.text(0.02, 0.94, "What we did, when, and why. Scores are weighted F1 on the same 16984 validation products.",
            fontsize=11.5, color=MUTED)

    axis_y = 0.86
    # Phase bands and bars
    for i, (step, name, a, b) in enumerate(PHASES):
        if i % 2 == 0:
            ax.add_patch(plt.Rectangle((x_of(a), axis_y - 0.05), x_of(b) - x_of(a), 0.05,
                                       facecolor=BAND, edgecolor="none", zorder=0))
        ax.plot([x_of(a) + 0.002, x_of(b) - 0.002], [axis_y, axis_y], color=INK, linewidth=4,
                solid_capstyle="butt", zorder=2)
        ax.text(x_of(a) + 0.003, axis_y - 0.02, step, fontsize=9, fontweight="bold", color=INK, va="top")
        ax.text(x_of(a) + 0.003, axis_y - 0.038, name, fontsize=9, color=MUTED, va="top")

    # Month and week ticks
    ax.plot([x_of(START), x_of(END)], [axis_y + 0.012, axis_y + 0.012], color=LINE, linewidth=1)
    for month, label in [(9, "SEPTEMBER"), (10, "OCTOBER"), (11, "NOVEMBER")]:
        first = max(date(2026, month, 1), START)
        ax.text(x_of(first), axis_y + 0.036, label, fontsize=10, color=INK, fontweight="bold")
    for ordinal in range(START.toordinal(), END.toordinal() + 1):
        d = date.fromordinal(ordinal)
        if d.weekday() == 0:
            ax.plot([x_of(d)] * 2, [axis_y + 0.008, axis_y + 0.016], color=FAINT, linewidth=1)
            ax.text(x_of(d), axis_y + 0.019, str(d.day), fontsize=7.5, color=FAINT, ha="center", va="bottom")

    # Today marker
    ax.plot([x_of(TODAY)] * 2, [axis_y - 0.05, axis_y + 0.016], color=DATE, linewidth=2, zorder=3)
    ax.text(x_of(TODAY), axis_y - 0.058, "today", fontsize=9, color=DATE, fontweight="bold", ha="center", va="top")

    # Lanes
    top = axis_y - 0.09
    for lane, items in LANES.items():
        ax.text(0.02, top, lane, fontsize=11, fontweight="bold", color=INK, va="top")
        ax.plot([0.02, 0.065], [top - 0.022, top - 0.022], color=INK, linewidth=2)
        levels = assign_levels(items, LANE_LEVELS[lane])
        for item, level in zip(items, levels):
            d, title, text = item[:3]
            future = len(item) > 3
            x = x_of(d)
            y = top - level * LEVEL_HEIGHT
            ax.plot([x, x], [axis_y - 0.05 if lane == "PROJECT" else top + 0.022, y + 0.004],
                    color=FAINT, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
            box = dict(facecolor="white", edgecolor="none", pad=1.5)
            ax.text(x + 0.003, y, f"{d.day:02d}/{d.month:02d}", fontsize=9.5, fontweight="bold",
                    color=FAINT if future else DATE, va="top", bbox=box, zorder=4)
            ax.text(x + 0.03, y, title.upper(), fontsize=9.5, fontweight="bold",
                    color=FAINT if future else INK, va="top", bbox=box, zorder=4)
            ax.text(x + 0.003, y - 0.016, "\n".join(textwrap.wrap(text, 32)), fontsize=8.6,
                    color=FAINT if future else MUTED, va="top", linespacing=1.3, bbox=box, zorder=4)
        top -= LANE_LEVELS[lane] * LEVEL_HEIGHT + LANE_GAP

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_DIR / "project_timeline_lanes.png", dpi=160)
    fig.savefig(OUTPUT_DIR / "project_timeline_lanes.pdf")
    print(f"Saved {OUTPUT_DIR / 'project_timeline_lanes.png'}")


if __name__ == "__main__":
    main()
