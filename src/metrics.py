from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


def compute_weighted_f1(y_true, y_pred):
    """Calculate the Weighted-F1 score (official challenge metric)."""
    return f1_score(y_true, y_pred, average="weighted")


def compute_macro_f1(y_true, y_pred):
    """Calculate the Macro-F1 score."""
    return f1_score(y_true, y_pred, average="macro")


def plot_confusion_matrix(y_true, y_pred, classes=None):
    """Plot normalized confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=classes, normalize="true")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()
