"""Metrics and plots that every model in the project is judged by.

All models (baselines, CNNs) go through the same functions here, so their
numbers are directly comparable.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from src.data import CLASS_NAMES, SHORT_NAMES
from src.plotting import SEQUENTIAL_CMAP, plt, save_figure

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"


def compute_metrics(y_true, y_pred, class_names=CLASS_NAMES):
    """Return accuracy, macro F1, and per class precision, recall, F1, and support.

    Accuracy counts every image equally, so common classes dominate it.
    Macro F1 averages the F1 score of each class, so every class counts equally.
    """
    labels = list(range(len(class_names)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": {
            name: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for name, p, r, f, s in zip(class_names, precision, recall, f1, support)
        },
    }


def per_class_table(metrics):
    """Turn the per class part of compute_metrics output into a DataFrame."""
    table = pd.DataFrame(metrics["per_class"]).T
    table.index.name = "class"
    return table


def save_metrics(metrics, name):
    """Write metrics to results/metrics/<name>.json and return the path."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / f"{name}.json"
    path.write_text(json.dumps(metrics, indent=2))
    return path


def load_metrics(name):
    return json.loads((METRICS_DIR / f"{name}.json").read_text())


def plot_confusion_matrix(y_true, y_pred, title, name):
    """Plot a confusion matrix where each row is divided by its row total.

    Rows are the true class and columns the predicted class, so the diagonal
    shows per class recall and each off diagonal cell shows how often one
    class was mistaken for another.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(SHORT_NAMES))))
    row_share = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(row_share, cmap=SEQUENTIAL_CMAP, vmin=0, vmax=1)
    ax.grid(False)
    for i in range(len(SHORT_NAMES)):
        for j in range(len(SHORT_NAMES)):
            value = row_share[i, j]
            if value >= 0.005:
                ax.text(j, i, f"{100 * value:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if value > 0.55 else "#0b0b0b")
    ax.set_xticks(range(len(SHORT_NAMES)), SHORT_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(len(SHORT_NAMES)), SHORT_NAMES)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title, loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Share of the true class (percent shown in cells)")
    return save_figure(fig, name)
