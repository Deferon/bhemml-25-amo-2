"""Generate evaluation plots for artifacts."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import RocCurveDisplay, confusion_matrix


def save_confusion_matrix(y_true, y_pred, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_roc_curve(y_true, y_proba, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_feature_importance(importances: dict, path: Path, top_n: int = 15) -> None:
    items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names, values = zip(*items) if items else ([], [])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names[::-1], values[::-1], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importance")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
