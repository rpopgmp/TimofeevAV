import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)


def top_k_recall(labels, probs, capacity=0.05):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    M = int(np.ceil(capacity * len(labels)))
    order = np.argsort(-probs)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    return float(labels[order[:M]].sum() / n_pos)


def best_f1_threshold(labels, probs):
    p, r, t = precision_recall_curve(labels, probs)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    f1 = f1[:-1]
    if len(f1) == 0 or np.all(np.isnan(f1)):
        return 0.5, float("nan")
    i = int(np.nanargmax(f1))
    return float(t[i]), float(f1[i])


def compute_metrics(labels, probs, threshold=None):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    if threshold is None:
        threshold, _ = best_f1_threshold(labels, probs)
    preds = (probs >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(labels, probs)),
        "auc_pr": float(average_precision_score(labels, probs)),
        "top5_recall": top_k_recall(labels, probs, 0.05),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "threshold": float(threshold),
        "n_pos": int(labels.sum()),
        "n_neg": int((labels == 0).sum()),
    }


def plot_roc(labels, probs, title="ROC", savepath=None):
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], lw=1, color="gray", linestyle="--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()
    return fig, ax


def save_metrics(metrics, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
