"""Visualization utilities for Model B (PyTorch)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Set non-interactive backend for matplotlib to avoid tkinter errors
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, auc


sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100


def _ensure_dir(save_path: Optional[Path]) -> None:
    if save_path is None:
        return
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)


def plot_sample_images(
    images: np.ndarray,
    labels: np.ndarray,
    n_samples: int = 9,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 12),
) -> None:
    """Plot sample images for Model B (expects images as numpy arrays)."""
    if class_names is None:
        class_names = ["Benign", "Malignant"]

    # Ensure images are (N, H, W)
    if len(images.shape) == 4 and images.shape[1] == 1:
        images = images.squeeze(1)

    n_rows = int(np.ceil(np.sqrt(n_samples)))
    n_cols = int(np.ceil(n_samples / n_rows))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_samples > 1 else [axes]

    # Try to balance classes
    unique_labels = np.unique(labels)
    per_class = max(1, n_samples // max(1, len(unique_labels)))
    indices: List[int] = []
    for lab in unique_labels:
        lab_idx = np.where(labels == lab)[0]
        if len(lab_idx) == 0:
            continue
        take = min(per_class, len(lab_idx))
        indices.extend(np.random.choice(lab_idx, size=take, replace=False).tolist())
    # Fill remaining randomly
    if len(indices) < n_samples:
        remaining = np.setdiff1d(np.arange(len(images)), np.array(indices, dtype=int))
        if len(remaining) > 0:
            extra = np.random.choice(remaining, size=min(n_samples - len(indices), len(remaining)), replace=False)
            indices.extend(extra.tolist())
    indices = indices[:n_samples]

    for i, ax in enumerate(axes):
        if i >= len(indices):
            ax.axis("off")
            continue
        idx = indices[i]
        img = images[idx].astype(np.float32, copy=False)

        # Display-friendly scaling
        img_disp = img.copy()
        if img_disp.min() < 0:  # z-score
            img_disp = (img_disp - img_disp.min()) / (img_disp.max() - img_disp.min() + 1e-8)
        elif img_disp.max() <= 1.0:  # minmax
            img_disp = img_disp
        else:
            img_disp = img_disp / 255.0

        ax.imshow(img_disp, cmap="gray")
        ax.set_title(f"Label: {class_names[int(labels[idx])]}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    if save_path is not None:
        _ensure_dir(save_path)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_class_distribution(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 5),
) -> None:
    """Plot class distribution across splits (same as Model A, but kept local to Model B)."""
    if class_names is None:
        class_names = ["Benign", "Malignant"]

    def _count(y: np.ndarray) -> Dict[int, int]:
        vals, counts = np.unique(y, return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, counts)}

    train_c, val_c, test_c = _count(y_train), _count(y_val), _count(y_test)
    classes = sorted(set(train_c.keys()) | set(val_c.keys()) | set(test_c.keys()))

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width, [train_c.get(c, 0) for c in classes], width, label="Train")
    ax.bar(x, [val_c.get(c, 0) for c in classes], width, label="Val")
    ax.bar(x + width, [test_c.get(c, 0) for c in classes], width, label="Test")
    ax.set_xticks(x)
    ax.set_xticklabels([class_names[c] if c < len(class_names) else str(c) for c in classes])
    ax.set_title("Class Distribution (Model B)")
    ax.set_ylabel("Count")
    ax.legend()
    plt.tight_layout()
    if save_path is not None:
        _ensure_dir(save_path)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_training_curves(history: Dict[str, List[float]], save_path: Optional[Path] = None) -> None:
    """Plot train/val loss and accuracy curves from trainer history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(history.get("train_loss", []), label="Train Loss")
    axes[0].plot(history.get("val_loss", []), label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    # Accuracy
    axes[1].plot(history.get("train_acc", []), label="Train Acc")
    axes[1].plot(history.get("val_acc", []), label="Val Acc")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    if save_path is not None:
        _ensure_dir(save_path)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    save_path: Optional[Path] = None,
    class_names: Optional[List[str]] = None,
) -> None:
    """Plot a single confusion matrix."""
    if class_names is None:
        class_names = ["Benign", "Malignant"]
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names, rotation=0)
    plt.tight_layout()
    if save_path is not None:
        _ensure_dir(save_path)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_side_by_side_confusion_matrices(
    y_true: np.ndarray,
    y_pred_without: np.ndarray,
    y_pred_with: np.ndarray,
    save_path: Optional[Path] = None,
    class_names: Optional[List[str]] = None,
) -> None:
    """Plot confusion matrices for augmentation OFF vs ON."""
    if class_names is None:
        class_names = ["Benign", "Malignant"]

    cm_without = confusion_matrix(y_true, y_pred_without)
    cm_with = confusion_matrix(y_true, y_pred_with)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, cm, title in [
        (axes[0], cm_without, "Without Augmentation"),
        (axes[1], cm_with, "With Augmentation"),
    ]:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names, rotation=0)

    plt.tight_layout()
    if save_path is not None:
        _ensure_dir(save_path)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba_pos: np.ndarray,
    title: str,
    save_path: Optional[Path] = None,
) -> float:
    """Plot ROC curve for probabilities of positive class; returns AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_proba_pos)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title(title)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save_path is not None:
        _ensure_dir(save_path)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    return float(roc_auc)


def plot_augmentation_roc_curves(
    y_true: np.ndarray,
    y_proba_without: np.ndarray,
    y_proba_with: np.ndarray,
    save_path: Optional[Path] = None,
) -> None:
    """Plot ROC curves for augmentation OFF vs ON (expects Nx2 probabilities)."""
    fpr_wout, tpr_wout, _ = roc_curve(y_true, y_proba_without[:, 1])
    auc_wout = auc(fpr_wout, tpr_wout)
    fpr_w, tpr_w, _ = roc_curve(y_true, y_proba_with[:, 1])
    auc_w = auc(fpr_w, tpr_w)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr_wout, tpr_wout, label=f"Without Aug (AUC={auc_wout:.3f})")
    ax.plot(fpr_w, tpr_w, label=f"With Aug (AUC={auc_w:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("ROC Curves: Without vs With Augmentation (Model B)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save_path is not None:
        _ensure_dir(save_path)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def plot_augmentation_comparison(
    metrics_without: Dict[str, Any],
    metrics_with: Dict[str, Any],
    save_path: Optional[Path] = None,
) -> None:
    """Bar chart comparing key test metrics for augmentation OFF vs ON."""
    keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    vals_without = [float(metrics_without.get(k, 0.0)) for k in keys]
    vals_with = [float(metrics_with.get(k, 0.0)) for k in keys]

    x = np.arange(len(keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, vals_without, width, label="Without Aug")
    ax.bar(x + width / 2, vals_with, width, label="With Aug")
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("_", "-").upper() for k in keys])
    ax.set_ylim(0, 1.0)
    ax.set_title("Test Metrics: Without vs With Augmentation (Model B)")
    ax.legend()
    plt.tight_layout()
    if save_path is not None:
        _ensure_dir(save_path)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()

