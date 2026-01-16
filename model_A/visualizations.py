"""Visualization utilities for Model A."""

import numpy as np

# Set non-interactive backend for matplotlib to avoid tkinter errors
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any, List
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedKFold, learning_curve
import warnings

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100


def plot_sample_images(
    images: np.ndarray,
    labels: np.ndarray,
    n_samples: int = 9,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 12),
):
    """
    Plot sample images from the dataset.

    Args:
        images: Array of images (N, H, W) or (N, 1, H, W)
        labels: Array of labels (N,)
        n_samples: Number of samples to display
        class_names: Optional list of class names
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    if class_names is None:
        class_names = ["Benign", "Malignant"]

    # Ensure images are 2D
    if len(images.shape) == 4:
        images = images.squeeze(axis=1)

    n_rows = int(np.ceil(np.sqrt(n_samples)))
    n_cols = int(np.ceil(n_samples / n_rows))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_samples > 1 else [axes]

    # Get random samples with balanced classes if possible
    unique_labels, counts = np.unique(labels, return_counts=True)
    samples_per_class = n_samples // len(unique_labels)

    indices = []
    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        n_samples_label = min(samples_per_class, len(label_indices))
        indices.extend(
            np.random.choice(label_indices, n_samples_label, replace=False).tolist()
        )

    # Add random samples to fill up to n_samples
    remaining = n_samples - len(indices)
    if remaining > 0:
        all_indices = np.arange(len(images))
        remaining_indices = np.setdiff1d(all_indices, indices)
        if len(remaining_indices) > 0:
            additional = np.random.choice(
                remaining_indices, min(remaining, len(remaining_indices)), replace=False
            )
            indices.extend(additional.tolist())

    indices = indices[:n_samples]

    for idx, ax in enumerate(axes):
        if idx < len(indices):
            img_idx = indices[idx]
            img = images[img_idx]
            label = labels[img_idx]

            # Handle normalized images
            img_display = img.copy()
            if img_display.min() < 0:  # Z-score normalized
                img_display = (img_display - img_display.min()) / (
                    img_display.max() - img_display.min()
                )
            elif img_display.max() <= 1.0:  # Min-max normalized
                img_display = img_display * 255

            ax.imshow(img_display, cmap="gray")
            ax.set_title(f"Label: {class_names[label]}", fontsize=10)
            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Sample images saved to: {save_path}")
    # Note: Using Agg backend (non-interactive), so plt.show() is not needed

    plt.close()


def plot_class_distribution(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5),
):
    """
    Plot class distribution across train, validation, and test sets.

    Args:
        y_train: Training labels
        y_val: Validation labels
        y_test: Test labels
        class_names: Optional list of class names
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    if class_names is None:
        class_names = ["Benign", "Malignant"]

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    splits = [("Train", y_train), ("Validation", y_val), ("Test", y_test)]

    for idx, (split_name, labels) in enumerate(splits):
        unique, counts = np.unique(labels, return_counts=True)
        colors = ["#3498db", "#e74c3c"]

        bars = axes[idx].bar(
            [class_names[u] for u in unique], counts, color=[colors[u] for u in unique]
        )
        axes[idx].set_title(
            f"{split_name} Set\n(n={len(labels)})", fontsize=12, fontweight="bold"
        )
        axes[idx].set_ylabel("Count", fontsize=10)
        axes[idx].set_xlabel("Class", fontsize=10)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[idx].text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}\n({height/len(labels)*100:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        axes[idx].grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Class distribution plot saved to: {save_path}")
    # Note: Using Agg backend (non-interactive), so plt.show() is not needed

    plt.close()


def plot_confusion_matrices(
    results_dict: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None,
    figsize: tuple = (15, 5),
):
    """
    Plot confusion matrices for all feature modes.

    Args:
        results_dict: Dictionary mapping feature mode to results
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    n_modes = len(results_dict)
    fig, axes = plt.subplots(1, n_modes, figsize=figsize)
    if n_modes == 1:
        axes = [axes]

    for idx, (mode, results) in enumerate(results_dict.items()):
        if "test" in results and "confusion_matrix" in results["test"]:
            cm = np.array(results["test"]["confusion_matrix"])

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=axes[idx],
                cbar_kws={"label": "Count"},
                square=True,
                linewidths=0.5,
            )
            axes[idx].set_title(
                f"{mode.upper()} Features\nTest Set", fontsize=12, fontweight="bold"
            )
            axes[idx].set_xlabel("Predicted", fontsize=10)
            axes[idx].set_ylabel("Actual", fontsize=10)
            axes[idx].set_xticklabels(["Benign", "Malignant"])
            axes[idx].set_yticklabels(["Benign", "Malignant"])

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Confusion matrices saved to: {save_path}")
    # Note: Using Agg backend (non-interactive), so plt.show() is not needed

    plt.close()


def plot_roc_curves(
    y_true_dict: Dict[str, np.ndarray],
    y_proba_dict: Dict[str, np.ndarray],
    save_path: Optional[str] = None,
    figsize: tuple = (10, 8),
):
    """
    Plot ROC curves for all feature modes.

    Args:
        y_true_dict: Dictionary mapping feature mode to true labels
        y_proba_dict: Dictionary mapping feature mode to predicted probabilities
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]

    for idx, (mode, y_true) in enumerate(y_true_dict.items()):
        if mode in y_proba_dict:
            y_proba = y_proba_dict[mode]
            # Get probabilities for positive class
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                y_proba_pos = y_proba[:, 1]
            else:
                y_proba_pos = y_proba

            fpr, tpr, _ = roc_curve(y_true, y_proba_pos)
            roc_auc = auc(fpr, tpr)

            ax.plot(
                fpr,
                tpr,
                color=colors[idx % len(colors)],
                lw=2,
                label=f"{mode.upper()} (AUC = {roc_auc:.3f})",
            )

    # Plot diagonal reference line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, lw=1, label="Random (AUC = 0.500)")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves Comparison\n(Test Set)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"ROC curves saved to: {save_path}")
    # Note: Using Agg backend (non-interactive), so plt.show() is not needed

    plt.close()


def plot_metrics_comparison(
    results_dict: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None,
    figsize: tuple = (14, 8),
):
    """
    Plot bar chart comparing metrics across different feature modes.

    Args:
        results_dict: Dictionary mapping feature mode to results
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    modes = list(results_dict.keys())

    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]

    colors = sns.color_palette("husl", len(modes))

    for metric_idx, metric in enumerate(metrics):
        values = []
        labels = []

        for mode in modes:
            if "test" in results_dict[mode] and "metrics" in results_dict[mode]["test"]:
                if metric in results_dict[mode]["test"]["metrics"]:
                    values.append(results_dict[mode]["test"]["metrics"][metric])
                    labels.append(mode.upper())

        if values:
            bars = axes[metric_idx].bar(labels, values, color=colors[: len(values)])
            axes[metric_idx].set_title(
                metric.replace("_", " ").title(), fontsize=11, fontweight="bold"
            )
            axes[metric_idx].set_ylabel("Score", fontsize=10)
            axes[metric_idx].set_ylim([0, 1.1])
            axes[metric_idx].grid(axis="y", alpha=0.3)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[metric_idx].text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Metrics comparison plot saved to: {save_path}")
    # Note: Using Agg backend (non-interactive), so plt.show() is not needed

    plt.close()


def plot_feature_comparison_table(
    results_dict: Dict[str, Dict[str, Any]],
    save_path: Optional[str] = None,
    figsize: tuple = (12, 6),
):
    """
    Create a table visualization comparing all metrics across feature modes.

    Args:
        results_dict: Dictionary mapping feature mode to results
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    modes = list(results_dict.keys())

    # Prepare data
    data = []
    for mode in modes:
        row = [mode.upper()]
        if "test" in results_dict[mode] and "metrics" in results_dict[mode]["test"]:
            for metric in metrics:
                if metric in results_dict[mode]["test"]["metrics"]:
                    row.append(f"{results_dict[mode]['test']['metrics'][metric]:.4f}")
                else:
                    row.append("N/A")
        data.append(row)

    # Create table
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("tight")
    ax.axis("off")

    columns = ["Feature Mode"] + [m.replace("_", " ").title() for m in metrics]
    table = ax.table(cellText=data, colLabels=columns, cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Style cells
    for i in range(1, len(data) + 1):
        for j in range(len(columns)):
            if j == 0:
                table[(i, j)].set_facecolor("#E8F5E9")
                table[(i, j)].set_text_props(weight="bold")
            else:
                table[(i, j)].set_facecolor("#F1F8E9")

    plt.title(
        "Feature Modes Performance Comparison (Test Set)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Feature comparison table saved to: {save_path}")
    # Note: Using Agg backend (non-interactive), so plt.show() is not needed

    plt.close()


def plot_augmentation_comparison(
    results_without_aug: Dict[str, Any],
    results_with_aug: Dict[str, Any],
    save_path: Optional[Path] = None,
):
    """
    Plot comparison of metrics with and without augmentation.

    Args:
        results_without_aug: Results dictionary from training without augmentation
        results_with_aug: Results dictionary from training with augmentation
        save_path: Optional path to save the figure
    """
    # Extract metrics for test set
    if (
        "test" not in results_without_aug
        or "metrics" not in results_without_aug["test"]
    ):
        print("Warning: Test metrics not found in results_without_aug")
        return

    if "test" not in results_with_aug or "metrics" not in results_with_aug["test"]:
        print("Warning: Test metrics not found in results_with_aug")
        return

    metrics_without = results_without_aug["test"]["metrics"]
    metrics_with = results_with_aug["test"]["metrics"]

    # Prepare data
    metrics_names = ["accuracy", "precision", "recall", "f1_score"]
    if "roc_auc" in metrics_without and "roc_auc" in metrics_with:
        metrics_names.append("roc_auc")

    without_values = [metrics_without.get(m, 0.0) for m in metrics_names]
    with_values = [metrics_with.get(m, 0.0) for m in metrics_names]

    # Create bar plot
    x = np.arange(len(metrics_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(
        x - width / 2,
        without_values,
        width,
        label="Without Augmentation",
        color="#FF6B6B",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        with_values,
        width,
        label="With Augmentation",
        color="#4ECDC4",
        alpha=0.8,
    )

    # Format labels
    labels = [m.replace("_", " ").title() for m in metrics_names]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "Performance Comparison: Without vs With Data Augmentation (Test Set)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim([0, 1.1])

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Augmentation comparison plot saved to: {save_path}")
    # Note: Using Agg backend (non-interactive), so plt.show() is not needed

    plt.close()


def plot_augmentation_confusion_matrices(
    results_without_aug: Dict[str, Any],
    results_with_aug: Dict[str, Any],
    save_path: Optional[Path] = None,
):
    """
    Plot confusion matrices for both augmentation settings.

    Args:
        results_without_aug: Results dictionary from training without augmentation
        results_with_aug: Results dictionary from training with augmentation
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (results, title_suffix) in enumerate(
        [
            (results_without_aug, "Without Augmentation"),
            (results_with_aug, "With Augmentation"),
        ]
    ):
        if "test" not in results or "confusion_matrix" not in results["test"]:
            axes[idx].text(
                0.5,
                0.5,
                "Confusion matrix not available",
                ha="center",
                va="center",
                transform=axes[idx].transAxes,
            )
            axes[idx].set_title(title_suffix, fontsize=12, fontweight="bold")
            continue

        cm = np.array(results["test"]["confusion_matrix"])

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=axes[idx],
            cbar_kws={"label": "Count"},
            square=True,
            linewidths=1,
        )

        axes[idx].set_xlabel("Predicted Label", fontsize=11)
        axes[idx].set_ylabel("True Label", fontsize=11)
        axes[idx].set_title(
            f"{title_suffix}\n(Test Set)", fontsize=12, fontweight="bold"
        )
        axes[idx].set_xticklabels(["Benign", "Malignant"])
        axes[idx].set_yticklabels(["Benign", "Malignant"])

    plt.suptitle(
        "Confusion Matrices: Without vs With Augmentation",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Augmentation confusion matrices saved to: {save_path}")
    # Note: Using Agg backend (non-interactive), so plt.show() is not needed

    plt.close()


def plot_augmentation_roc_curves(
    y_true: np.ndarray,
    y_proba_without: np.ndarray,
    y_proba_with: np.ndarray,
    save_path: Optional[Path] = None,
):
    """
    Plot ROC curves for both augmentation settings.

    Args:
        y_true: True labels
        y_proba_without: Predicted probabilities without augmentation (N, 2)
        y_proba_with: Predicted probabilities with augmentation (N, 2)
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Compute ROC curves
    fpr_without, tpr_without, _ = roc_curve(y_true, y_proba_without[:, 1])
    fpr_with, tpr_with, _ = roc_curve(y_true, y_proba_with[:, 1])

    # Compute AUC
    auc_without = auc(fpr_without, tpr_without)
    auc_with = auc(fpr_with, tpr_with)

    # Plot ROC curves
    ax.plot(
        fpr_without,
        tpr_without,
        label=f"Without Augmentation (AUC = {auc_without:.3f})",
        linewidth=2,
        color="#FF6B6B",
    )
    ax.plot(
        fpr_with,
        tpr_with,
        label=f"With Augmentation (AUC = {auc_with:.3f})",
        linewidth=2,
        color="#4ECDC4",
    )
    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier", linewidth=1, alpha=0.5)

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(
        "ROC Curves: Without vs With Augmentation",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Augmentation ROC curves saved to: {save_path}")
    # Note: Using Agg backend (non-interactive), so plt.show() is not needed

    plt.close()


def plot_learning_curve_model_a(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    scoring: str = "accuracy",
    random_state: int = 42,
    train_sizes: Optional[np.ndarray] = None,
    title: str = "Model A Learning Curve",
    save_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
):
    """
    Plot a scikit-learn learning curve for Model A.

    Note: For classical models (e.g., SVM) there is no epoch-by-epoch training curve.
    A learning curve shows how performance scales with training set size.

    Args:
        estimator: Fitted or unfitted sklearn estimator (must be cloneable)
        X: Feature matrix (N, D)
        y: Labels (N,)
        cv: Number of CV folds (StratifiedKFold)
        scoring: sklearn scoring string (default: accuracy)
        random_state: Seed for shuffling CV splits
        train_sizes: Fractions or absolute sizes for learning_curve
        title: Plot title
        save_path: Optional path to save the figure
        figsize: Figure size
    """
    if train_sizes is None:
        # Use fractions to adapt to different dataset sizes
        train_sizes = np.linspace(0.1, 1.0, 6)

    cv_splitter = StratifiedKFold(
        n_splits=int(cv), shuffle=True, random_state=int(random_state)
    )

    sizes, train_scores, val_scores = learning_curve(
        estimator=estimator,
        X=X,
        y=y,
        cv=cv_splitter,
        scoring=scoring,
        train_sizes=train_sizes,
        n_jobs=-1,
    )

    train_mean = np.nanmean(train_scores, axis=1)
    train_std = np.nanstd(train_scores, axis=1)
    val_mean = np.nanmean(val_scores, axis=1)
    val_std = np.nanstd(val_scores, axis=1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(sizes, train_mean, "o-", lw=2, label="Training score")
    ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.15)
    ax.plot(sizes, val_mean, "o-", lw=2, label=f"CV score ({cv}-fold)")
    ax.fill_between(sizes, val_mean - val_std, val_mean + val_std, alpha=0.15)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Number of training samples", fontsize=12)
    ax.set_ylabel(scoring.replace("_", " ").title(), fontsize=12)
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_ylim(0.0, 1.05)
    ax.legend(loc="best", fontsize=11)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Learning curve saved to: {save_path}")
    plt.close()
