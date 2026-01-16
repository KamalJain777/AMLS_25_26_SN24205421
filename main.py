"""Main entry point for AMLS Assignment - BreastMNIST."""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import yaml

# Set non-interactive backend for matplotlib to avoid tkinter errors
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from model_A import (
    BreastMNISTDataset,
    FeatureExtractor,
    train_model_a,
    evaluate_model_a,
    compare_feature_modes,
    compare_augmentation,
)
from model_A.augmentation import augment_dataset
from model_B import (
    get_breastmnist_dataloaders,
    ModelBNet,
    train_model_b,
    evaluate_model_b,
)


def _safe_get(d: Dict[str, Any], path: str, default=None):
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


# ==========================
# Cross-model comparison plots
# (merged from comparison_visualizations.py)
# ==========================


@dataclass
class _ModelResultForComparison:
    name: str
    variant: str
    test_metrics: Dict[str, float]
    test_confusion_matrix: np.ndarray
    test_probabilities: Optional[np.ndarray]  # shape (N, 2) if available


def _pick_model_a_variant_for_comparison(
    model_a_json: Dict[str, Any],
) -> tuple[str, Dict[str, Any]]:
    """Pick the best (highest test roc_auc) feature mode for Model A."""
    if model_a_json.get("run_type") == "feature_comparison":
        results_by_mode = model_a_json.get("results_by_mode", {})
        best_mode = None
        best_auc = -1.0
        best_obj = None
        for mode, res in results_by_mode.items():
            m = _safe_get(res, "test.metrics", {})
            if not isinstance(m, dict):
                continue
            auc_v = float(m.get("roc_auc", -1.0))
            if auc_v > best_auc:
                best_auc = auc_v
                best_mode = mode
                best_obj = res
        if best_mode and best_obj:
            return best_mode, best_obj
    return "default", model_a_json


def _pick_model_b_variant_for_comparison(
    model_b_json: Dict[str, Any],
) -> tuple[str, Dict[str, Any]]:
    """Pick the with-augmentation variant if available."""
    if model_b_json.get("run_type") == "augmentation_ablation":
        with_aug = model_b_json.get("with_augmentation", None)
        if isinstance(with_aug, dict):
            return "with_augmentation", with_aug
        without_aug = model_b_json.get("without_augmentation", None)
        if isinstance(without_aug, dict):
            return "without_augmentation", without_aug
    if model_b_json.get("run_type") == "single_run" and isinstance(
        model_b_json.get("run"), dict
    ):
        return "single_run", model_b_json["run"]
    return "default", model_b_json


def _load_model_comparison_inputs(
    base_out_dir: Path,
    data_dir: Path = Path("Datasets"),
    random_state: int = 42,
) -> tuple[
    Optional[_ModelResultForComparison],
    Optional[_ModelResultForComparison],
    Optional[np.ndarray],
]:
    """Load Model A and Model B test outputs, plus y_test labels."""
    import json

    base_out_dir = Path(base_out_dir)
    model_a_path = base_out_dir / "model_A" / "model_A_results.json"
    model_b_path = base_out_dir / "model_B" / "model_B_results.json"

    a: Optional[_ModelResultForComparison] = None
    b: Optional[_ModelResultForComparison] = None

    if model_a_path.exists():
        with open(model_a_path, "r", encoding="utf-8") as f:
            a_json = json.load(f)
        a_mode, a_obj = _pick_model_a_variant_for_comparison(a_json)
        a_metrics = _safe_get(a_obj, "test.metrics", {}) or {}
        a_cm = np.array(_safe_get(a_obj, "test.confusion_matrix", [[0, 0], [0, 0]]))
        a_probs = _safe_get(a_obj, "test.probabilities", None)
        a_probs_arr = np.array(a_probs) if a_probs is not None else None
        a = _ModelResultForComparison(
            name="Model A",
            variant=f"test ({a_mode})",
            test_metrics={k: float(v) for k, v in a_metrics.items()},
            test_confusion_matrix=a_cm,
            test_probabilities=a_probs_arr,
        )

    if model_b_path.exists():
        with open(model_b_path, "r", encoding="utf-8") as f:
            b_json = json.load(f)
        b_var, b_obj = _pick_model_b_variant_for_comparison(b_json)
        b_metrics = _safe_get(b_obj, "test.metrics", None)
        b_cm = _safe_get(b_obj, "test.confusion_matrix", None)
        b_probs = _safe_get(b_obj, "test.probabilities", None)
        if b_metrics is None:
            # legacy file format
            b_metrics = _safe_get(b_obj, "test_results.metrics", {}) or {}
            b_cm = _safe_get(b_obj, "test_results.confusion_matrix", [[0, 0], [0, 0]])
            b_probs = _safe_get(b_obj, "test_results.probabilities", None)
        b = _ModelResultForComparison(
            name="Model B",
            variant=f"test ({b_var})",
            test_metrics={k: float(v) for k, v in (b_metrics or {}).items()},
            test_confusion_matrix=np.array(
                b_cm if b_cm is not None else [[0, 0], [0, 0]]
            ),
            test_probabilities=np.array(b_probs) if b_probs is not None else None,
        )

    # y_test is needed for ROC curves. Reload from dataset (test split is not subsampled).
    y_test = None
    try:
        ds = BreastMNISTDataset(
            data_dir=str(data_dir),
            normalize=False,
            normalize_mode="minmax",
            random_state=random_state,
        )
        _, _, _, _, _, y_test = ds.get_all_splits(train_fraction=1.0, val_fraction=1.0)
    except Exception:
        y_test = None

    return a, b, y_test


def _plot_model_comparison_metrics(
    a: _ModelResultForComparison, b: _ModelResultForComparison, save_path: Path
) -> None:
    keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    a_vals = [float(a.test_metrics.get(k, 0.0)) for k in keys]
    b_vals = [float(b.test_metrics.get(k, 0.0)) for k in keys]

    x = np.arange(len(keys))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, a_vals, width, label=f"{a.name} ({a.variant})")
    ax.bar(x + width / 2, b_vals, width, label=f"{b.name} ({b.variant})")
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("_", "-").upper() for k in keys])
    ax.set_ylim(0, 1.0)
    ax.set_title("Model A vs Model B — Test Metrics")
    ax.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def _plot_model_comparison_confusion_matrices(
    a: _ModelResultForComparison,
    b: _ModelResultForComparison,
    save_path: Path,
) -> None:
    import seaborn as sns

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, cm, title in [
        (axes[0], a.test_confusion_matrix, f"{a.name} ({a.variant})"),
        (axes[1], b.test_confusion_matrix, f"{b.name} ({b.variant})"),
    ]:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticklabels(["Benign", "Malignant"])
        ax.set_yticklabels(["Benign", "Malignant"], rotation=0)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def _plot_model_comparison_roc_curves(
    a: _ModelResultForComparison,
    b: _ModelResultForComparison,
    y_test: np.ndarray,
    save_path: Path,
) -> None:
    from sklearn.metrics import roc_curve, auc

    if a.test_probabilities is None or b.test_probabilities is None:
        raise ValueError("Missing probabilities for ROC curves.")
    if len(y_test) != len(a.test_probabilities) or len(y_test) != len(
        b.test_probabilities
    ):
        raise ValueError("y_test length does not match probabilities length.")

    fpr_a, tpr_a, _ = roc_curve(y_test, a.test_probabilities[:, 1])
    auc_a = auc(fpr_a, tpr_a)
    fpr_b, tpr_b, _ = roc_curve(y_test, b.test_probabilities[:, 1])
    auc_b = auc(fpr_b, tpr_b)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr_a, tpr_a, label=f"{a.name} ({a.variant}) AUC={auc_a:.3f}")
    ax.plot(fpr_b, tpr_b, label=f"{b.name} ({b.variant}) AUC={auc_b:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("Model A vs Model B — ROC Curves (Test)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def _generate_model_comparison_visualizations(
    base_out_dir: Path,
    data_dir: Path = Path("Datasets"),
    random_state: int = 42,
) -> Dict[str, Optional[str]]:
    a, b, y_test = _load_model_comparison_inputs(
        base_out_dir, data_dir=data_dir, random_state=random_state
    )
    if a is None or b is None:
        return {"error": "Missing Model A or Model B results."}

    out_dir = Path(base_out_dir)
    metrics_path = out_dir / "model_comparison_metrics.png"
    cm_path = out_dir / "model_comparison_confusion_matrices.png"
    roc_path = out_dir / "model_comparison_roc_curves.png"

    _plot_model_comparison_metrics(a, b, metrics_path)
    _plot_model_comparison_confusion_matrices(a, b, cm_path)

    roc_written = None
    if y_test is not None:
        try:
            _plot_model_comparison_roc_curves(a, b, y_test, roc_path)
            roc_written = str(roc_path)
        except Exception:
            roc_written = None

    return {
        "metrics_png": str(metrics_path),
        "confusion_png": str(cm_path),
        "roc_png": roc_written,
    }


def _format_metrics_table(metrics: Dict[str, Any]) -> str:
    keys = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
    rows = []
    for k in keys:
        v = metrics.get(k, None)
        if v is None:
            continue
        rows.append((k, float(v)))
    out = ["| Metric | Value |", "|--------|-------|"]
    for k, v in rows:
        out.append(f"| **{k.replace('_', '-').upper()}** | {v:.4f} |")
    return "\n".join(out)


def _load_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_model_a_test_metrics(model_a_path: Path) -> Optional[Dict[str, Any]]:
    """
    Model A can write either:
      - a direct results dict with keys train/val/test
      - OR a summary dict with results_by_mode
    We extract a single test metrics dict for the summary.
    """
    data = _load_json_if_exists(model_a_path)
    if not data:
        return None

    # direct results
    direct = _safe_get(data, "test.metrics", None)
    if isinstance(direct, dict):
        return direct

    # comparison summary: pick RAW if present, else pick best by roc_auc
    results_by_mode = _safe_get(data, "results_by_mode", None)
    if isinstance(results_by_mode, dict) and results_by_mode:
        if "raw" in results_by_mode and isinstance(results_by_mode["raw"], dict):
            m = _safe_get(results_by_mode["raw"], "test.metrics", None)
            if isinstance(m, dict):
                return m

        best = None
        best_auc = -1.0
        for mode, res in results_by_mode.items():
            m = _safe_get(res, "test.metrics", None)
            if not isinstance(m, dict):
                continue
            auc_v = float(m.get("roc_auc", -1.0))
            if auc_v > best_auc:
                best_auc = auc_v
                best = m
        return best

    return None


def _extract_model_b_test_metrics(model_b_path: Path) -> Optional[Dict[str, Any]]:
    """
    Model B can write either:
      - legacy: {test_results:{metrics:...}}
      - new: {without_augmentation:{test:{metrics}}, with_augmentation:{test:{metrics}}}
    For the high-level comparison, prefer with_augmentation if available.
    """
    data = _load_json_if_exists(model_b_path)
    if not data:
        return None

    # new (ablation)
    m_with = _safe_get(data, "with_augmentation.test.metrics", None)
    if isinstance(m_with, dict):
        return m_with

    # single-run
    m_single = _safe_get(data, "run.test.metrics", None)
    if isinstance(m_single, dict):
        return m_single

    m_legacy = _safe_get(data, "test_results.metrics", None)
    if isinstance(m_legacy, dict):
        return m_legacy

    return None


def write_comparison_summary(base_out_dir: Path) -> Optional[Path]:
    """
    Generate results/comparison_summary.md from the latest Model A and Model B results.
    """
    base_out_dir = Path(base_out_dir)
    model_a_path = base_out_dir / "model_A" / "model_A_results.json"
    model_b_path = base_out_dir / "model_B" / "model_B_results.json"
    out_path = base_out_dir / "comparison_summary.md"

    a_metrics = _extract_model_a_test_metrics(model_a_path)
    b_metrics = _extract_model_b_test_metrics(model_b_path)

    if not a_metrics and not b_metrics:
        return None

    lines = [
        "# Model A vs Model B Comparison Summary",
        "",
        "> Auto-generated by `main.py` from the latest results files.",
        "",
    ]

    if a_metrics:
        lines += [
            "## Model A (Classical ML) — Test Metrics",
            "",
            _format_metrics_table(a_metrics),
            "",
        ]
    else:
        lines += [
            "## Model A (Classical ML) — Test Metrics",
            "",
            "_No Model A results found._",
            "",
        ]

    if b_metrics:
        lines += [
            "## Model B (Deep Learning) — Test Metrics",
            "",
            _format_metrics_table(b_metrics),
            "",
        ]
    else:
        lines += [
            "## Model B (Deep Learning) — Test Metrics",
            "",
            "_No Model B results found._",
            "",
        ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def convert_to_serializable(obj):
    """Convert numpy/Path types into JSON-serializable Python types."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj


def write_json(path: Path, data: Any, indent: int = 2) -> None:
    """Write JSON to disk, ensuring parent directories exist."""
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(convert_to_serializable(data), f, indent=indent)


def remove_if_exists(path: Path) -> None:
    """Best-effort remove a file if it exists."""
    try:
        p = Path(path)
        if p.exists() and p.is_file():
            p.unlink()
    except Exception:
        # Best-effort cleanup only; don't fail due to file locks, etc.
        pass


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Dictionary containing configuration
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def visualize_samples(
    images: np.ndarray,
    labels: np.ndarray,
    class_names: list = ["Benign", "Malignant"],
    n_samples: int = 5,
    save_path: Optional[str] = None,
):
    """
    Visualize sample images from each class.

    Args:
        images: Images array of shape (N, H, W)
        labels: Labels array of shape (N,)
        class_names: Names of classes
        n_samples: Number of samples per class to visualize
        save_path: Optional path to save figure
    """
    num_classes = len(class_names)
    fig, axes = plt.subplots(
        num_classes, n_samples, figsize=(n_samples * 2, num_classes * 2)
    )

    if num_classes == 1:
        axes = axes.reshape(1, -1)

    for class_idx in range(num_classes):
        class_mask = labels == class_idx
        class_images = images[class_mask]

        if len(class_images) == 0:
            continue

        # Select random samples
        n_available = min(n_samples, len(class_images))
        indices = np.random.choice(len(class_images), size=n_available, replace=False)

        for i, idx in enumerate(indices):
            ax = axes[class_idx, i]
            img = class_images[idx]

            # Denormalize if needed (assuming z-score normalization)
            # This is a simple approach - may need adjustment based on actual normalization
            if img.min() < -1 or img.max() > 1:
                # Likely normalized, try to denormalize for visualization
                img_vis = np.clip(
                    (img - img.min()) / (img.max() - img.min() + 1e-8), 0, 1
                )
            else:
                img_vis = img

            ax.imshow(img_vis, cmap="gray")
            ax.axis("off")
            if i == 0:
                ax.set_ylabel(class_names[class_idx], fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to: {save_path}")
    # Note: Using Agg backend (non-interactive), so plt.show() is not needed

    plt.close()


def set_random_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_model_a(args):
    """Run Model A (classical ML) training and evaluation."""
    print("=" * 80)
    print("Model A: Classical ML (SVM)")
    print("=" * 80)

    # Load configuration
    if args.config:
        config = load_config(args.config)
        model_config = config.get("model", {})
        feature_config = config.get("features", {})
        data_config = config.get("data", {})
        random_state = config.get("random_state", 42)
    else:
        # Default configuration (with GridSearchCV enabled)
        model_config = {"use_grid_search": True, "cv": 5, "verbose": 2}
        feature_config = {"mode": args.feature_mode if args.feature_mode else "raw"}
        data_config = {
            "train_fraction": args.train_fraction,
            "val_fraction": args.val_fraction,
            "normalize": True,
        }
        random_state = 42

    set_random_seeds(random_state)

    # Load dataset
    print("\nLoading BreastMNIST dataset...")
    dataset = BreastMNISTDataset(
        data_dir=args.data_dir,
        normalize=data_config.get("normalize", True),
        normalize_mode="zscore",
        random_state=random_state,
    )

    # Summarize dataset
    dataset.summarize()

    # Get data splits
    train_fraction = data_config.get("train_fraction", args.train_fraction)
    val_fraction = data_config.get("val_fraction", args.val_fraction)

    X_train, y_train, X_val, y_val, X_test, y_test = dataset.get_all_splits(
        train_fraction=train_fraction, val_fraction=val_fraction
    )

    # Create output directory if it doesn't exist
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    # Visualize input data
    if args.out_dir:
        from model_A.visualizations import plot_sample_images, plot_class_distribution

        print("\nGenerating input data visualizations...")
        # Plot sample images
        plot_sample_images(
            X_train, y_train, n_samples=9, save_path=args.out_dir / "sample_images.png"
        )

        # Plot class distribution
        plot_class_distribution(
            y_train, y_val, y_test, save_path=args.out_dir / "class_distribution.png"
        )

    # Visualize samples if requested (legacy function)
    if args.show_examples:
        print("\nVisualizing sample images...")
        visualize_samples(
            X_train,
            y_train,
            n_samples=5,
            save_path=(
                args.out_dir / "sample_images_legacy.png" if args.out_dir else None
            ),
        )

    # Feature extraction
    feature_mode = feature_config.get(
        "mode", args.feature_mode if args.feature_mode else "raw"
    )

    # By default, compare all features (raw vs processed) for rubric requirement
    # This satisfies the requirement to compare performance between raw and processed features
    # User can override with --feature_mode <specific_mode> to test a single mode
    should_compare_features = args.compare_all_features or (
        (args.feature_mode is None) and not args.single_feature_mode
    )

    # If we're NOT generating comparisons this run, remove comparison outputs so the
    # results directory doesn't show stale PNG/CSV/JSON from a previous run.
    if args.out_dir and not should_compare_features:
        for filename in [
            "feature_comparison_results.csv",
            "feature_comparison_detailed.json",
            "confusion_matrices.png",
            "metrics_comparison.png",
            "feature_comparison_table.png",
            "roc_curves.png",
            "augmentation_comparison_results.csv",
            "augmentation_comparison_detailed.json",
            "augmentation_comparison.png",
            "augmentation_confusion_matrices.png",
            "augmentation_roc_curves.png",
        ]:
            remove_if_exists(args.out_dir / filename)

    if should_compare_features:
        # Compare all feature modes (raw vs processed features)
        print("\n" + "=" * 80)
        print("Feature Comparison: Raw vs Processed Features")
        print("=" * 80)
        print(
            "Comparing performance between raw and processed features as per rubric requirement:"
        )
        print("  • Raw features: Flattened pixel values (baseline)")
        print(
            "  • PCA features: Dimensionality-reduced representation via Principal Component Analysis"
        )
        print(
            "  • HOG features: Histogram of Oriented Gradients (edge/texture-based representation)"
        )
        print("=" * 80)

        feature_modes = ["raw", "pca", "hog"]
        all_results = {}

        for mode in feature_modes:
            print(f"\n{'=' * 80}")
            print(f"Feature Mode: {mode.upper()}")
            print(f"{'=' * 80}")

            # Extract features (fit on train only; transform val/test)
            extractor = FeatureExtractor(
                mode=mode,
                pca_components=feature_config.get("pca_components", 64),
                standardize_features=feature_config.get("standardize", False),
            )
            X_train_feat = extractor.fit_transform(X_train)
            X_val_feat = extractor.transform(X_val)
            X_test_feat = extractor.transform(X_test)

            # Print feature dimension information
            original_dim = X_train.shape[1] * X_train.shape[2]  # 28 * 28 = 784
            feature_dim = X_train_feat.shape[1]
            reduction = (
                ((original_dim - feature_dim) / original_dim * 100)
                if feature_dim < original_dim
                else 0
            )
            print(
                f"Feature dimensions: {feature_dim} (from {original_dim} original pixels)"
            )
            if reduction > 0:
                print(f"Dimensionality reduction: {reduction:.1f}%")

            # Train model
            # Combine configs for trainer (model config should be nested)
            # Update mode in config to match the actual feature mode being used
            trainer_config = {
                "model": model_config,
                **feature_config,
                **data_config,
                "mode": mode,  # Ensure config reflects the actual feature mode
                "random_state": random_state,
            }
            train_result = train_model_a(
                X_train_feat,
                y_train,
                X_val_feat,
                y_val,
                X_test_feat,
                y_test,
                config=trainer_config,
                feature_mode=mode,
            )

            all_results[mode] = train_result["results"]

        # Compare results and save
        comparison_output = (
            args.out_dir / "feature_comparison_results.csv"
            if args.out_dir
            else Path("Datasets/feature_comparison_results.csv")
        )
        compare_feature_modes(all_results, output_path=comparison_output)

        # Generate evaluation visualizations
        if args.out_dir:
            from model_A.visualizations import (
                plot_confusion_matrices,
                plot_metrics_comparison,
                plot_feature_comparison_table,
            )

            print("\nGenerating evaluation metric visualizations...")
            # Plot confusion matrices
            plot_confusion_matrices(
                all_results, save_path=args.out_dir / "confusion_matrices.png"
            )

            # Plot metrics comparison
            plot_metrics_comparison(
                all_results, save_path=args.out_dir / "metrics_comparison.png"
            )

            # Plot feature comparison table
            plot_feature_comparison_table(
                all_results, save_path=args.out_dir / "feature_comparison_table.png"
            )

            # Plot ROC curves if probabilities are available
            try:
                y_true_dict = {}
                y_proba_dict = {}
                for mode, results in all_results.items():
                    if "test" in results and "probabilities" in results["test"]:
                        y_true_dict[mode] = y_test
                        y_proba_dict[mode] = np.array(results["test"]["probabilities"])

                if y_true_dict and y_proba_dict:
                    from model_A.visualizations import plot_roc_curves

                    plot_roc_curves(
                        y_true_dict,
                        y_proba_dict,
                        save_path=args.out_dir / "roc_curves.png",
                    )
            except Exception as e:
                print(f"Note: Could not generate ROC curves: {e}")

        # Also save detailed results for each feature mode
        if args.out_dir:
            detailed_results_path = args.out_dir / "feature_comparison_detailed.json"
            write_json(detailed_results_path, all_results)
            print(f"\nDetailed comparison results saved to: {detailed_results_path}")

            # Also always update a single "Model A results" file so it doesn't look stale
            # when running comparisons.
            model_a_results_path = args.out_dir / "model_A_results.json"
            write_json(
                model_a_results_path,
                {
                    "run_type": "feature_comparison",
                    "results_by_mode": all_results,
                    "config": {
                        "model": model_config,
                        "features": feature_config,
                        "data": data_config,
                        "random_state": random_state,
                    },
                },
            )
            print(f"\nModel A summary saved to: {model_a_results_path}")

        # Augmentation comparison: Compare performance with and without augmentation
        # Use 'raw' features as baseline for augmentation comparison
        print("\n" + "=" * 80)
        print("Augmentation Comparison: Without vs With Data Augmentation")
        print("=" * 80)
        print(
            "Comparing performance with and without data augmentation using PCA features..."
        )
        print("=" * 80)

        # Get augmentation config
        aug_config = config.get("augmentation", {}) if args.config else {}
        aug_enabled_default = aug_config.get("enabled", False)

        # Extract 'pca' features for augmentation comparison
        extractor_raw = FeatureExtractor(
            mode="pca",
            pca_components=feature_config.get("pca_components", 64),
            standardize_features=feature_config.get("standardize", False),
        )

        # Extract features from original (non-augmented) data
        X_train_raw = extractor_raw.fit_transform(X_train)
        X_val_raw = extractor_raw.transform(X_val)
        X_test_raw = extractor_raw.transform(X_test)

        # Train WITHOUT augmentation
        print("\n" + "-" * 80)
        print("Training WITHOUT augmentation...")
        print("-" * 80)
        trainer_config_no_aug = {
            "model": model_config,
            **feature_config,
            **data_config,
            "mode": "pca",
            "augmentation": {"enabled": False},
            "random_state": random_state,
        }
        train_result_no_aug = train_model_a(
            X_train_raw,
            y_train,
            X_val_raw,
            y_val,
            X_test_raw,
            y_test,
            config=trainer_config_no_aug,
            feature_mode="pca",
        )
        results_no_aug = train_result_no_aug["results"]

        # Train WITH augmentation (apply augmentation to images BEFORE feature extraction)
        print("\n" + "-" * 80)
        print("Training WITH augmentation...")
        print("-" * 80)

        # Apply augmentation to training images
        aug_factor = aug_config.get("augmentation_factor", 1.0)
        aug_kwargs = {
            "rotation_range": aug_config.get("rotation_range", 5.0),
            "translation_range": aug_config.get("translation_range", 1.0),
            "flip_horizontal": aug_config.get("flip_horizontal", False),
            "gaussian_noise_std": aug_config.get("gaussian_noise_std", 0.0),
            "brightness_range": aug_config.get("brightness_range", 0.0),
            "blur_sigma": aug_config.get("blur_sigma", None),
        }

        print(f"Applying augmentation to training set...")
        print(f"Augmentation parameters: {aug_kwargs}")
        print(f"Original training samples: {len(X_train)}")

        X_train_aug, y_train_aug = augment_dataset(
            X_train, y_train, augmentation_factor=aug_factor, **aug_kwargs
        )
        print(f"Augmented training samples: {len(X_train_aug)}")

        # Combine original + augmented samples (augmentation should ADD data, not replace it)
        X_train_mix = np.concatenate([X_train, X_train_aug], axis=0)
        y_train_mix = np.concatenate([y_train, y_train_aug], axis=0)
        print(f"Total training samples after mixing: {len(X_train_mix)}")

        # Extract features from augmented data
        # Need to refit extractor on the mixed data
        extractor_aug = FeatureExtractor(
            mode="pca",
            pca_components=feature_config.get("pca_components", 64),
            standardize_features=feature_config.get("standardize", False),
        )
        X_train_aug_feat = extractor_aug.fit_transform(X_train_mix)
        # Use same extractor for val/test (no augmentation)
        X_val_aug_feat = extractor_aug.transform(X_val)
        X_test_aug_feat = extractor_aug.transform(X_test)

        trainer_config_with_aug = {
            "model": model_config,
            **feature_config,
            **data_config,
            "mode": "pca",
            "augmentation": {"enabled": True, **aug_config},
            "random_state": random_state,
        }
        train_result_with_aug = train_model_a(
            X_train_aug_feat,
            y_train_mix,
            X_val_aug_feat,
            y_val,
            X_test_aug_feat,
            y_test,
            config=trainer_config_with_aug,
            feature_mode="pca",
        )
        results_with_aug = train_result_with_aug["results"]

        # Compare augmentation results
        aug_comparison_output = (
            args.out_dir / "augmentation_comparison_results.csv"
            if args.out_dir
            else Path("Datasets/augmentation_comparison_results.csv")
        )
        compare_augmentation(
            results_no_aug, results_with_aug, output_path=aug_comparison_output
        )

        # Generate augmentation comparison visualizations
        if args.out_dir:
            from model_A.visualizations import (
                plot_augmentation_comparison,
                plot_augmentation_confusion_matrices,
                plot_augmentation_roc_curves,
            )

            print("\nGenerating augmentation comparison visualizations...")

            # Plot metrics comparison
            plot_augmentation_comparison(
                results_no_aug,
                results_with_aug,
                save_path=args.out_dir / "augmentation_comparison.png",
            )

            # Plot confusion matrices
            plot_augmentation_confusion_matrices(
                results_no_aug,
                results_with_aug,
                save_path=args.out_dir / "augmentation_confusion_matrices.png",
            )

            # Plot ROC curves
            try:
                if (
                    "test" in results_no_aug
                    and "probabilities" in results_no_aug["test"]
                    and "test" in results_with_aug
                    and "probabilities" in results_with_aug["test"]
                ):
                    plot_augmentation_roc_curves(
                        y_test,
                        np.array(results_no_aug["test"]["probabilities"]),
                        np.array(results_with_aug["test"]["probabilities"]),
                        save_path=args.out_dir / "augmentation_roc_curves.png",
                    )
            except Exception as e:
                print(f"Note: Could not generate augmentation ROC curves: {e}")

        # Save detailed augmentation comparison results
        if args.out_dir:
            aug_detailed_path = args.out_dir / "augmentation_comparison_detailed.json"
            aug_comparison_data = {
                "without_augmentation": results_no_aug,
                "with_augmentation": results_with_aug,
            }
            write_json(aug_detailed_path, aug_comparison_data)
            print(
                f"\nDetailed augmentation comparison results saved to: {aug_detailed_path}"
            )

    else:
        # Single feature mode
        print(f"\nExtracting {feature_mode} features...")
        extractor = FeatureExtractor(
            mode=feature_mode,
            pca_components=feature_config.get("pca_components", 64),
            standardize_features=feature_config.get("standardize", False),
        )

        X_train_feat = extractor.fit_transform(X_train)
        X_val_feat = extractor.transform(X_val)
        X_test_feat = extractor.transform(X_test)

        # Combine configs for trainer (model config should be nested)
        # Update mode in config to match the actual feature mode being used
        trainer_config = {
            "model": model_config,
            **feature_config,
            **data_config,
            "mode": feature_mode,  # Ensure config reflects the actual feature mode
            "random_state": random_state,
        }
        train_result = train_model_a(
            X_train_feat,
            y_train,
            X_val_feat,
            y_val,
            X_test_feat,
            y_test,
            config=trainer_config,
            feature_mode=feature_mode,
        )

        # Save results
        if args.out_dir:
            results_path = args.out_dir / "model_A_results.json"
            write_json(results_path, train_result["results"])
            print(f"\nResults saved to: {results_path}")


def run_model_b(args):
    """Run Model B (deep learning) training and evaluation."""
    print("=" * 80)
    print("Model B: Deep Learning (ResNet-style)")
    print("=" * 80)

    # Load configuration
    if args.config:
        config = load_config(args.config)
        model_config = config.get("model", {})
        training_config = config.get("training", {})
        data_config = config.get("data", {})
        aug_config = config.get("augmentation", {})
        device = config.get("device", "cpu")
        random_state = config.get("random_state", 42)
    else:
        # Default configuration
        model_config = {"base_channels": 16, "num_blocks": 2, "dropout_rate": 0.5}
        training_config = {
            "num_epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "early_stopping_patience": 10,
        }
        data_config = {
            "train_fraction": args.train_fraction,
            "val_fraction": args.val_fraction,
            "normalize": True,
        }
        aug_config = {"enabled": True}
        device = "cuda" if torch.cuda.is_available() else "cpu"
        random_state = 42

    set_random_seeds(random_state)

    # Optional CPU performance knobs (safe defaults for Windows)
    try:
        t_threads = training_config.get("torch_num_threads", None)
        if t_threads is not None:
            torch.set_num_threads(int(t_threads))
        t_interop = training_config.get("torch_num_interop_threads", None)
        if t_interop is not None:
            torch.set_num_interop_threads(int(t_interop))
    except Exception:
        pass

    # Set device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    # Load dataset
    print("\nLoading BreastMNIST dataset...")
    # For Model B (deep learning), use minmax normalization to [0, 1] instead of z-score
    # This is more standard for deep learning models (z-score can cause issues with PIL Image conversion)
    dataset = BreastMNISTDataset(
        data_dir=args.data_dir,
        normalize=data_config.get("normalize", True),
        normalize_mode="minmax",  # Use minmax for deep learning
        random_state=random_state,
    )

    # Summarize dataset
    dataset.summarize()

    # Get data splits
    train_fraction = data_config.get("train_fraction", args.train_fraction)
    val_fraction = data_config.get("val_fraction", args.val_fraction)

    X_train, y_train, X_val, y_val, X_test, y_test = dataset.get_all_splits(
        train_fraction=train_fraction, val_fraction=val_fraction
    )

    # Create output directory if it doesn't exist
    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    # Model B visualizations (data)
    if args.out_dir:
        from model_B.visualizations import plot_sample_images, plot_class_distribution

        plot_sample_images(
            X_train, y_train, n_samples=9, save_path=args.out_dir / "sample_images.png"
        )
        plot_class_distribution(
            y_train, y_val, y_test, save_path=args.out_dir / "class_distribution.png"
        )

    # Visualize samples if requested
    if args.show_examples:
        print("\nVisualizing sample images...")
        visualize_samples(
            X_train,
            y_train,
            n_samples=5,
            save_path=args.out_dir / "sample_images.png" if args.out_dir else None,
        )

    from model_B.evaluator import save_history, compare_augmentation_b

    def _build_model() -> ModelBNet:
        m = ModelBNet(
            num_classes=2,
            base_channels=model_config.get("base_channels", 16),
            num_blocks=model_config.get("num_blocks", 2),
            dropout_rate=model_config.get("dropout_rate", 0.5),
        )
        return m.to(device)

    def _train_eval_once(augment_enabled: bool, tag: str) -> Dict[str, Any]:
        augmentation_kwargs = {k: v for k, v in aug_config.items() if k != "enabled"}
        num_workers = int(training_config.get("num_workers", 0))
        train_loader, val_loader, test_loader = get_breastmnist_dataloaders(
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            batch_size=training_config.get("batch_size", 32),
            augment=augment_enabled,
            augmentation_kwargs=augmentation_kwargs,
            num_workers=num_workers,
        )

        model = _build_model()
        print(f"\nModel Parameters: {model.get_num_parameters():,}")

        checkpoint_dir = (args.out_dir / "checkpoints" / tag) if args.out_dir else None
        train_result = train_model_b(
            model,
            train_loader,
            val_loader,
            config={**training_config, "random_state": random_state},
            device=device,
            checkpoint_dir=str(checkpoint_dir) if checkpoint_dir else None,
        )

        # Evaluate on train/val/test (Model-A-like)
        train_results = evaluate_model_b(
            model, train_loader, device=device, split="train", print_results=False
        )
        val_results = evaluate_model_b(
            model, val_loader, device=device, split="val", print_results=False
        )
        test_results = evaluate_model_b(
            model, test_loader, device=device, split="test", print_results=True
        )

        # Save history + training curves
        if args.out_dir:
            save_history(
                train_result["history"],
                str(args.out_dir / f"history_{tag}.csv"),
                format="csv",
            )
            save_history(
                train_result["history"],
                str(args.out_dir / f"history_{tag}.json"),
                format="json",
            )
            from model_B.visualizations import plot_training_curves

            plot_training_curves(
                train_result["history"],
                save_path=args.out_dir / f"training_curves_{tag}.png",
            )

        return {
            "tag": tag,
            "augment_enabled": bool(augment_enabled),
            "train": train_results,
            "val": val_results,
            "test": test_results,
            "training_history": train_result["history"],
            "best_val_acc": train_result["best_val_acc"],
        }

    # Determine whether to run ablation (OFF vs ON). CLI overrides config.
    run_ablation = bool(training_config.get("run_ablation", True))
    if hasattr(args, "model_b_ablation") and args.model_b_ablation is not None:
        run_ablation = args.model_b_ablation == "on"

    if not run_ablation:
        # Single run (faster on CPU)
        augment_enabled = bool(aug_config.get("enabled", True))
        result = _train_eval_once(augment_enabled, "single")

        if args.out_dir:
            from model_B.visualizations import plot_confusion_matrix, plot_roc_curve

            y_true = np.array(y_test)
            y_pred = np.array(result["test"]["predictions"])
            y_proba = np.array(result["test"]["probabilities"])
            plot_confusion_matrix(
                y_true,
                y_pred,
                title="Model B Confusion Matrix (Test)",
                save_path=args.out_dir / "confusion_matrices.png",
            )
            plot_roc_curve(
                y_true,
                y_proba[:, 1],
                title="Model B ROC Curve (Test)",
                save_path=args.out_dir / "roc_curves.png",
            )

            write_json(
                args.out_dir / "model_B_results.json",
                {
                    "run_type": "single_run",
                    "run": result,
                    "config": {
                        "model": model_config,
                        "training": training_config,
                        "data": data_config,
                        "augmentation": aug_config,
                        "device": device,
                        "random_state": random_state,
                    },
                },
            )
        return

    # Augmentation ablation: OFF vs ON (default)
    results_without = _train_eval_once(False, "without_aug")
    results_with = _train_eval_once(True, "with_aug")

    # Save detailed comparison
    if args.out_dir:
        write_json(
            args.out_dir / "augmentation_comparison_detailed.json",
            {
                "without_augmentation": results_without,
                "with_augmentation": results_with,
            },
        )
        compare_augmentation_b(
            results_without_aug=results_without,
            results_with_aug=results_with,
            output_path=str(args.out_dir / "augmentation_comparison_results.csv"),
        )

        # Plots mirroring Model A
        from model_B.visualizations import (
            plot_augmentation_comparison,
            plot_side_by_side_confusion_matrices,
            plot_augmentation_roc_curves,
            plot_confusion_matrix,
            plot_roc_curve,
        )

        plot_augmentation_comparison(
            results_without["test"]["metrics"],
            results_with["test"]["metrics"],
            save_path=args.out_dir / "augmentation_comparison.png",
        )
        y_true = np.array(y_test)
        y_pred_without = np.array(results_without["test"]["predictions"])
        y_pred_with = np.array(results_with["test"]["predictions"])
        plot_side_by_side_confusion_matrices(
            y_true,
            y_pred_without,
            y_pred_with,
            save_path=args.out_dir / "augmentation_confusion_matrices.png",
        )
        # ROC curves
        y_proba_without = np.array(results_without["test"]["probabilities"])
        y_proba_with = np.array(results_with["test"]["probabilities"])
        plot_augmentation_roc_curves(
            y_true,
            y_proba_without,
            y_proba_with,
            save_path=args.out_dir / "augmentation_roc_curves.png",
        )

        # Also write single-run style plots for the WITH-augmentation result (parity with Model A)
        plot_confusion_matrix(
            y_true,
            np.array(results_with["test"]["predictions"]),
            title="Model B Confusion Matrix (Test, With Aug)",
            save_path=args.out_dir / "confusion_matrices.png",
        )
        plot_roc_curve(
            y_true,
            y_proba_with[:, 1],
            title="Model B ROC Curve (Test, With Aug)",
            save_path=args.out_dir / "roc_curves.png",
        )

        # Also save a single consolidated results file
        write_json(
            args.out_dir / "model_B_results.json",
            {
                "run_type": "augmentation_ablation",
                "without_augmentation": results_without,
                "with_augmentation": results_with,
                "config": {
                    "model": model_config,
                    "training": training_config,
                    "data": data_config,
                    "augmentation": aug_config,
                    "device": device,
                    "random_state": random_state,
                },
            },
        )


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="AMLS Assignment - BreastMNIST Classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["A", "B", "both"],
        default="both",
        help="Model to train (A for classical ML, B for deep learning, both for both models)",
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        default="train",
        help="Mode: train or test",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file (can specify separate configs with --config_A and --config_B)",
    )

    parser.add_argument(
        "--config_A",
        type=str,
        default=None,
        help="Path to configuration YAML file for Model A",
    )

    parser.add_argument(
        "--config_B",
        type=str,
        default=None,
        help="Path to configuration YAML file for Model B",
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="Datasets",
        help="Path to directory containing BreastMNIST.npz file",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="results",
        help="Output directory for results and checkpoints (default: results)",
    )

    parser.add_argument(
        "--train_fraction",
        type=float,
        default=1.0,
        help="Fraction of training set to use (0 < fraction <= 1.0)",
    )

    parser.add_argument(
        "--val_fraction",
        type=float,
        default=1.0,
        help="Fraction of validation set to use (0 < fraction <= 1.0)",
    )

    parser.add_argument(
        "--feature_mode",
        type=str,
        choices=["raw", "pca", "hog"],
        default=None,
        help="Feature mode for Model A (if not specified, compares all modes: raw, PCA, HOG)",
    )

    parser.add_argument(
        "--compare_all_features",
        action="store_true",
        help="Compare all feature modes (raw, PCA, HOG) - default behavior when no feature_mode specified",
    )

    parser.add_argument(
        "--single_feature_mode",
        action="store_true",
        help="Use single feature mode instead of comparing all (overrides default comparison behavior)",
    )

    parser.add_argument(
        "--show_examples",
        action="store_true",
        help="Visualize sample images from the dataset",
    )

    parser.add_argument(
        "--model_b_ablation",
        type=str,
        choices=["on", "off"],
        default=None,
        help="Model B only: run augmentation ablation (on=OFF vs ON, off=single run). Overrides config if set.",
    )

    args = parser.parse_args()

    # Convert out_dir to Path
    if args.out_dir:
        args.out_dir = Path(args.out_dir)

    # Determine which models to run
    models_to_run = []
    if args.model == "A":
        models_to_run = ["A"]
    elif args.model == "B":
        models_to_run = ["B"]
    else:  # 'both' or default
        models_to_run = ["A", "B"]

    # Run the selected models
    for model in models_to_run:
        # Create model-specific args
        model_args = argparse.Namespace(**vars(args))

        # Set model-specific config if provided
        # Default to model-specific config files if no config specified
        if model == "A":
            if args.config_A:
                model_args.config = args.config_A
            elif not args.config:
                # Default to model_A/config.yaml
                default_config = Path("model_A/config.yaml")
                if default_config.exists():
                    model_args.config = str(default_config)
            elif args.config:
                model_args.config = args.config
        elif model == "B":
            if args.config_B:
                model_args.config = args.config_B
            elif not args.config:
                # Default to model_B/config.yaml
                default_config = Path("model_B/config.yaml")
                if default_config.exists():
                    model_args.config = str(default_config)
            elif args.config:
                model_args.config = args.config

        # Set model-specific output directories
        if model == "A":
            model_args.out_dir = (
                args.out_dir / "model_A" if args.out_dir else Path("results/model_A")
            )
        elif model == "B":
            model_args.out_dir = (
                args.out_dir / "model_B" if args.out_dir else Path("results/model_B")
            )

        # Run the model
        if model == "A":
            print("\n" + "=" * 80)
            print("RUNNING MODEL A")
            print("=" * 80)
            run_model_a(model_args)
        elif model == "B":
            print("\n" + "=" * 80)
            print("RUNNING MODEL B")
            print("=" * 80)
            run_model_b(model_args)

        # Add separator between models if running both
        if len(models_to_run) > 1 and model != models_to_run[-1]:
            print("\n" + "=" * 80)
            print("=" * 80)
            print("\n")

    # Always refresh comparison_summary.md in the base out_dir (default: results/)
    if args.out_dir:
        try:
            out_path = write_comparison_summary(args.out_dir)
            if out_path:
                print(f"\nComparison summary updated: {out_path}")
        except Exception as e:
            print(f"\nNote: Could not update comparison_summary.md: {e}")

        # Also generate cross-model comparison visualizations
        try:
            paths = _generate_model_comparison_visualizations(
                args.out_dir,
                data_dir=Path(args.data_dir),
                random_state=42,
            )
            if "error" in paths:
                print(
                    f"\nNote: Could not generate model comparison plots: {paths['error']}"
                )
            else:
                print("\nModel comparison plots written:")
                for k, v in paths.items():
                    if v:
                        print(f"  - {k}: {v}")
        except Exception as e:
            print(f"\nNote: Could not generate model comparison plots: {e}")


if __name__ == "__main__":
    main()
