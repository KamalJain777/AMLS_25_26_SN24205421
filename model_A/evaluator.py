"""Evaluation utilities for Model A."""

import numpy as np
from typing import Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import json
import csv
from pathlib import Path


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (for AUC calculation)

    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, average="binary", zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, average="binary", zero_division=0)
        ),
        "f1_score": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
    }

    # Compute AUC if probabilities are available
    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
        except ValueError:
            metrics["roc_auc"] = 0.0

    return metrics


def print_metrics(metrics: Dict[str, float], split: str = "test"):
    """
    Print metrics in a formatted table.

    Args:
        metrics: Dictionary of metrics
        split: Name of the data split
    """
    print(f"\n{split.upper()} Set Metrics:")
    print("-" * 40)
    for key, value in metrics.items():
        print(f"{key.replace('_', ' ').title()}: {value:.4f}")
    print("-" * 40)


def evaluate_model_a(
    model,
    X: np.ndarray,
    y: np.ndarray,
    split: str = "test",
    print_results: bool = True,
    return_proba: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate Model A on a dataset.

    Args:
        model: Trained ModelAClassifier instance
        X: Feature matrix
        y: True labels
        split: Name of the split ('train', 'val', 'test')
        print_results: Whether to print metrics
        return_proba: Whether to return probabilities

    Returns:
        Dictionary containing metrics, predictions, and optionally probabilities
    """
    # Make predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    # Compute metrics
    metrics = compute_metrics(y, y_pred, y_proba)

    # Compute confusion matrix
    cm = confusion_matrix(y, y_pred)

    # Print results
    if print_results:
        print_metrics(metrics, split=split)
        print(f"\nConfusion Matrix ({split}):")
        print(cm)

    results = {
        "metrics": metrics,
        "predictions": y_pred.tolist(),
        "confusion_matrix": cm.tolist(),
    }

    if return_proba:
        results["probabilities"] = y_proba.tolist()

    return results


def save_results(results: Dict[str, Any], output_path: str, format: str = "json"):
    """
    Save evaluation results to file.

    Args:
        results: Dictionary of results to save
        output_path: Path to output file
        format: File format ('json' or 'csv')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
    elif format == "csv":
        # Flatten results for CSV
        rows = []
        for split in ["train", "val", "test"]:
            if split in results:
                row = {"split": split}
                row.update(results[split].get("metrics", {}))
                rows.append(row)

        if rows:
            fieldnames = ["split"] + list(rows[0].keys())[1:]
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'json' or 'csv'.")


def compare_feature_modes(
    results_dict: Dict[str, Dict[str, Any]], output_path: Optional[str] = None
):
    """
    Compare results across different feature modes (raw vs processed features).

    This function compares performance between:
    - Raw features: Baseline flattened pixel features
    - Processed features: PCA and HOG feature representations

    Args:
        results_dict: Dictionary mapping feature mode to results
        output_path: Optional path to save comparison CSV
    """
    print("\n" + "=" * 80)
    print("Feature Mode Comparison: Raw vs Processed Features")
    print("=" * 80)
    print("\nThis comparison evaluates the performance difference between:")
    print("  - Raw features: Direct pixel values (baseline)")
    print("  - PCA features: Dimensionality-reduced representation")
    print("  - HOG features: Edge/texture-based representation")
    print("\n" + "=" * 80)

    # Prepare comparison table
    rows = []
    for mode, results in results_dict.items():
        for split in ["val", "test"]:
            if split in results and "metrics" in results[split]:
                row = {"feature_mode": mode, "split": split}
                row.update(results[split]["metrics"])
                rows.append(row)

    # Print comprehensive comparison table
    if rows:
        # Header
        headers = [
            "Feature Mode",
            "Split",
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
        ]
        if "roc_auc" in rows[0]:
            headers.append("ROC-AUC")

        # Calculate column widths for better formatting
        col_width = 14
        table_width = len(headers) * col_width + (len(headers) - 1) * 3

        # Print comprehensive comparison table
        print("\n" + "=" * table_width)
        print("PERFORMANCE COMPARISON: RAW vs PROCESSED FEATURES".center(table_width))
        print("=" * table_width)

        # Group by split for better readability
        for split in ["val", "test"]:
            print(f"\n{split.upper()} SET RESULTS:")
            print("-" * table_width)

            # Print header
            header_row = " | ".join(f"{h:>{col_width}}" for h in headers)
            print(header_row)
            print("-" * table_width)

            split_rows = [r for r in rows if r["split"] == split]

            # Print each feature mode row
            for row in split_rows:
                # Format feature mode name (capitalize and add description)
                mode_name = row["feature_mode"].upper()
                if mode_name == "RAW":
                    mode_display = f"{mode_name} (baseline)"
                elif mode_name == "PCA":
                    mode_display = f"{mode_name} (dimensionality reduction)"
                elif mode_name == "HOG":
                    mode_display = f"{mode_name} (edge/texture)"
                else:
                    mode_display = mode_name

                values = [
                    mode_display,
                    split.upper(),
                    f"{row['accuracy']:.4f}",
                    f"{row['precision']:.4f}",
                    f"{row['recall']:.4f}",
                    f"{row['f1_score']:.4f}",
                ]
                if "roc_auc" in row:
                    values.append(f"{row['roc_auc']:.4f}")

                # Format values with proper alignment
                formatted_values = []
                for i, v in enumerate(values):
                    if i == 0:  # Feature mode - left align
                        formatted_values.append(f"{v:<{col_width}}")
                    else:  # Numbers - right align
                        formatted_values.append(f"{v:>{col_width}}")

                print(" | ".join(formatted_values))

            print("-" * table_width)

        # Find best performing feature mode for test set and create summary
        test_rows = [r for r in rows if r["split"] == "test"]
        if test_rows:
            best_accuracy = max(test_rows, key=lambda x: x["accuracy"])
            best_f1 = max(test_rows, key=lambda x: x["f1_score"])
            raw_results = [r for r in test_rows if r["feature_mode"] == "raw"]
            processed_results = [r for r in test_rows if r["feature_mode"] != "raw"]

            print("\n" + "=" * table_width)
            print("SUMMARY: RAW vs PROCESSED FEATURES COMPARISON".center(table_width))
            print("=" * table_width)
            print(
                f"  Best Accuracy on Test Set: {best_accuracy['feature_mode'].upper()} ({best_accuracy['accuracy']:.4f})"
            )
            print(
                f"  Best F1-Score on Test Set: {best_f1['feature_mode'].upper()} ({best_f1['f1_score']:.4f})"
            )

            if raw_results and processed_results:
                raw_acc = raw_results[0]["accuracy"]
                best_processed_acc = max([r["accuracy"] for r in processed_results])
                best_processed_mode = max(
                    processed_results, key=lambda x: x["accuracy"]
                )["feature_mode"]
                improvement = (
                    ((best_processed_acc - raw_acc) / raw_acc * 100)
                    if raw_acc > 0
                    else 0
                )

                print(f"\n  Raw features (baseline) accuracy:        {raw_acc:.4f}")
                print(
                    f"  Best processed features ({best_processed_mode.upper()}) accuracy: {best_processed_acc:.4f}"
                )
                # Use ASCII-only output for Windows consoles (avoid UnicodeEncodeError)
                if improvement > 0:
                    print(
                        f"  -> Improvement with processed features: +{improvement:.2f}%"
                    )
                elif improvement < 0:
                    print(f"  -> Change with processed features: {improvement:.2f}%")
                else:
                    print("  -> No difference between raw and processed features")

            # Create side-by-side comparison table
            print("\n" + "=" * table_width)
            print("DETAILED PERFORMANCE COMPARISON TABLE".center(table_width))
            print("=" * table_width)

            # Create comparison table with all metrics side by side
            comp_headers = [
                "Feature Mode",
                "Accuracy",
                "Precision",
                "Recall",
                "F1-Score",
            ]
            if "roc_auc" in rows[0]:
                comp_headers.append("ROC-AUC")

            comp_table_width = (
                len(comp_headers) * col_width + (len(comp_headers) - 1) * 3
            )
            print("-" * comp_table_width)
            print(" | ".join(f"{h:>{col_width}}" for h in comp_headers))
            print("-" * comp_table_width)

            # Print test set results for all feature modes
            for row in sorted(test_rows, key=lambda x: x["accuracy"], reverse=True):
                mode_name = row["feature_mode"].upper()
                if mode_name == "RAW":
                    mode_display = f"{mode_name} (baseline)"
                elif mode_name == "PCA":
                    mode_display = f"{mode_name} (PCA)"
                elif mode_name == "HOG":
                    mode_display = f"{mode_name} (HOG)"
                else:
                    mode_display = mode_name

                comp_values = [
                    f"{mode_display:<{col_width}}",
                    f"{row['accuracy']:.4f}",
                    f"{row['precision']:.4f}",
                    f"{row['recall']:.4f}",
                    f"{row['f1_score']:.4f}",
                ]
                if "roc_auc" in row:
                    comp_values.append(f"{row['roc_auc']:.4f}")

                formatted_comp = []
                for i, v in enumerate(comp_values):
                    if i == 0:  # Feature mode - left align
                        formatted_comp.append(f"{mode_display:<{col_width}}")
                    else:  # Numbers - right align
                        formatted_comp.append(f"{v:>{col_width}}")

                print(" | ".join(formatted_comp))

            print("-" * comp_table_width)
            print("=" * table_width)

        # Save to CSV if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            fieldnames = ["feature_mode", "split"] + [
                k for k in rows[0].keys() if k not in ["feature_mode", "split"]
            ]
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"\nComparison saved to: {output_path}")

    print("=" * 80)


def compare_augmentation(
    results_without_aug: Dict[str, Any],
    results_with_aug: Dict[str, Any],
    output_path: Optional[str] = None,
):
    """
    Compare results with and without data augmentation.

    Args:
        results_without_aug: Results dictionary from training without augmentation
        results_with_aug: Results dictionary from training with augmentation
        output_path: Optional path to save comparison CSV
    """
    print("\n" + "=" * 80)
    print("Augmentation Comparison: Without vs With Data Augmentation")
    print("=" * 80)
    print("\nThis comparison evaluates the performance difference between:")
    print("  - Without augmentation: Baseline training on original dataset")
    print("  - With augmentation: Training with augmented dataset")
    print("\n" + "=" * 80)

    # Prepare comparison table
    rows = []
    for split in ["val", "test"]:
        if split in results_without_aug and "metrics" in results_without_aug[split]:
            row = {"augmentation": "without", "split": split}
            row.update(results_without_aug[split]["metrics"])
            rows.append(row)

        if split in results_with_aug and "metrics" in results_with_aug[split]:
            row = {"augmentation": "with", "split": split}
            row.update(results_with_aug[split]["metrics"])
            rows.append(row)

    # Print comprehensive comparison table
    if rows:
        # Header
        headers = [
            "Augmentation",
            "Split",
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
        ]
        if "roc_auc" in rows[0]:
            headers.append("ROC-AUC")

        # Calculate column widths for better formatting
        col_width = 14
        table_width = len(headers) * col_width + (len(headers) - 1) * 3

        # Print comprehensive comparison table
        print("\n" + "=" * table_width)
        print(
            "PERFORMANCE COMPARISON: WITHOUT vs WITH AUGMENTATION".center(table_width)
        )
        print("=" * table_width)

        # Group by split for better readability
        for split in ["val", "test"]:
            print(f"\n{split.upper()} SET RESULTS:")
            print("-" * table_width)

            # Print header
            header_row = " | ".join(f"{h:>{col_width}}" for h in headers)
            print(header_row)
            print("-" * table_width)

            split_rows = [r for r in rows if r["split"] == split]

            # Print each augmentation setting row
            for row in split_rows:
                aug_name = row["augmentation"].upper()
                if aug_name == "WITHOUT":
                    aug_display = "Without (baseline)"
                else:
                    aug_display = "With augmentation"

                values = [
                    aug_display,
                    split.upper(),
                    f"{row['accuracy']:.4f}",
                    f"{row['precision']:.4f}",
                    f"{row['recall']:.4f}",
                    f"{row['f1_score']:.4f}",
                ]
                if "roc_auc" in row:
                    values.append(f"{row['roc_auc']:.4f}")

                # Format values with proper alignment
                formatted_values = []
                for i, v in enumerate(values):
                    if i == 0:  # Augmentation - left align
                        formatted_values.append(f"{v:<{col_width}}")
                    else:  # Numbers - right align
                        formatted_values.append(f"{v:>{col_width}}")

                print(" | ".join(formatted_values))

            print("-" * table_width)

        # Find best performing setting for test set and create summary
        test_rows = [r for r in rows if r["split"] == "test"]
        if len(test_rows) == 2:
            without_row = [r for r in test_rows if r["augmentation"] == "without"][0]
            with_row = [r for r in test_rows if r["augmentation"] == "with"][0]

            print("\n" + "=" * table_width)
            print("SUMMARY: AUGMENTATION COMPARISON".center(table_width))
            print("=" * table_width)

            without_acc = without_row["accuracy"]
            with_acc = with_row["accuracy"]
            improvement = (
                ((with_acc - without_acc) / without_acc * 100) if without_acc > 0 else 0
            )

            print(f"  Without augmentation (baseline) accuracy: {without_acc:.4f}")
            print(f"  With augmentation accuracy:                {with_acc:.4f}")
            # Use ASCII-only output for Windows consoles (avoid UnicodeEncodeError)
            if improvement > 0:
                print(f"  -> Improvement with augmentation: +{improvement:.2f}%")
            elif improvement < 0:
                print(f"  -> Change with augmentation: {improvement:.2f}%")
            else:
                print("  -> No difference with augmentation")

            # Detailed comparison table
            print("\n" + "=" * table_width)
            print("DETAILED PERFORMANCE COMPARISON TABLE".center(table_width))
            print("=" * table_width)

            comp_headers = [
                "Augmentation",
                "Accuracy",
                "Precision",
                "Recall",
                "F1-Score",
            ]
            if "roc_auc" in rows[0]:
                comp_headers.append("ROC-AUC")

            comp_table_width = (
                len(comp_headers) * col_width + (len(comp_headers) - 1) * 3
            )
            print("-" * comp_table_width)
            print(" | ".join(f"{h:>{col_width}}" for h in comp_headers))
            print("-" * comp_table_width)

            # Print test set results
            for row in [without_row, with_row]:
                aug_name = row["augmentation"].upper()
                if aug_name == "WITHOUT":
                    aug_display = "Without (baseline)"
                else:
                    aug_display = "With augmentation"

                comp_values = [
                    f"{aug_display:<{col_width}}",
                    f"{row['accuracy']:.4f}",
                    f"{row['precision']:.4f}",
                    f"{row['recall']:.4f}",
                    f"{row['f1_score']:.4f}",
                ]
                if "roc_auc" in row:
                    comp_values.append(f"{row['roc_auc']:.4f}")

                formatted_comp = []
                for i, v in enumerate(comp_values):
                    if i == 0:  # Augmentation - left align
                        formatted_comp.append(f"{aug_display:<{col_width}}")
                    else:  # Numbers - right align
                        formatted_comp.append(f"{v:>{col_width}}")

                print(" | ".join(formatted_comp))

            print("-" * comp_table_width)
            print("=" * table_width)

        # Save to CSV if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            fieldnames = ["augmentation", "split"] + [
                k for k in rows[0].keys() if k not in ["augmentation", "split"]
            ]
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"\nComparison saved to: {output_path}")

    print("=" * 80)
