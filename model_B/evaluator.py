"""Evaluation utilities for Model B."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, Optional, List
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
import json
from pathlib import Path
import csv


def evaluate_model_b(
    model: nn.Module,
    test_loader: DataLoader,
    device: str = "cpu",
    split: str = "test",
    print_results: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate Model B on a dataset.

    Args:
        model: Trained model
        test_loader: Data loader for test set
        device: Device ('cpu' or 'cuda')
        split: Name of the split ('train', 'val', 'test')
        print_results: Whether to print metrics

    Returns:
        Dictionary containing metrics, predictions, and probabilities
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)

            # Get predictions
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_proba = np.array(all_probs)

    # Compute metrics
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

    # Compute AUC
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
    except ValueError:
        metrics["roc_auc"] = 0.0

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Print results
    if print_results:
        print(f"\n{split.upper()} Set Metrics:")
        print("-" * 40)
        for key, value in metrics.items():
            print(f"{key.replace('_', ' ').title()}: {value:.4f}")
        print("-" * 40)
        print(f"\nConfusion Matrix ({split}):")
        print(cm)

    return {
        "metrics": metrics,
        "predictions": y_pred.tolist(),
        "probabilities": y_proba.tolist(),
        "confusion_matrix": cm.tolist(),
    }


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
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {
                    key: convert_to_serializable(value) for key, value in obj.items()
                }
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj

        serializable_results = convert_to_serializable(results)
        with open(output_path, "w") as f:
            json.dump(serializable_results, f, indent=2)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'json'.")


def save_history(
    history: Dict[str, Any], output_path: str, format: str = "csv"
) -> None:
    """
    Save training history to disk.

    Args:
        history: Dict with keys like train_loss, train_acc, val_loss, val_acc
        output_path: Output file path
        format: 'csv' or 'json'
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        with open(output_path, "w") as f:
            json.dump(history, f, indent=2)
        return

    if format != "csv":
        raise ValueError(f"Unknown format: {format}. Use 'csv' or 'json'.")

    # Normalize lengths
    keys = [
        k for k in ["train_loss", "train_acc", "val_loss", "val_acc"] if k in history
    ]
    n = max((len(history.get(k, [])) for k in keys), default=0)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch"] + keys)
        writer.writeheader()
        for i in range(n):
            row = {"epoch": i + 1}
            for k in keys:
                vals = history.get(k, [])
                row[k] = vals[i] if i < len(vals) else None
            writer.writerow(row)


def compare_augmentation_b(
    results_without_aug: Dict[str, Any],
    results_with_aug: Dict[str, Any],
    output_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Create a simple comparison table between augmentation OFF vs ON.
    Returns rows and optionally writes CSV.
    """
    rows: List[Dict[str, Any]] = []
    for split in ["train", "val", "test"]:
        if split in results_without_aug and "metrics" in results_without_aug[split]:
            row = {"augmentation": "without", "split": split}
            row.update(results_without_aug[split]["metrics"])
            rows.append(row)
        if split in results_with_aug and "metrics" in results_with_aug[split]:
            row = {"augmentation": "with", "split": split}
            row.update(results_with_aug[split]["metrics"])
            rows.append(row)

    if output_path and rows:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["augmentation", "split"] + [
            k for k in rows[0].keys() if k not in ["augmentation", "split"]
        ]
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    return rows
