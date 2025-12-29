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
    roc_curve
)
import json
import csv
from pathlib import Path


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
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
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, average='binary', zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, average='binary', zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, average='binary', zero_division=0))
    }
    
    # Compute AUC if probabilities are available
    if y_proba is not None:
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba[:, 1]))
        except ValueError:
            metrics['roc_auc'] = 0.0
    
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
    return_proba: bool = False
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
        'metrics': metrics,
        'predictions': y_pred.tolist(),
        'confusion_matrix': cm.tolist()
    }
    
    if return_proba:
        results['probabilities'] = y_proba.tolist()
    
    return results


def save_results(
    results: Dict[str, Any],
    output_path: str,
    format: str = "json"
):
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
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    elif format == "csv":
        # Flatten results for CSV
        rows = []
        for split in ['train', 'val', 'test']:
            if split in results:
                row = {'split': split}
                row.update(results[split].get('metrics', {}))
                rows.append(row)
        
        if rows:
            fieldnames = ['split'] + list(rows[0].keys())[1:]
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'json' or 'csv'.")


def compare_feature_modes(
    results_dict: Dict[str, Dict[str, Any]],
    output_path: Optional[str] = None
):
    """
    Compare results across different feature modes.
    
    Args:
        results_dict: Dictionary mapping feature mode to results
        output_path: Optional path to save comparison CSV
    """
    print("\n" + "=" * 80)
    print("Feature Mode Comparison")
    print("=" * 80)
    
    # Prepare comparison table
    rows = []
    for mode, results in results_dict.items():
        for split in ['val', 'test']:
            if split in results and 'metrics' in results[split]:
                row = {
                    'feature_mode': mode,
                    'split': split
                }
                row.update(results[split]['metrics'])
                rows.append(row)
    
    # Print table
    if rows:
        # Header
        headers = ['Feature Mode', 'Split', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
        if 'roc_auc' in rows[0]:
            headers.append('ROC-AUC')
        
        print("\n" + " | ".join(f"{h:>12}" for h in headers))
        print("-" * 80)
        
        for row in rows:
            values = [
                row['feature_mode'],
                row['split'],
                f"{row['accuracy']:.4f}",
                f"{row['precision']:.4f}",
                f"{row['recall']:.4f}",
                f"{row['f1_score']:.4f}"
            ]
            if 'roc_auc' in row:
                values.append(f"{row['roc_auc']:.4f}")
            print(" | ".join(f"{v:>12}" for v in values))
        
        # Save to CSV if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            fieldnames = ['feature_mode', 'split'] + [k for k in rows[0].keys() if k not in ['feature_mode', 'split']]
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"\nComparison saved to: {output_path}")
    
    print("=" * 80)

