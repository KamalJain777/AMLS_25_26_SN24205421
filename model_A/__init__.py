"""Classical machine learning models for Category A (Model A)."""

from .dataset import BreastMNISTDataset
from .preprocessing import Preprocessor
from .features import FeatureExtractor
from .classifier import ModelAClassifier
from .trainer import train_model_a
from .evaluator import evaluate_model_a, compare_feature_modes, compare_augmentation
from .visualizations import (
    plot_sample_images,
    plot_class_distribution,
    plot_confusion_matrices,
    plot_metrics_comparison,
    plot_feature_comparison_table,
    plot_augmentation_comparison,
    plot_augmentation_confusion_matrices,
    plot_augmentation_roc_curves,
)

__all__ = [
    "BreastMNISTDataset",
    "Preprocessor",
    "FeatureExtractor",
    "ModelAClassifier",
    "train_model_a",
    "evaluate_model_a",
    "compare_feature_modes",
    "compare_augmentation",
    "plot_sample_images",
    "plot_class_distribution",
    "plot_confusion_matrices",
    "plot_metrics_comparison",
    "plot_feature_comparison_table",
    "plot_augmentation_comparison",
    "plot_augmentation_confusion_matrices",
    "plot_augmentation_roc_curves",
]
