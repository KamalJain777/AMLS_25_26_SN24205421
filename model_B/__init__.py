"""Deep learning models for Category B (Model B)."""

from .dataset import get_breastmnist_dataloaders
from .model import ModelBNet
from .trainer import train_model_b
from .evaluator import evaluate_model_b
from .visualizations import (
    plot_sample_images,
    plot_class_distribution,
    plot_training_curves,
    plot_confusion_matrix,
    plot_side_by_side_confusion_matrices,
    plot_roc_curve,
    plot_augmentation_roc_curves,
    plot_augmentation_comparison,
)

__all__ = [
    'get_breastmnist_dataloaders',
    'ModelBNet',
    'train_model_b',
    'evaluate_model_b',
    'plot_sample_images',
    'plot_class_distribution',
    'plot_training_curves',
    'plot_confusion_matrix',
    'plot_side_by_side_confusion_matrices',
    'plot_roc_curve',
    'plot_augmentation_roc_curves',
    'plot_augmentation_comparison',
]
