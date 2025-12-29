"""Classical machine learning models for Category A (Model A)."""

from .dataset import BreastMNISTDataset
from .preprocessing import Preprocessor
from .features import FeatureExtractor
from .classifier import ModelAClassifier
from .trainer import train_model_a
from .evaluator import evaluate_model_a, compare_feature_modes

__all__ = [
    'BreastMNISTDataset',
    'Preprocessor',
    'FeatureExtractor',
    'ModelAClassifier',
    'train_model_a',
    'evaluate_model_a',
    'compare_feature_modes',
]

