"""Preprocessing utilities for BreastMNIST images."""

import numpy as np
from typing import Tuple


class Preprocessor:
    """
    Image preprocessing pipeline for BreastMNIST.

    Handles:
    - Reshaping to fixed size (28, 28)
    - Intensity scaling [0, 255] -> [0, 1]
    - Z-score normalization using training set statistics
    """

    def __init__(self):
        """Initialize preprocessor."""
        self.mean = None
        self.std = None
        self.fitted = False

    def fit(self, X_train: np.ndarray):
        """
        Compute normalization statistics from training set.

        Args:
            X_train: Training images of shape (N, H, W) or (N, 1, H, W)
        """
        # Ensure images are in [0, 1] range first
        X_train = X_train.astype(np.float32)
        if X_train.max() > 1.0:
            X_train = X_train / 255.0

        # Compute mean and std across all pixels in training set
        self.mean = np.mean(X_train)
        self.std = np.std(X_train)
        self.fitted = True

    def transform(self, X: np.ndarray, apply_normalization: bool = True) -> np.ndarray:
        """
        Apply preprocessing transformations.

        Args:
            X: Images of shape (N, H, W) or (N, 1, H, W)
            apply_normalization: Whether to apply z-score normalization

        Returns:
            Preprocessed images of shape (N, H, W)
        """
        X = X.copy().astype(np.float32)

        # Reshape: ensure (N, H, W) format
        if len(X.shape) == 4 and X.shape[1] == 1:
            X = X.squeeze(1)

        # Intensity scaling: [0, 255] -> [0, 1]
        if X.max() > 1.0:
            X = X / 255.0

        # Z-score normalization
        if apply_normalization and self.fitted:
            X = (X - self.mean) / self.std

        return X

    def fit_transform(
        self, X_train: np.ndarray, apply_normalization: bool = True
    ) -> np.ndarray:
        """
        Fit on training data and transform.

        Args:
            X_train: Training images
            apply_normalization: Whether to apply z-score normalization

        Returns:
            Preprocessed training images
        """
        self.fit(X_train)
        return self.transform(X_train, apply_normalization=apply_normalization)


def preprocess_images(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray, normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess train, validation, and test images.

    Computes normalization statistics from training set only,
    then applies them to all splits.

    Args:
        X_train: Training images (N, H, W)
        X_val: Validation images (M, H, W)
        X_test: Test images (K, H, W)
        normalize: Whether to apply z-score normalization

    Returns:
        Preprocessed X_train, X_val, X_test
    """
    preprocessor = Preprocessor()

    if normalize:
        X_train = preprocessor.fit_transform(X_train, apply_normalization=True)
        X_val = preprocessor.transform(X_val, apply_normalization=True)
        X_test = preprocessor.transform(X_test, apply_normalization=True)
    else:
        # Just scale to [0, 1]
        X_train = preprocessor.fit_transform(X_train, apply_normalization=False)
        X_val = preprocessor.transform(X_val, apply_normalization=False)
        X_test = preprocessor.transform(X_test, apply_normalization=False)

    return X_train, X_val, X_test
