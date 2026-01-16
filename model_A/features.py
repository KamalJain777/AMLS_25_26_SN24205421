"""Feature extraction pipeline for classical ML models."""

import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    from skimage.feature import hog
except ImportError:
    hog = None


class FeatureExtractor:
    """
    Feature extraction for classical ML models.

    Supports multiple feature modes:
    - 'raw': Flattened pixel values
    - 'pca': PCA-reduced features
    - 'hog': Histogram of Oriented Gradients
    """

    def __init__(
        self,
        mode: str = "raw",
        pca_components: int = 64,
        hog_params: Optional[Dict] = None,
        standardize_features: bool = False,
    ):
        """
        Initialize feature extractor.

        Args:
            mode: Feature mode ('raw', 'pca', 'hog')
            pca_components: Number of PCA components (for 'pca' mode)
            hog_params: Dictionary of HOG parameters (for 'hog' mode)
            standardize_features: Whether to apply StandardScaler to features
        """
        self.mode = mode
        self.pca_components = pca_components
        self.standardize_features = standardize_features

        # Default HOG parameters for 28x28 images
        if hog_params is None:
            hog_params = {
                "orientations": 9,
                "pixels_per_cell": (4, 4),
                "cells_per_block": (2, 2),
                "block_norm": "L2-Hys",
                "visualize": False,
                "feature_vector": True,
            }
        self.hog_params = hog_params

        # Fitted models
        self.pca = None
        self.scaler = None
        self.fitted = False

        if mode not in ["raw", "pca", "hog"]:
            raise ValueError(f"Unknown mode: {mode}. Use 'raw', 'pca', or 'hog'.")

        if mode == "hog" and hog is None:
            raise ImportError(
                "skimage is required for HOG features. "
                "Install with: pip install scikit-image"
            )

    def fit(self, X_train: np.ndarray):
        """
        Fit feature extractor on training data.

        Args:
            X_train: Training images of shape (N, H, W) or (N, 1, H, W)
        """
        # Flatten images for raw/PCA modes
        if self.mode in ["raw", "pca"]:
            X_train_flat = X_train.reshape(X_train.shape[0], -1)

            # Fit PCA if needed
            if self.mode == "pca":
                self.pca = PCA(n_components=self.pca_components, random_state=42)
                X_train_features = self.pca.fit_transform(X_train_flat)
            else:
                X_train_features = X_train_flat

            # Fit StandardScaler if needed
            if self.standardize_features:
                self.scaler = StandardScaler()
                self.scaler.fit(X_train_features)
        else:
            # For HOG, optionally fit a scaler on training HOG features (best practice:
            # fit on train only, apply to val/test with the same scaler).
            if self.standardize_features:
                X_train_features = self._extract_hog_features(X_train)
                self.scaler = StandardScaler()
                self.scaler.fit(X_train_features)

        self.fitted = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform images to feature vectors.

        Args:
            X: Images of shape (N, H, W) or (N, 1, H, W)

        Returns:
            Feature matrix of shape (N, D) where D depends on the mode
        """
        if not self.fitted and self.mode != "hog":
            raise ValueError("Feature extractor must be fitted before transform.")

        if self.mode == "raw":
            # Flatten images
            features = X.reshape(X.shape[0], -1)

        elif self.mode == "pca":
            # Flatten and apply PCA
            X_flat = X.reshape(X.shape[0], -1)
            features = self.pca.transform(X_flat)

        elif self.mode == "hog":
            features = self._extract_hog_features(X)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Apply StandardScaler if needed
        if self.standardize_features and self.scaler is not None:
            features = self.scaler.transform(features)

        return features

    def _extract_hog_features(self, X: np.ndarray) -> np.ndarray:
        """
        Extract HOG features for a batch of images.

        Best practice notes:
        - Keep extraction deterministic given X.
        - Keep scaling consistent: we rescale each image to [0, 1] float32 for HOG.
        """
        features = []
        for img in X:
            # Ensure image is 2D
            if len(img.shape) == 3:
                img = img.squeeze()

            img = img.astype(np.float32, copy=False)
            # Rescale to [0, 1] for HOG (handles z-scored images too)
            img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img_norm = np.clip(img_norm, 0.0, 1.0)

            hog_feat = hog(img_norm, **self.hog_params)
            features.append(hog_feat)

        return np.array(features)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """
        Fit on training data and transform.

        Args:
            X_train: Training images

        Returns:
            Feature matrix for training data
        """
        self.fit(X_train)
        return self.transform(X_train)


def extract_features(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    mode: str = "raw",
    pca_components: int = 64,
    hog_params: Optional[Dict] = None,
    standardize: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract features from train, val, and test images.

    Fits feature extractor on training set only, then applies to all splits.

    Args:
        X_train: Training images (N, H, W)
        X_val: Validation images (M, H, W)
        X_test: Test images (K, H, W)
        mode: Feature mode ('raw', 'pca', 'hog')
        pca_components: Number of PCA components (for 'pca' mode)
        hog_params: HOG parameters (for 'hog' mode)
        standardize: Whether to standardize features

    Returns:
        Feature matrices: X_train_feat, X_val_feat, X_test_feat
    """
    extractor = FeatureExtractor(
        mode=mode,
        pca_components=pca_components,
        hog_params=hog_params,
        standardize_features=standardize,
    )

    X_train_feat = extractor.fit_transform(X_train)
    X_val_feat = extractor.transform(X_val)
    X_test_feat = extractor.transform(X_test)

    return X_train_feat, X_val_feat, X_test_feat
