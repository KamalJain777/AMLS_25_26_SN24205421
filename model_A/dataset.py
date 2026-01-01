"""Dataset loading and management for BreastMNIST."""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from pathlib import Path


class BreastMNISTDataset:
    """
    Unified interface for loading and managing BreastMNIST dataset.
    
    Loads train, validation, and test splits from BreastMNIST.npz file.
    Assumes the structure follows MedMNIST format with .npz file containing:
    - train_images, train_labels
    - val_images, val_labels
    - test_images, test_labels
    """
    
    def __init__(
        self,
        data_dir: str = "Datasets",
        normalize: bool = True,
        normalize_mode: str = "zscore",
        random_state: int = 42
    ):
        """
        Initialize BreastMNIST dataset loader.
        
        Args:
            data_dir: Path to directory containing the BreastMNIST.npz file
            normalize: Whether to normalize pixel values to [0, 1]
            normalize_mode: Normalization mode ('minmax' for [0,1] or 'zscore' for z-score)
            random_state: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.normalize = normalize
        self.normalize_mode = normalize_mode
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        
        # Statistics computed from training set
        self.train_mean = None
        self.train_std = None
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load data from .npz file."""
        # Look for .npz file in the data directory
        npz_files = list(self.data_dir.glob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(
                f"No .npz file found in {self.data_dir}. "
                "Please ensure BreastMNIST data is available."
            )
        
        # Use the first .npz file found (typically BreastMNIST.npz)
        data_path = npz_files[0]
        data = np.load(data_path, allow_pickle=True)
        
        # Extract splits (handle both 'images' and 'imgs' key variants)
        if 'train_images' in data:
            self.train_images = data['train_images']
            self.train_labels = data['train_labels'].flatten()
            self.val_images = data['val_images']
            self.val_labels = data['val_labels'].flatten()
            self.test_images = data['test_images']
            self.test_labels = data['test_labels'].flatten()
        elif 'train_imgs' in data:
            self.train_images = data['train_imgs']
            self.train_labels = data['train_labels'].flatten()
            self.val_images = data['val_imgs']
            self.val_labels = data['val_labels'].flatten()
            self.test_images = data['test_imgs']
            self.test_labels = data['test_labels'].flatten()
        else:
            raise KeyError(
                "Expected keys 'train_images'/'train_imgs' in .npz file. "
                f"Found keys: {list(data.keys())}"
            )
        
        # Ensure images are 2D or 3D (N, H, W) or (N, 1, H, W)
        if len(self.train_images.shape) == 4:
            # If shape is (N, 1, H, W), squeeze to (N, H, W)
            if self.train_images.shape[1] == 1:
                self.train_images = self.train_images.squeeze(1)
                self.val_images = self.val_images.squeeze(1)
                self.test_images = self.test_images.squeeze(1)
        
        # Ensure images are in correct range and shape
        self._preprocess_images()
    
    def _preprocess_images(self):
        """Preprocess images: reshape and normalize."""
        # Ensure shape is (N, 28, 28)
        if len(self.train_images.shape) != 3:
            raise ValueError(
                f"Expected 3D image array (N, H, W), got shape {self.train_images.shape}"
            )
        
        # Convert to float and normalize to [0, 1]
        if self.normalize:
            if self.normalize_mode == "minmax":
                # Simple min-max normalization to [0, 1]
                self.train_images = self.train_images.astype(np.float32) / 255.0
                self.val_images = self.val_images.astype(np.float32) / 255.0
                self.test_images = self.test_images.astype(np.float32) / 255.0
            elif self.normalize_mode == "zscore":
                # First normalize to [0, 1]
                self.train_images = self.train_images.astype(np.float32) / 255.0
                self.val_images = self.val_images.astype(np.float32) / 255.0
                self.test_images = self.test_images.astype(np.float32) / 255.0
                
                # Compute mean and std from training set
                self.train_mean = np.mean(self.train_images)
                self.train_std = np.std(self.train_images)
                
                # Apply z-score normalization
                self.train_images = (self.train_images - self.train_mean) / self.train_std
                self.val_images = (self.val_images - self.train_mean) / self.train_std
                self.test_images = (self.test_images - self.train_mean) / self.train_std
        else:
            self.train_images = self.train_images.astype(np.float32)
            self.val_images = self.val_images.astype(np.float32)
            self.test_images = self.test_images.astype(np.float32)
    
    def get_split(
        self,
        split: str = "train",
        fraction: float = 0.75,
        return_labels: bool = True
    ) -> Tuple[np.ndarray, ...]:
        """
        Get a data split with optional subsampling.
        
        Args:
            split: One of 'train', 'val', 'test'
            fraction: Fraction of data to return (0 < fraction <= 1.0)
            return_labels: Whether to return labels
        
        Returns:
            Images array, and optionally labels array if return_labels=True
        """
        if split == "train":
            images = self.train_images
            labels = self.train_labels
        elif split == "val":
            images = self.val_images
            labels = self.val_labels
        elif split == "test":
            images = self.test_images
            labels = self.test_labels
        else:
            raise ValueError(f"Unknown split: {split}. Use 'train', 'val', or 'test'.")
        
        # Subsample if fraction < 1.0 (but never subsample test set for final evaluation)
        if fraction < 1.0 and split != "test":
            n_samples = int(len(images) * fraction)
            indices = self.rng.choice(len(images), size=n_samples, replace=False)
            images = images[indices]
            labels = labels[indices]
        
        if return_labels:
            return images, labels
        return images
    
    def summarize(self) -> Dict[str, Any]:
        """
        Print basic dataset information.
        
        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            'train_samples': len(self.train_images),
            'val_samples': len(self.val_images),
            'test_samples': len(self.test_images),
            'image_shape': self.train_images.shape[1:],
            'image_dtype': self.train_images.dtype,
            'train_benign': np.sum(self.train_labels == 0),
            'train_malignant': np.sum(self.train_labels == 1),
            'val_benign': np.sum(self.val_labels == 0),
            'val_malignant': np.sum(self.val_labels == 1),
            'test_benign': np.sum(self.test_labels == 0),
            'test_malignant': np.sum(self.test_labels == 1),
        }
        
        print("=" * 60)
        print("BreastMNIST Dataset Summary")
        print("=" * 60)
        print(f"Train samples: {stats['train_samples']}")
        print(f"  - Benign: {stats['train_benign']}, Malignant: {stats['train_malignant']}")
        print(f"Val samples: {stats['val_samples']}")
        print(f"  - Benign: {stats['val_benign']}, Malignant: {stats['val_malignant']}")
        print(f"Test samples: {stats['test_samples']}")
        print(f"  - Benign: {stats['test_benign']}, Malignant: {stats['test_malignant']}")
        print(f"Image shape: {stats['image_shape']}")
        print(f"Image dtype: {stats['image_dtype']}")
        if self.train_mean is not None:
            print(f"Normalization: Z-score (mean={self.train_mean:.4f}, std={self.train_std:.4f})")
        print("=" * 60)
        
        return stats
    
    def get_all_splits(
        self,
        train_fraction: float = 0.75,
        val_fraction: float = 0.25
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all splits (train, val, test) with optional subsampling.
        
        Args:
            train_fraction: Fraction of training set to use
            val_fraction: Fraction of validation set to use
        
        Returns:
            X_train, y_train, X_val, y_val, X_test, y_test
        """
        X_train, y_train = self.get_split("train", fraction=train_fraction)
        X_val, y_val = self.get_split("val", fraction=val_fraction)
        X_test, y_test = self.get_split("test", fraction=1.0)  # Always use full test set
        
        return X_train, y_train, X_val, y_val, X_test, y_test

