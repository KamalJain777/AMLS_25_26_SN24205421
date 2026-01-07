"""Dataset and DataLoader utilities for Model B (PyTorch)."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
from pathlib import Path
from PIL import Image


class BreastMNISTDatasetPyTorch(Dataset):
    """PyTorch Dataset class for BreastMNIST."""
    
    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        transform=None
    ):
        """
        Initialize PyTorch dataset.
        
        Args:
            images: Images of shape (N, H, W) with values in [0, 1] or [0, 255]
            labels: Labels of shape (N,)
            transform: Optional transform to apply to images
        """
        self.images = images
        self.labels = labels
        self.transform = transform
        
        # Ensure images are in [0, 255] uint8 format for PIL
        # Handle both [0,1] normalized and raw [0, 255] images
        # Note: Model B typically uses minmax normalization to [0, 1]
        img_max = images.max()
        if img_max <= 1.0:
            # Already in [0, 1] range (minmax normalized)
            self.images = (images * 255).astype(np.uint8)
        elif img_max <= 255.0:
            # Already in [0, 255] range
            self.images = images.astype(np.uint8)
        else:
            # Unexpected range, assume [0, 1] and scale
            self.images = (np.clip(images / img_max, 0, 1) * 255).astype(np.uint8)
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get item by index.
        
        Args:
            idx: Index
        
        Returns:
            Tuple of (image tensor, label tensor)
        """
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to PIL Image
        image = Image.fromarray(image, mode='L')  # 'L' for grayscale
        
        # Apply transform (should include ToTensor)
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor
            image = torch.from_numpy(np.array(image)).float().unsqueeze(0) / 255.0
        
        # Ensure image has shape (1, H, W) for grayscale
        if len(image.shape) == 2:
            image = image.unsqueeze(0)
        
        return image, torch.tensor(label, dtype=torch.long)


def get_breastmnist_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 32,
    augment: bool = True,
    augmentation_kwargs: Optional[dict] = None,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for train, val, and test sets.
    
    Args:
        X_train: Training images (N, H, W)
        y_train: Training labels (N,)
        X_val: Validation images (M, H, W)
        y_val: Validation labels (M,)
        X_test: Test images (K, H, W)
        y_test: Test labels (K,)
        batch_size: Batch size
        augment: Whether to apply augmentation to training set
        augmentation_kwargs: Arguments for augmentation transforms
        num_workers: Number of worker processes for data loading
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from .augmentation import get_train_transforms, get_val_test_transforms
    
    # Get transforms
    if augmentation_kwargs is None:
        augmentation_kwargs = {}
    
    train_transform = get_train_transforms(augment=augment, **augmentation_kwargs)
    val_test_transform = get_val_test_transforms()
    
    # Create datasets
    train_dataset = BreastMNISTDatasetPyTorch(X_train, y_train, transform=train_transform)
    val_dataset = BreastMNISTDatasetPyTorch(X_val, y_val, transform=val_test_transform)
    test_dataset = BreastMNISTDatasetPyTorch(X_test, y_test, transform=val_test_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader

