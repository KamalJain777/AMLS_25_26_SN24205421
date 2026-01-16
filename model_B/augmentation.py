"""Data augmentation for deep learning model (PyTorch)."""

import torch
import torchvision.transforms as transforms
from typing import Optional


class GaussianNoise:
    """Add Gaussian noise to tensor images."""

    def __init__(self, std: float = 0.05):
        """
        Initialize Gaussian noise transform.

        Args:
            std: Standard deviation of noise
        """
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to tensor.

        Args:
            tensor: Input tensor

        Returns:
            Tensor with added noise
        """
        if self.std > 0:
            noise = torch.randn_like(tensor) * self.std
            return torch.clamp(tensor + noise, 0.0, 1.0)
        return tensor


def get_train_transforms(
    augment: bool = True, **augmentation_kwargs
) -> transforms.Compose:
    """
    Get training transforms with optional augmentation.

    Note: PyTorch's RandomRotation, RandomAffine, RandomHorizontalFlip, and ColorJitter
    work on PIL Images, so they must come BEFORE ToTensor.
    Tensor-based augmentations (like Gaussian noise) come AFTER ToTensor.

    Args:
        augment: Whether to apply augmentation
        **augmentation_kwargs: Arguments for augmentation

    Returns:
        Compose transform object
    """
    transform_list = []

    # PIL-based augmentations (must come before ToTensor)
    if augment:
        # Random rotation
        if augmentation_kwargs.get("rotation_range", 0) > 0:
            transform_list.append(
                transforms.RandomRotation(augmentation_kwargs["rotation_range"])
            )

        # Random affine (translation)
        if augmentation_kwargs.get("translation_range", 0) > 0:
            transform_list.append(
                transforms.RandomAffine(
                    degrees=0,
                    translate=(
                        augmentation_kwargs["translation_range"],
                        augmentation_kwargs["translation_range"],
                    ),
                )
            )

        # Random horizontal flip
        if augmentation_kwargs.get("flip_horizontal", True):
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

        # Color jitter (brightness and contrast)
        brightness = augmentation_kwargs.get("brightness_range", 0)
        contrast = augmentation_kwargs.get("contrast_range", 0)
        if brightness > 0 or contrast > 0:
            transform_list.append(
                transforms.ColorJitter(brightness=brightness, contrast=contrast)
            )

    # Convert to tensor (must come after PIL transforms, before tensor transforms)
    transform_list.append(transforms.ToTensor())

    # Tensor-based augmentations (must come after ToTensor)
    if augment:
        # Gaussian noise
        if augmentation_kwargs.get("gaussian_noise_std"):
            transform_list.append(
                GaussianNoise(std=augmentation_kwargs["gaussian_noise_std"])
            )

    return transforms.Compose(transform_list)


def get_val_test_transforms() -> transforms.Compose:
    """
    Get validation/test transforms (no augmentation).

    Returns:
        Compose transform object
    """
    return transforms.Compose([transforms.ToTensor()])
