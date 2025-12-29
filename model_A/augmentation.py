"""Data augmentation utilities for medical images."""

import numpy as np
from typing import Tuple, Optional
from scipy.ndimage import rotate, shift
from scipy.ndimage import gaussian_filter


def augment_image(
    image: np.ndarray,
    rotation_range: float = 10.0,
    translation_range: float = 2.0,
    flip_horizontal: bool = True,
    gaussian_noise_std: Optional[float] = 0.05,
    brightness_range: float = 0.1,
    blur_sigma: Optional[float] = None
) -> np.ndarray:
    """
    Apply random augmentations to a single image.
    
    Args:
        image: Input image of shape (H, W)
        rotation_range: Maximum rotation angle in degrees
        translation_range: Maximum translation in pixels
        flip_horizontal: Whether to apply random horizontal flip
        gaussian_noise_std: Standard deviation of Gaussian noise (None to disable)
        brightness_range: Range for brightness adjustment (0-1)
        blur_sigma: Gaussian blur sigma (None to disable)
    
    Returns:
        Augmented image
    """
    aug_image = image.copy()
    
    # Random rotation
    if rotation_range > 0:
        angle = np.random.uniform(-rotation_range, rotation_range)
        aug_image = rotate(aug_image, angle, reshape=False, mode='reflect')
    
    # Random translation
    if translation_range > 0:
        shift_x = np.random.uniform(-translation_range, translation_range)
        shift_y = np.random.uniform(-translation_range, translation_range)
        aug_image = shift(aug_image, (shift_y, shift_x), mode='reflect')
    
    # Random horizontal flip
    if flip_horizontal and np.random.random() < 0.5:
        aug_image = np.fliplr(aug_image)
    
    # Gaussian noise
    if gaussian_noise_std is not None and gaussian_noise_std > 0:
        noise = np.random.normal(0, gaussian_noise_std, aug_image.shape)
        aug_image = aug_image + noise
        # Clip to valid range
        aug_image = np.clip(aug_image, aug_image.min(), aug_image.max())
    
    # Brightness adjustment
    if brightness_range > 0:
        brightness_factor = np.random.uniform(1 - brightness_range, 1 + brightness_range)
        aug_image = aug_image * brightness_factor
        # Clip to valid range
        aug_image = np.clip(aug_image, aug_image.min(), aug_image.max())
    
    # Gaussian blur
    if blur_sigma is not None and blur_sigma > 0:
        aug_image = gaussian_filter(aug_image, sigma=blur_sigma)
    
    return aug_image


def augment_dataset(
    X: np.ndarray,
    y: np.ndarray,
    augmentation_factor: float = 1.0,
    **augmentation_kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create augmented dataset.
    
    Args:
        X: Images of shape (N, H, W)
        y: Labels of shape (N,)
        augmentation_factor: Factor by which to multiply dataset size (1.0 = same size)
        **augmentation_kwargs: Arguments passed to augment_image
    
    Returns:
        Augmented images and labels
    """
    n_samples = len(X)
    n_augmented = int(n_samples * augmentation_factor)
    
    # Select indices to augment (with replacement if needed)
    if n_augmented <= n_samples:
        indices = np.random.choice(n_samples, size=n_augmented, replace=False)
    else:
        indices = np.random.choice(n_samples, size=n_augmented, replace=True)
    
    X_aug = np.array([augment_image(X[i], **augmentation_kwargs) for i in indices])
    y_aug = y[indices]
    
    return X_aug, y_aug

