"""Training utilities for Model A."""

import numpy as np
from typing import Dict, Any
from .classifier import ModelAClassifier
from .evaluator import evaluate_model_a, save_results, compare_feature_modes


def train_model_a(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: Dict[str, Any],
    feature_mode: str = "raw",
) -> Dict[str, Any]:
    """
    Train Model A classifier.

    Args:
        X_train: Training features (already feature-extracted)
        y_train: Training labels
        X_val: Validation features (already feature-extracted)
        y_val: Validation labels
        X_test: Test features (already feature-extracted)
        y_test: Test labels
        config: Configuration dictionary with model hyperparameters
        feature_mode: Feature mode used (for logging)

    Returns:
        Dictionary containing trained model and evaluation results
    """
    # Initialize classifier with GridSearchCV
    # Get GridSearchCV parameters from config (look in 'model' key if nested)
    model_config = config.get("model", config) if "model" in config else config

    use_grid_search = model_config.get("use_grid_search", True)
    cv = model_config.get("cv", 5)
    verbose = model_config.get("verbose", 2)

    # Allow custom param_grid from config, otherwise use default
    param_grid = model_config.get("param_grid", None)

    model = ModelAClassifier(
        param_grid=param_grid,
        use_grid_search=use_grid_search,
        cv=cv,
        verbose=verbose,
        random_state=config.get("random_state", 42),
    )

    print(f"\nTraining Model A with {feature_mode} features...")
    print(f"Model parameters: {model.get_params()}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    # Train model
    model.fit(X_train, y_train)

    # Evaluate on all splits (with probabilities for ROC curves)
    results = {
        "feature_mode": feature_mode,
        "config": config,
        "train": evaluate_model_a(
            model,
            X_train,
            y_train,
            split="train",
            print_results=False,
            return_proba=True,
        ),
        "val": evaluate_model_a(
            model, X_val, y_val, split="val", print_results=False, return_proba=True
        ),
        "test": evaluate_model_a(
            model, X_test, y_test, split="test", print_results=True, return_proba=True
        ),
    }

    return {"model": model, "results": results}
