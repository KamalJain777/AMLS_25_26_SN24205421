"""SVM classifier for Model A (Category A: Classical ML)."""

import numpy as np
from typing import Optional, Dict, Any
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


class ModelAClassifier:
    """
    SVM-based classifier for Model A using GridSearchCV.

    Uses GridSearchCV to find optimal hyperparameters.
    Can use raw or processed features (PCA, HOG).
    """

    # Parameter grid for grid search (SVM with RBF kernel)
    PARAM_GRID = {
        "C": [0.1, 1, 10, 100, 1000, 10000],
        "gamma": [0.00001, 0.0001, 0.001, 0.01],
        "kernel": ["rbf"],
    }

    def __init__(
        self,
        param_grid: Optional[Dict[str, Any]] = None,
        use_grid_search: bool = True,
        cv: int = 5,
        verbose: int = 2,
        random_state: int = 42,
    ):
        """
        Initialize Model A classifier.

        Args:
            param_grid: Parameter grid for GridSearchCV (default: uses PARAM_GRID)
            use_grid_search: Whether to use GridSearchCV (default: True)
            cv: Number of cross-validation folds for GridSearchCV
            verbose: Verbosity level for GridSearchCV
            random_state: Random seed for reproducibility
        """
        self.use_grid_search = use_grid_search
        self.cv = cv
        self.verbose = verbose
        self.random_state = random_state
        self.param_grid = param_grid if param_grid is not None else self.PARAM_GRID

        # Initialize base SVM estimator
        base_estimator = SVC(probability=True, random_state=random_state)

        # Initialize model (GridSearchCV or direct SVC)
        self.model = (
            GridSearchCV(
                base_estimator,
                self.param_grid,
                refit=True,
                verbose=verbose,
                cv=cv,
                n_jobs=-1,  # Use all available cores
            )
            if use_grid_search
            else base_estimator
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the classifier using GridSearchCV.

        Args:
            X: Feature matrix of shape (N, D)
            y: Labels of shape (N,)
        """
        self.model.fit(X, y)
        self._print_grid_search_results()
        return self

    def _print_grid_search_results(self):
        """Print GridSearchCV results if available."""
        if self.use_grid_search and hasattr(self.model, "best_params_"):
            print("\n" + "=" * 60)
            print("GridSearchCV Results")
            print("=" * 60)
            print("Best parameters found:")
            for param, value in self.model.best_params_.items():
                print(f"  {param}: {value}")
            print(f"\nBest estimator: {self.model.best_estimator_}")
            print("=" * 60)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Feature matrix of shape (N, D)

        Returns:
            Predicted labels of shape (N,)
        """
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix of shape (N, D)

        Returns:
            Class probabilities of shape (N, 2) for binary classification
        """
        return self.model.predict_proba(X)

    @property
    def best_estimator_(self):
        """Access the best estimator from GridSearchCV."""
        if self.use_grid_search and hasattr(self.model, "best_estimator_"):
            return self.model.best_estimator_
        return self.model

    def get_params(self) -> Dict[str, Any]:
        """
        Get model hyperparameters.

        Returns:
            Dictionary containing configuration and best parameters (if available)
        """
        params = {
            "param_grid": self.param_grid,
            "use_grid_search": self.use_grid_search,
            "cv": self.cv,
            "random_state": self.random_state,
        }

        # Add best parameters if grid search was performed
        if self.use_grid_search and hasattr(self.model, "best_params_"):
            params["best_params"] = self.model.best_params_

        return params

    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """
        Get best parameters from GridSearchCV.

        Returns:
            Dictionary of best parameters, or None if grid search not used
        """
        if self.use_grid_search and hasattr(self.model, "best_params_"):
            return (
                self.model.best_params_.copy()
            )  # Return a copy to prevent external modification
        return None
