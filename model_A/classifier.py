"""SVM classifier for Model A (Category A: Classical ML)."""

import numpy as np
from typing import Optional, Dict, Any
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


class ModelAClassifier:
    """
    SVM-based classifier for Model A.
    
    Supports both linear and non-linear (RBF) kernels.
    Can use raw or processed features (PCA, HOG).
    """
    
    def __init__(
        self,
        model_type: str = "svm",
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: Optional[str] = "scale",
        random_state: int = 42
    ):
        """
        Initialize Model A classifier.
        
        Args:
            model_type: Type of model ('svm' or 'logistic')
            kernel: Kernel type for SVM ('linear', 'rbf', 'poly')
            C: Regularization parameter (higher = less regularization)
            gamma: Kernel coefficient for RBF ('scale', 'auto', or float)
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.random_state = random_state
        
        # Initialize model
        if model_type == "svm":
            self.model = SVC(
                kernel=kernel,
                C=C,
                gamma=gamma,
                probability=True,  # Enable predict_proba
                random_state=random_state
            )
        elif model_type == "logistic":
            self.model = LogisticRegression(
                C=C,
                max_iter=1000,
                random_state=random_state,
                solver='lbfgs'  # Good for small datasets
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'svm' or 'logistic'.")
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Train the classifier.
        
        Args:
            X: Feature matrix of shape (N, D)
            y: Labels of shape (N,)
        """
        self.model.fit(X, y)
        return self
    
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
    
    def get_params(self) -> Dict[str, Any]:
        """Get model hyperparameters."""
        return {
            'model_type': self.model_type,
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma,
            'random_state': self.random_state
        }
    
    def set_params(self, **params):
        """Set model hyperparameters and reinitialize model."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Reinitialize model with new parameters
        if self.model_type == "svm":
            self.model = SVC(
                kernel=self.kernel,
                C=self.C,
                gamma=self.gamma,
                probability=True,
                random_state=self.random_state
            )
        elif self.model_type == "logistic":
            self.model = LogisticRegression(
                C=self.C,
                max_iter=1000,
                random_state=self.random_state,
                solver='lbfgs'
            )
        
        return self

