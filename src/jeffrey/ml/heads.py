"""Linear head for emotion classification (Phase 2 Sprint 1)."""

import logging

import numpy as np
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class LinearHead:
    """Simple linear head on frozen embeddings with calibration."""

    def __init__(self, n_classes: int, in_dim: int = 384):
        self.n_classes = n_classes
        self.in_dim = in_dim
        self.model = None
        self.classes_ = None

    def fit(self, X_train, y_train, class_weights=None, val_data=None):
        """
        Train linear head with L2 regularization.

        Args:
            X_train: (N, 384) embeddings
            y_train: (N,) labels
            class_weights: dict {class: weight} or None (auto-compute)
            val_data: (X_val, y_val) for validation score
        """
        # Compute class weights if not provided
        if class_weights is None:
            unique, counts = np.unique(y_train, return_counts=True)
            weights = 1.0 / counts
            weights = weights / weights.mean()  # Normalize
            weights = np.clip(weights, 0, 4.0)  # Cap à 4x
            class_weights = dict(zip(unique, weights))
            logger.info(f"Class weights (capped 4x): {class_weights}")

        # Train logistic regression with L2
        self.model = LogisticRegression(
            C=1.0,  # Inverse regularization strength
            class_weight=class_weights,
            max_iter=1000,
            random_state=42,
            multi_class='multinomial',
            solver='lbfgs',
        )

        self.model.fit(X_train, y_train)
        self.classes_ = self.model.classes_

        train_score = self.model.score(X_train, y_train)
        logger.info(f"✅ Linear head trained: {train_score:.3f} train accuracy")

        if val_data is not None:
            X_val, y_val = val_data
            val_score = self.model.score(X_val, y_val)
            logger.info(f"   Validation accuracy: {val_score:.3f}")

        return self

    def decision_function(self, X):
        """
        Get raw logits (CRITICAL for calibration).

        Returns:
            (N, n_classes) logits array
        """
        if self.model is None:
            raise RuntimeError("Model not fitted")
        return self.model.decision_function(X)

    def predict_proba(self, X):
        """Get class probabilities (post-softmax, NOT for calibration)."""
        if self.model is None:
            raise RuntimeError("Model not fitted")
        return self.model.predict_proba(X)

    def predict(self, X):
        """Get class predictions."""
        if self.model is None:
            raise RuntimeError("Model not fitted")
        return self.model.predict(X)
