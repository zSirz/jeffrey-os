"""Temperature scaling for calibration (Phase 2 Sprint 1)."""

import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy.optimize import minimize_scalar

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("SciPy not available, falling back to grid search for temperature calibration")


def fit_temperature(logits_val, y_val_encoded, class_order, method='grid'):
    """
    Find optimal temperature via NLL minimization on logits.

    Args:
        logits_val: (N, n_classes) raw logits from decision_function
        y_val_encoded: (N,) ENCODED labels as integers 0..7 (NOT strings)
        class_order: List of class names (for logging only)
        method: 'grid' (default) or 'binary' (faster, requires scipy)

    Returns:
        best_T: float (optimal temperature)
    """
    # === AJUSTEMENT 2 : Labels déjà encodés ===
    # y_val_encoded sont déjà des indices 0..7, pas besoin de mapper
    y_idx = y_val_encoded.astype(np.int32)

    # Validation safety check
    if y_idx.min() < 0 or y_idx.max() >= len(class_order):
        raise ValueError(
            f"Invalid encoded labels: min={y_idx.min()}, max={y_idx.max()}, expected 0..{len(class_order) - 1}"
        )

    def nll_loss(T):
        """Negative log-likelihood for given temperature."""
        # Protect against T close to 0
        T = max(1e-3, float(T))

        # Scale logits by temperature
        scaled = logits_val / T
        scaled = scaled - scaled.max(axis=1, keepdims=True)

        # Softmax
        probs = np.exp(scaled)
        probs = probs / probs.sum(axis=1, keepdims=True)

        # NLL
        probs_y = probs[np.arange(len(y_idx)), y_idx]
        probs_y = np.clip(probs_y, 1e-8, 1.0)
        return -np.mean(np.log(probs_y))

    if method == 'binary' and HAS_SCIPY:
        # Binary search optimization (2x faster)
        result = minimize_scalar(nll_loss, bounds=(0.5, 3.0), method='bounded')
        best_T = max(1e-3, float(result.x))
        best_nll = result.fun
    else:
        # Grid search (default, more stable)
        best_T = 1.0
        best_nll = float('inf')

        for T in np.arange(0.8, 2.5, 0.05):
            nll = nll_loss(T)
            if nll < best_nll:
                best_nll = nll
                best_T = T

    logger.info(f"✅ Temperature calibrated: T={best_T:.3f} (NLL={best_nll:.4f}, method={method})")
    return best_T


def apply_temperature(logits, T):
    """
    Apply temperature scaling to logits and return calibrated probabilities.

    Args:
        logits: (N, n_classes) raw logits
        T: float (temperature)

    Returns:
        probs: (N, n_classes) calibrated probabilities
    """
    # Protect against T close to 0
    T = max(1e-3, float(T))

    scaled = logits / T
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    probs = np.exp(scaled)
    return probs / probs.sum(axis=1, keepdims=True)
