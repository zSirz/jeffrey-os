"""
Entropy Guardian - Monitors entropy and detects bias
"""

import logging
import time
from collections import defaultdict, deque
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    from scipy.stats import entropy as scipy_entropy

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.info("scipy not available, using simple entropy calculation")


class EntropyGuardian:
    """Surveille entropie et détecte biais"""

    def __init__(self):
        self.entropy_history = defaultdict(lambda: deque(maxlen=100))
        self.bias_alerts = []
        self.recommendations_cache = []
        self.last_check_time = {}

    def check_entropy(self, component: str, data: Any) -> float:
        """Calcule et surveille entropie"""
        entropy = self._calculate_entropy(data)
        self.entropy_history[component].append(entropy)
        self.last_check_time[component] = time.time()

        # Alerte si entropie baisse
        if len(self.entropy_history[component]) > 10:
            recent = list(self.entropy_history[component])[-10:]
            avg_recent = sum(recent) / len(recent)

            if avg_recent < 1.5:  # Seuil critique
                alert = {
                    "component": component,
                    "entropy": avg_recent,
                    "risk": "high" if avg_recent < 1.0 else "medium",
                    "timestamp": time.time(),
                }
                self.bias_alerts.append(alert)
                logger.warning(f"Bias risk in {component}: entropy={avg_recent:.2f}")

                # Limiter les alertes
                if len(self.bias_alerts) > 100:
                    self.bias_alerts = self.bias_alerts[-100:]

        return entropy

    def _calculate_entropy(self, data: Any) -> float:
        """Calcule entropie selon type de données"""
        if isinstance(data, dict):
            # Pour Q-table : entropie des valeurs
            values = []
            for actions in data.values():
                if isinstance(actions, dict):
                    values.extend(actions.values())
                else:
                    values.append(actions)

            if not values:
                return 0.0

            # Normaliser et calculer distribution
            try:
                values = [float(v) for v in values if v is not None]
                if not values:
                    return 0.0

                min_val = min(values)
                max_val = max(values)

                if max_val == min_val:
                    return 0.0

                normalized = [(v - min_val) / (max_val - min_val) for v in values]

                # Bins pour distribution
                bins = 10
                hist, _ = np.histogram(normalized, bins=bins)
                probs = hist / hist.sum() if hist.sum() > 0 else hist

                # Shannon entropy
                if HAS_SCIPY:
                    return scipy_entropy(probs + 1e-10, base=2)  # +epsilon pour éviter log(0)
                else:
                    return self._simple_entropy(probs + 1e-10)

            except Exception as e:
                logger.error(f"Error calculating dict entropy: {e}")
                return 0.0

        elif isinstance(data, list):
            # Pour clusters : diversité
            if not data:
                return 0.0
            unique = len(set(str(d) for d in data))
            total = len(data)
            return np.log2(unique) if unique > 1 else 0.0

        elif isinstance(data, (np.ndarray, list)):
            # Pour arrays : distribution
            if len(data) == 0:
                return 0.0

            try:
                # Calculer distribution
                unique, counts = np.unique(data, return_counts=True)
                if len(unique) <= 1:
                    return 0.0

                probs = counts / counts.sum()
                if HAS_SCIPY:
                    return scipy_entropy(probs, base=2)
                else:
                    return self._simple_entropy(probs)

            except Exception as e:
                logger.error(f"Error calculating array entropy: {e}")
                return 0.0

        else:
            return 0.0

    def _simple_entropy(self, probs: np.ndarray) -> float:
        """Calcul d'entropie simple sans scipy"""
        entropy_val = 0.0
        for p in probs:
            if p > 0:
                entropy_val -= p * np.log2(p)
        return entropy_val

    def get_recommendations(self) -> list[str]:
        """Recommandations pour augmenter entropie"""
        recommendations = []

        for component, history in self.entropy_history.items():
            if not history:
                continue

            current = history[-1]

            if current < 1.0:
                recommendations.append(
                    f"{component}: Increase exploration (ε-greedy), current entropy={current:.2f} is too low"
                )
            elif current < 1.5:
                recommendations.append(
                    f"{component}: Consider diversity injection, entropy={current:.2f} approaching critical"
                )

        # Cache recommendations for efficiency
        self.recommendations_cache = recommendations
        return recommendations

    def get_status(self) -> dict[str, Any]:
        """Get current status of entropy monitoring"""
        status = {
            "monitored_components": list(self.entropy_history.keys()),
            "total_checks": sum(len(h) for h in self.entropy_history.values()),
            "active_alerts": len([a for a in self.bias_alerts if time.time() - a["timestamp"] < 300]),  # Last 5 min
            "current_recommendations": len(self.get_recommendations()),
        }

        # Average entropy per component
        avg_entropies = {}
        for comp, history in self.entropy_history.items():
            if history:
                avg_entropies[comp] = sum(history) / len(history)

        status["average_entropies"] = avg_entropies

        # Risk levels
        risk_counts = {"high": 0, "medium": 0, "low": 0}
        for alert in self.bias_alerts:
            if time.time() - alert["timestamp"] < 300:  # Last 5 min
                risk_counts[alert["risk"]] += 1

        status["risk_levels"] = risk_counts

        return status

    def clear_old_alerts(self, max_age_seconds: float = 3600):
        """Clean up old alerts"""
        current_time = time.time()
        self.bias_alerts = [alert for alert in self.bias_alerts if current_time - alert["timestamp"] < max_age_seconds]

    def inject_diversity(self, component: str, current_value: float, diversity_factor: float = 0.1) -> float:
        """Inject random diversity to increase entropy"""
        import random

        # Check if component needs diversity
        if component in self.entropy_history:
            recent_entropy = self.entropy_history[component][-1] if self.entropy_history[component] else 2.0

            if recent_entropy < 1.5:
                # Add noise proportional to how low entropy is
                noise_scale = (1.5 - recent_entropy) * diversity_factor
                noise = random.gauss(0, noise_scale)
                new_value = current_value + noise

                logger.debug(f"Injected diversity in {component}: {current_value:.3f} -> {new_value:.3f}")
                return new_value

        return current_value
