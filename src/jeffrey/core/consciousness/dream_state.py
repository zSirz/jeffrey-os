"""
Dream State Management for Jeffrey OS DreamMode
Thread-safe state management with comprehensive metrics and privacy protection.
"""

import asyncio
import hashlib
import time
from collections import deque
from datetime import datetime
from typing import Any


class DreamState:
    """État thread-safe pour sessions de rêve avec metrics complètes."""

    def __init__(self, max_history_size: int = 1000):
        self._lock = asyncio.Lock()
        self.active = False
        self.context = {}
        self.generated_ids: set[str] = set()
        self.session_start_time = None
        self.session_id = None
        self.max_history_size = max_history_size

        # Metrics with thread-safe collections
        self.session_metrics = {
            "dreams_generated": 0,
            "dreams_successful": 0,
            "success_rate": 0.0,
            "avg_confidence": 0.0,
            "total_confidence": 0.0,
            "creativity_scores": deque(maxlen=50),  # Keep last 50 scores
            "generation_times": deque(maxlen=100),  # Keep last 100 times
            "error_count": 0,
            "last_activity": None,
        }

        # Detailed history for analysis
        self.dream_history = deque(maxlen=max_history_size)

    async def activate(self, context: dict = None, session_id: str = None):
        """Active une session de rêve."""
        async with self._lock:
            self.active = True
            self.context = context or {}
            self.session_start_time = datetime.now()

            # Generate or use provided session ID
            if session_id:
                # Hash for privacy
                self.session_id = hashlib.sha256(session_id.encode()).hexdigest()[:16]
            else:
                # Generate from timestamp and context
                unique_str = f"{time.time()}_{id(self)}_{str(context)}"
                self.session_id = hashlib.sha256(unique_str.encode()).hexdigest()[:16]

            self._reset_metrics()

    async def deactivate(self):
        """Désactive la session et nettoie."""
        async with self._lock:
            # Record session end
            if self.session_start_time:
                duration = (datetime.now() - self.session_start_time).total_seconds()
                self.session_metrics["session_duration"] = duration

            self.active = False
            self.generated_ids.clear()
            self.session_start_time = None
            self.session_id = None

    async def record_dream(
        self,
        dream_id: str,
        success: bool,
        confidence: float,
        generation_time: float = None,
        dream_data: dict = None,
        error_msg: str = None,
    ):
        """Enregistre un rêve généré avec mise à jour des métriques."""
        async with self._lock:
            # Hash l'ID pour privacy
            hashed_id = hashlib.sha256(dream_id.encode()).hexdigest()[:16]
            self.generated_ids.add(hashed_id)

            # Create dream record
            dream_record = {
                "id": hashed_id,
                "timestamp": datetime.now().isoformat(),
                "success": success,
                "confidence": confidence,
                "generation_time": generation_time,
                "session_id": self.session_id,
                "error": error_msg,
            }

            # Add anonymized dream data if provided
            if dream_data:
                dream_record["data_preview"] = self._anonymize_dream_data(dream_data)

            self.dream_history.append(dream_record)

            # Update basic metrics
            self.session_metrics["dreams_generated"] += 1
            if success:
                self.session_metrics["dreams_successful"] += 1
            else:
                self.session_metrics["error_count"] += 1

            # Update success rate
            total = self.session_metrics["dreams_generated"]
            successful = self.session_metrics["dreams_successful"]
            self.session_metrics["success_rate"] = successful / total if total > 0 else 0.0

            # Update confidence metrics (incremental average)
            if confidence is not None:
                self.session_metrics["total_confidence"] += confidence
                self.session_metrics["avg_confidence"] = self.session_metrics["total_confidence"] / total

            # Track generation time if provided
            if generation_time is not None:
                self.session_metrics["generation_times"].append(generation_time)

            # Update last activity
            self.session_metrics["last_activity"] = datetime.now().isoformat()

    async def add_creativity_score(self, score: float, context: str = None):
        """Ajoute un score de créativité avec contexte optionnel."""
        async with self._lock:
            score_record = {
                "score": score,
                "timestamp": datetime.now().isoformat(),
                "context": context,
            }
            self.session_metrics["creativity_scores"].append(score_record)

    async def record_error(self, error_type: str, error_msg: str, context: dict = None):
        """Enregistre une erreur avec contexte."""
        async with self._lock:
            error_record = {
                "type": error_type,
                "message": error_msg[:200],  # Truncate long messages
                "timestamp": datetime.now().isoformat(),
                "context": self._anonymize_dream_data(context) if context else None,
            }

            # Add to dream history as error record
            self.dream_history.append(
                {
                    "id": f"error_{len(self.dream_history)}",
                    "timestamp": error_record["timestamp"],
                    "success": False,
                    "error": error_record,
                    "session_id": self.session_id,
                }
            )

            self.session_metrics["error_count"] += 1

    async def get_current_stats(self) -> dict[str, Any]:
        """Retourne les stats en temps réel."""
        async with self._lock:
            stats = self.session_metrics.copy()

            # Convert deques to lists for JSON serialization
            stats["creativity_scores"] = list(stats["creativity_scores"])
            stats["generation_times"] = list(stats["generation_times"])

            # Add computed metrics
            if self.session_start_time:
                duration = (datetime.now() - self.session_start_time).total_seconds()
                stats["session_duration_seconds"] = duration
                stats["dreams_per_minute"] = stats["dreams_generated"] / (duration / 60) if duration > 0 else 0

            # Average generation time
            if stats["generation_times"]:
                avg_time = sum(stats["generation_times"]) / len(stats["generation_times"])
                stats["avg_generation_time_ms"] = avg_time * 1000

            # Creativity trend analysis
            creativity_scores = [s["score"] if isinstance(s, dict) else s for s in stats["creativity_scores"]]

            if len(creativity_scores) > 5:
                recent = creativity_scores[-3:]
                older = creativity_scores[-6:-3] if len(creativity_scores) > 6 else creativity_scores[:3]
                if recent and older:
                    recent_avg = sum(recent) / len(recent)
                    older_avg = sum(older) / len(older)

                    if recent_avg > older_avg * 1.1:
                        stats["creativity_trend"] = "improving"
                    elif recent_avg < older_avg * 0.9:
                        stats["creativity_trend"] = "declining"
                    else:
                        stats["creativity_trend"] = "stable"
                else:
                    stats["creativity_trend"] = "insufficient_data"
            else:
                stats["creativity_trend"] = "insufficient_data"

            return stats

    async def export_metrics(self, include_history: bool = False) -> dict:
        """Exporte les métriques pour logging/monitoring."""
        async with self._lock:
            export_data = {
                "session_id": self.session_id,
                "active": self.active,
                "metrics": await self.get_current_stats(),
                "context_keys": list(self.context.keys()) if self.context else [],
                "export_timestamp": datetime.now().isoformat(),
            }

            if include_history:
                # Include anonymized dream history
                export_data["dream_history"] = [
                    {
                        "id": dream.get("id"),
                        "timestamp": dream.get("timestamp"),
                        "success": dream.get("success"),
                        "confidence": dream.get("confidence"),
                        "generation_time": dream.get("generation_time"),
                        "error_type": dream.get("error", {}).get("type") if dream.get("error") else None,
                    }
                    for dream in list(self.dream_history)
                ]

            return export_data

    async def get_performance_insights(self) -> dict[str, Any]:
        """Analyse de performance avancée."""
        async with self._lock:
            insights = {}

            # Generation time analysis
            gen_times = list(self.session_metrics["generation_times"])
            if gen_times:
                insights["generation_time"] = {
                    "avg_ms": (sum(gen_times) / len(gen_times)) * 1000,
                    "min_ms": min(gen_times) * 1000,
                    "max_ms": max(gen_times) * 1000,
                    "std_ms": (sum((t - sum(gen_times) / len(gen_times)) ** 2 for t in gen_times) / len(gen_times))
                    ** 0.5
                    * 1000,
                }

            # Success rate trends
            recent_dreams = (
                list(self.dream_history)[-20:] if len(self.dream_history) >= 20 else list(self.dream_history)
            )
            if recent_dreams:
                recent_success_rate = sum(1 for d in recent_dreams if d.get("success", False)) / len(recent_dreams)
                insights["recent_success_rate"] = recent_success_rate

                # Compare with overall rate
                overall_rate = self.session_metrics["success_rate"]
                if recent_success_rate > overall_rate * 1.1:
                    insights["performance_trend"] = "improving"
                elif recent_success_rate < overall_rate * 0.9:
                    insights["performance_trend"] = "declining"
                else:
                    insights["performance_trend"] = "stable"

            # Error analysis
            error_dreams = [d for d in self.dream_history if not d.get("success", True)]
            if error_dreams:
                error_types = {}
                for dream in error_dreams:
                    error_info = dream.get("error", {})
                    error_type = error_info.get("type", "unknown") if isinstance(error_info, dict) else "unknown"
                    error_types[error_type] = error_types.get(error_type, 0) + 1

                insights["error_breakdown"] = error_types

            return insights

    def _anonymize_dream_data(self, data: dict) -> dict:
        """Anonymise les données sensibles."""
        if not isinstance(data, dict):
            return {"type": type(data).__name__, "anonymized": True}

        anonymized = {}

        for key, value in data.items():
            # Remove or hash sensitive keys
            if any(sensitive in key.lower() for sensitive in ["id", "user", "email", "session", "personal"]):
                if isinstance(value, str):
                    anonymized[key] = hashlib.sha256(value.encode()).hexdigest()[:8]
                else:
                    anonymized[key] = "ANONYMIZED"
            elif isinstance(value, dict):
                anonymized[key] = self._anonymize_dream_data(value)
            elif isinstance(value, str) and len(value) > 100:
                # Truncate long strings
                anonymized[key] = value[:100] + "..."
            else:
                anonymized[key] = value

        return anonymized

    def _reset_metrics(self):
        """Reset metrics for new session."""
        self.session_metrics = {
            "dreams_generated": 0,
            "dreams_successful": 0,
            "success_rate": 0.0,
            "avg_confidence": 0.0,
            "total_confidence": 0.0,
            "creativity_scores": deque(maxlen=50),
            "generation_times": deque(maxlen=100),
            "error_count": 0,
            "last_activity": None,
        }

        # Clear history but preserve recent entries if desired
        # self.dream_history.clear()  # Uncomment if you want fresh start each session

    async def is_healthy(self) -> bool:
        """Vérifie si la session est en bonne santé."""
        async with self._lock:
            # Check basic health indicators
            if not self.active:
                return False

            # Too many errors?
            if self.session_metrics["dreams_generated"] > 10 and self.session_metrics["success_rate"] < 0.3:
                return False

            # Recent activity?
            if self.session_metrics["last_activity"]:
                try:
                    last_activity = datetime.fromisoformat(self.session_metrics["last_activity"])
                    time_since_activity = (datetime.now() - last_activity).total_seconds()
                    if time_since_activity > 3600:  # 1 hour
                        return False
                except:
                    pass  # Invalid timestamp format

            return True
