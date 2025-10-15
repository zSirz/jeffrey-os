"""
Cache Guardian - Système de rate limiting intelligent avec ML
"""

import asyncio
import logging
import os
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


class CacheGuardian:
    """
    Guardian intelligent pour rate limiting avec détection d'anomalies
    """

    def __init__(self):
        # Rate limiting basique
        self.request_counts: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.blocked_clients: dict[str, datetime] = {}

        # ML pour détection d'anomalies
        self.ml_model: IsolationForest | None = None
        self.feature_history: deque = deque(maxlen=10000)

        # Configuration
        self.rate_limits = {
            "default": {"requests": 100, "window": 60},  # 100 req/min
            "api": {"requests": 50, "window": 60},
            "auth": {"requests": 5, "window": 60},
            "admin": {"requests": 200, "window": 60},
        }

        # Métriques
        self.stats = {
            "requests_allowed": 0,
            "requests_blocked": 0,
            "anomalies_detected": 0,
            "clients_blocked": 0,
        }

    async def initialize(self):
        """Initialise le guardian avec modèle ML optionnel"""
        mode = os.getenv("SECURITY_MODE", "dev")

        # Charger ou créer le modèle ML
        model_path = "models/cache_guardian.pkl"
        if os.path.exists(model_path) and mode == "prod":
            try:
                self.ml_model = joblib.load(model_path)
                logger.info("✅ ML model loaded for Cache Guardian")
            except Exception as e:
                logger.warning(f"Could not load ML model: {e}")
                self._create_ml_model()
        else:
            if mode == "prod":
                self._create_ml_model()
            else:
                logger.info("📝 Cache Guardian in basic mode (DEV)")

        # Lancer le nettoyage périodique
        asyncio.create_task(self._cleanup_loop())

    def _create_ml_model(self):
        """Crée un nouveau modèle ML pour détection d'anomalies"""
        try:
            self.ml_model = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
            logger.info("✅ New ML model created for Cache Guardian")
        except Exception as e:
            logger.error(f"Failed to create ML model: {e}")
            self.ml_model = None

    async def check_access(self, key: str, action: str, metadata: dict[str, Any]) -> tuple[bool, str | None]:
        """
        Vérifie si l'accès est autorisé
        Retourne (allowed, reason)
        """
        client_id = metadata.get("client_id", "unknown")
        now = datetime.now()

        # Vérifier si le client est bloqué
        if client_id in self.blocked_clients:
            block_until = self.blocked_clients[client_id]
            if now < block_until:
                self.stats["requests_blocked"] += 1
                return False, f"BLOCKED_UNTIL_{block_until.isoformat()}"
            else:
                del self.blocked_clients[client_id]

        # Obtenir les limites appropriées
        rate_limit = self._get_rate_limit(action, metadata)

        # Compter les requêtes récentes
        self._record_request(client_id, now)
        request_count = self._count_recent_requests(client_id, now - timedelta(seconds=rate_limit["window"]))

        # Vérifier la limite de taux
        if request_count > rate_limit["requests"]:
            self.stats["requests_blocked"] += 1

            # Bloquer temporairement si violations répétées
            self._handle_violation(client_id, now)

            return False, f"RATE_LIMIT_EXCEEDED_{request_count}/{rate_limit['requests']}"

        # Détection d'anomalies ML si disponible
        if self.ml_model and len(self.feature_history) > 100:
            is_anomaly = self._detect_anomaly(client_id, action, metadata)
            if is_anomaly:
                self.stats["anomalies_detected"] += 1
                logger.warning(f"⚠️ Anomaly detected for client {client_id}")
                return False, "ANOMALY_DETECTED"

        self.stats["requests_allowed"] += 1
        return True, None

    def _get_rate_limit(self, action: str, metadata: dict) -> dict:
        """Détermine les limites de taux basées sur l'action"""
        # Limites spéciales pour certaines actions
        if "auth" in action.lower():
            return self.rate_limits["auth"]
        elif "admin" in metadata.get("role", ""):
            return self.rate_limits["admin"]
        elif "api" in action.lower():
            return self.rate_limits["api"]
        else:
            return self.rate_limits["default"]

    def _record_request(self, client_id: str, timestamp: datetime):
        """Enregistre une requête"""
        self.request_counts[client_id].append(timestamp)

        # Enregistrer les features pour ML
        if self.ml_model:
            features = self._extract_features(client_id, timestamp)
            self.feature_history.append(features)

    def _count_recent_requests(self, client_id: str, since: datetime) -> int:
        """Compte les requêtes récentes d'un client"""
        if client_id not in self.request_counts:
            return 0

        requests = self.request_counts[client_id]
        return sum(1 for req_time in requests if req_time > since)

    def _handle_violation(self, client_id: str, now: datetime):
        """Gère une violation de rate limit"""
        # Bloquer pour une durée croissante
        violations = self._count_violations(client_id)

        if violations >= 3:
            # Blocage exponentiel
            block_duration = min(300 * (2 ** (violations - 3)), 3600)  # Max 1h
            self.blocked_clients[client_id] = now + timedelta(seconds=block_duration)
            self.stats["clients_blocked"] += 1
            logger.warning(f"Client {client_id} blocked for {block_duration}s")

    def _count_violations(self, client_id: str) -> int:
        """Compte les violations récentes"""
        # Simplification: compter les requêtes très récentes
        now = datetime.now()
        recent = now - timedelta(minutes=5)
        return self._count_recent_requests(client_id, recent) // 50

    def _extract_features(self, client_id: str, timestamp: datetime) -> np.ndarray:
        """Extrait les features pour le ML"""
        # Features simples pour la détection d'anomalies
        hour = timestamp.hour
        minute = timestamp.minute
        weekday = timestamp.weekday()

        # Statistiques de requêtes
        req_1min = self._count_recent_requests(client_id, timestamp - timedelta(minutes=1))
        req_5min = self._count_recent_requests(client_id, timestamp - timedelta(minutes=5))
        req_15min = self._count_recent_requests(client_id, timestamp - timedelta(minutes=15))

        # Vélocité (accélération des requêtes)
        velocity = req_1min - (req_5min / 5)

        return np.array([hour, minute, weekday, req_1min, req_5min, req_15min, velocity])

    def _detect_anomaly(self, client_id: str, action: str, metadata: dict) -> bool:
        """Détecte les anomalies avec ML"""
        if not self.ml_model or len(self.feature_history) < 100:
            return False

        try:
            # Entraîner périodiquement sur l'historique
            if len(self.feature_history) % 1000 == 0:
                X = np.array(list(self.feature_history))
                self.ml_model.fit(X)

            # Prédire sur les features actuelles
            features = self._extract_features(client_id, datetime.now())
            prediction = self.ml_model.predict([features])[0]

            return prediction == -1  # -1 = anomalie dans IsolationForest

        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return False

    async def _cleanup_loop(self):
        """Nettoie périodiquement les données anciennes"""
        while True:
            try:
                await asyncio.sleep(300)  # Toutes les 5 minutes

                now = datetime.now()
                cutoff = now - timedelta(hours=1)

                # Nettoyer les requêtes anciennes
                for client_id in list(self.request_counts.keys()):
                    requests = self.request_counts[client_id]
                    # Garder seulement les requêtes récentes
                    self.request_counts[client_id] = deque((req for req in requests if req > cutoff), maxlen=1000)

                    # Supprimer si vide
                    if not self.request_counts[client_id]:
                        del self.request_counts[client_id]

                # Nettoyer les blocages expirés
                expired = [client for client, until in self.blocked_clients.items() if until < now]
                for client in expired:
                    del self.blocked_clients[client]

                logger.debug(f"Cleaned up {len(expired)} expired blocks")

            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    def get_status(self) -> dict[str, Any]:
        """Retourne le statut du guardian"""
        return {
            "stats": self.stats,
            "active_clients": len(self.request_counts),
            "blocked_clients": len(self.blocked_clients),
            "ml_enabled": self.ml_model is not None,
            "feature_history_size": len(self.feature_history),
            "rate_limits": self.rate_limits,
            "mode": os.getenv("SECURITY_MODE", "dev"),
        }
