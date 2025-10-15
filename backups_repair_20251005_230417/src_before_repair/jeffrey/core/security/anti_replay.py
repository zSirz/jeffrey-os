"""
Système Anti-Replay avec Redis, Bloom filters et timestamps UTC
VERSION CORRIGÉE : Fallback mémoire si Redis indisponible
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from datetime import UTC, datetime
from typing import Any

import redis.asyncio as redis
from pybloom_live import BloomFilter

from .pii_redactor import PIIRedactor

logger = logging.getLogger(__name__)


class AntiReplaySystem:
    """
    Protection contre les attaques par rejeu avec:
    - Redis pour stockage distribué (si disponible)
    - Fallback mémoire en mode DEV
    - Bloom filters pour performance
    - Timestamps UTC avec drift tolerance
    """

    def __init__(self):
        self.redis_client: redis.Redis | None = None
        self.bloom_filter: BloomFilter | None = None
        self.secret_key = secrets.token_bytes(32)

        # Fallback mémoire si Redis down
        self._memory_nonces: set[str] = set()
        self._memory_nonces_timestamps: dict[str, datetime] = {}

        # Configuration
        self.timestamp_tolerance = 60  # secondes
        self.drift_alert_threshold = 10  # secondes
        self.nonce_ttl = 120  # secondes

        # Métriques
        self.stats = {
            "replay_attempts": 0,
            "valid_requests": 0,
            "drift_warnings": 0,
            "bloom_hits": 0,
            "using_redis": False,
        }

        # Monotonic pour fenêtre glissante
        self.start_monotonic = time.monotonic()

    async def initialize(self):
        """Initialise avec Redis si disponible, sinon fallback mémoire"""
        mode = os.getenv("SECURITY_MODE", "dev")

        # Tente la connexion Redis
        try:
            self.redis_client = await redis.from_url(
                "redis://localhost:6379",
                decode_responses=True,
                socket_connect_timeout=2 if mode == "dev" else 5,
            )
            await self.redis_client.ping()
            self.stats["using_redis"] = True
            logger.info("✅ Redis connected for anti-replay")
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            self.redis_client = None
            self.stats["using_redis"] = False

            if mode == "prod":
                raise Exception("Redis required in production mode")
            else:
                logger.info("📝 Using in-memory fallback for anti-replay (DEV mode)")

        # Bloom filter dans tous les cas
        self.bloom_filter = BloomFilter(capacity=100000, error_rate=0.001)

        # Nettoyage périodique
        asyncio.create_task(self._cleanup_loop())

    async def validate_request(self, request_data: dict) -> tuple[bool, str | None]:
        """
        Valide qu'une requête n'est pas un rejeu
        Fonctionne avec Redis ou fallback mémoire
        """
        # Log safe
        safe_data = PIIRedactor.redact_dict(request_data)
        logger.debug(f"Validating request: {json.dumps(safe_data)}")

        # Extraire les éléments
        nonce = request_data.get("nonce")
        timestamp_str = request_data.get("timestamp")
        signature = request_data.get("signature")
        client_id = request_data.get("client_id")

        # Vérifications basiques
        if not all([nonce, timestamp_str, signature, client_id]):
            return False, "MISSING_SECURITY_PARAMS"

        # Vérifier le timestamp
        valid_time, time_error = self._validate_timestamp_utc(timestamp_str)
        if not valid_time:
            self.stats["replay_attempts"] += 1
            return False, time_error

        # Vérifier la signature
        expected_sig = self._compute_signature(client_id, nonce, timestamp_str)
        if not hmac.compare_digest(signature, expected_sig):
            self.stats["replay_attempts"] += 1
            return False, "INVALID_SIGNATURE"

        # Check Bloom filter d'abord
        bloom_key = f"{client_id}:{nonce}"
        if bloom_key in self.bloom_filter:
            self.stats["bloom_hits"] += 1

            # Vérifier dans Redis ou mémoire
            if self.redis_client:
                is_used = await self._is_nonce_used_redis(client_id, nonce)
            else:
                is_used = self._is_nonce_used_memory(client_id, nonce)

            if is_used:
                self.stats["replay_attempts"] += 1
                return False, "NONCE_ALREADY_USED"

        # Enregistrer le nonce
        if self.redis_client:
            success = await self._register_nonce_redis(client_id, nonce)
        else:
            success = self._register_nonce_memory(client_id, nonce)

        if not success:
            self.stats["replay_attempts"] += 1
            return False, "NONCE_REGISTRATION_FAILED"

        # Ajouter au Bloom filter
        self.bloom_filter.add(bloom_key)

        self.stats["valid_requests"] += 1
        return True, None

    def _validate_timestamp_utc(self, timestamp_str: str) -> tuple[bool, str | None]:
        """Valide le timestamp avec UTC et drift detection"""
        try:
            # Parser avec support timezone
            if timestamp_str.endswith("Z"):
                timestamp_str = timestamp_str[:-1] + "+00:00"

            request_time = datetime.fromisoformat(timestamp_str)

            # Forcer UTC si pas de timezone
            if request_time.tzinfo is None:
                request_time = request_time.replace(tzinfo=UTC)

            now_utc = datetime.now(UTC)

            # Calculer la différence
            time_diff = (now_utc - request_time).total_seconds()

            # Alerter si drift important
            if abs(time_diff) > self.drift_alert_threshold:
                self.stats["drift_warnings"] += 1
                logger.warning(f"⚠️ Clock drift detected: {time_diff:.1f}s")

            # Vérifier la fenêtre
            if time_diff > self.timestamp_tolerance:
                return False, f"TIMESTAMP_TOO_OLD_{time_diff:.0f}s"
            elif time_diff < -5:
                return False, f"TIMESTAMP_IN_FUTURE_{-time_diff:.0f}s"

            return True, None

        except Exception as e:
            logger.error(f"Timestamp parsing error: {e}")
            return False, "INVALID_TIMESTAMP_FORMAT"

    def _compute_signature(self, client_id: str, nonce: str, timestamp: str) -> str:
        """Calcule la signature HMAC-SHA256"""
        message = f"{client_id}:{nonce}:{timestamp}".encode()
        return hmac.new(self.secret_key, message, hashlib.sha256).hexdigest()

    # Méthodes Redis
    async def _is_nonce_used_redis(self, client_id: str, nonce: str) -> bool:
        """Vérifie si le nonce existe dans Redis"""
        if not self.redis_client:
            return False

        key = f"nonce:{client_id}:{nonce}"
        exists = await self.redis_client.exists(key)
        return bool(exists)

    async def _register_nonce_redis(self, client_id: str, nonce: str) -> bool:
        """Enregistre le nonce dans Redis avec TTL"""
        if not self.redis_client:
            return False

        key = f"nonce:{client_id}:{nonce}"

        result = await self.redis_client.set(
            key,
            json.dumps(
                {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "monotonic": time.monotonic() - self.start_monotonic,
                }
            ),
            nx=True,  # Only if not exists
            ex=self.nonce_ttl,  # TTL
        )

        return bool(result)

    # Méthodes fallback mémoire
    def _is_nonce_used_memory(self, client_id: str, nonce: str) -> bool:
        """Vérifie si le nonce existe en mémoire (fallback)"""
        key = f"{client_id}:{nonce}"
        return key in self._memory_nonces

    def _register_nonce_memory(self, client_id: str, nonce: str) -> bool:
        """Enregistre le nonce en mémoire (fallback)"""
        key = f"{client_id}:{nonce}"

        if key in self._memory_nonces:
            return False

        self._memory_nonces.add(key)
        self._memory_nonces_timestamps[key] = datetime.now(UTC)
        return True

    async def _cleanup_loop(self):
        """Nettoie périodiquement les nonces expirés"""
        while True:
            try:
                await asyncio.sleep(60)  # Toutes les minutes

                # Nettoyer la mémoire si pas Redis
                if not self.redis_client:
                    now = datetime.now(UTC)
                    expired_keys = []

                    for key, timestamp in self._memory_nonces_timestamps.items():
                        if (now - timestamp).total_seconds() > self.nonce_ttl:
                            expired_keys.append(key)

                    for key in expired_keys:
                        self._memory_nonces.discard(key)
                        del self._memory_nonces_timestamps[key]

                    if expired_keys:
                        logger.debug(f"Cleaned {len(expired_keys)} expired nonces from memory")

                # Reset Bloom filter si trop plein
                if self.bloom_filter and self.bloom_filter.count > 90000:
                    self.bloom_filter = BloomFilter(capacity=100000, error_rate=0.001)
                    logger.info("🔄 Bloom filter reset")

            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    def generate_secure_request(self, client_id: str, data: dict) -> dict:
        """Génère une requête sécurisée avec anti-replay"""
        nonce = secrets.token_urlsafe(32)
        timestamp = datetime.now(UTC).isoformat()
        signature = self._compute_signature(client_id, nonce, timestamp)

        return {
            **data,
            "client_id": client_id,
            "nonce": nonce,
            "timestamp": timestamp,
            "signature": signature,
        }

    def get_status(self) -> dict[str, Any]:
        """Retourne les métriques du système"""
        return {
            "stats": self.stats,
            "using_redis": self.stats["using_redis"],
            "memory_nonces_count": len(self._memory_nonces) if not self.redis_client else 0,
            "bloom_filter_count": self.bloom_filter.count if self.bloom_filter else 0,
            "config": {
                "timestamp_tolerance": self.timestamp_tolerance,
                "drift_alert_threshold": self.drift_alert_threshold,
                "nonce_ttl": self.nonce_ttl,
            },
        }
