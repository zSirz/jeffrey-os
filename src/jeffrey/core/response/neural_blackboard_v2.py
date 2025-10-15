import asyncio
import json
import secrets
import time
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Any

import mmh3  # Pour Counting Bloom Filter
from cachetools import TTLCache


@dataclass
class CapabilityToken:
    """Token de capacité pour accès sécurisé au blackboard"""

    module_id: str  # Ajout pour tracking
    allowed_keys: set[str]
    expires_at: float
    issued_at: float = field(default_factory=time.time)
    correlation_id: str | None = None  # Optionnel maintenant

    def is_valid(self) -> bool:
        """Vérifie si le token est encore valide"""
        return time.time() < self.expires_at

    def can_access(self, key: str) -> bool:
        """Vérifie l'accès avec support des wildcards

        Exemples:
            - "phase_*" match "phase_1", "phase_2", etc.
            - "thalamus_*" match "thalamus_context", "thalamus_state"
            - "specific" match exactement "specific"
        """
        if not self.is_valid():
            return False

        # Support des patterns wildcards avec fnmatch
        for pattern in self.allowed_keys:
            if fnmatch(key, pattern):
                return True
        return False


class CountingBloomFilter:
    """Bloom Filter pour déduplication efficace sans explosion mémoire"""

    def __init__(self, size: int = 10000, num_hashes: int = 3):
        self.size = size
        self.num_hashes = num_hashes
        self.buckets = [0] * size

    def add(self, item: str):
        for seed in range(self.num_hashes):
            index = mmh3.hash(item, seed) % self.size
            if self.buckets[index] < 255:  # Saturation à 255
                self.buckets[index] += 1

    def contains(self, item: str) -> bool:
        for seed in range(self.num_hashes):
            index = mmh3.hash(item, seed) % self.size
            if self.buckets[index] == 0:
                return False
        return True

    def remove(self, item: str):
        # Approximatif mais suffisant
        for seed in range(self.num_hashes):
            index = mmh3.hash(item, seed) % self.size
            if self.buckets[index] > 0:
                self.buckets[index] -= 1


class NeuralBlackboard:
    """
    Blackboard durci v2 avec capability tokens et optimisations
    """

    def __init__(self, ttl_seconds: int = 120, max_entries: int = 1000, max_memory_mb: int = 100):
        self._storage = {}  # correlation_id -> data dict
        self._metadata = {}  # correlation_id -> metadata
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._max_memory_mb = max_memory_mb

        # Protection
        self._lock = asyncio.Lock()
        self._tokens = {}  # token_id -> CapabilityToken
        self._dedup_bloom = CountingBloomFilter(size=20000)

        # Cache LRU pour hot paths
        self._hot_cache = TTLCache(maxsize=100, ttl=10)

        # Métriques - Initialiser tous les compteurs pour éviter KeyError
        self._stats = {
            "hits": 0,
            "misses": 0,
            "writes": 0,
            "evictions": 0,
            "security_denied": 0,
            "dedup_blocked": 0,
            "dedup_cleaned": 0,
            "memory_drops": 0,
        }

        # Pour tracking dédup par corrélation
        self._dedup_keys_by_correlation = {}

        # Pour estimation mémoire robuste
        self._total_bytes = 0
        self._entry_bytes = {}  # (correlation_id, key) -> size

        # Cleanup task
        self._cleanup_task = None

    async def start(self):
        """Démarre le blackboard avec cleanup automatique"""
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())

    async def stop(self):
        """Arrête le blackboard proprement"""
        if self._cleanup_task:
            self._cleanup_task.cancel()

    async def create_capability_token(
        self,
        module_id: str,
        allowed_keys: set[str],
        ttl_ms: float = 60000,
        correlation_id: str | None = None,
    ) -> str:
        """Crée un token de capacité pour accès limité

        Args:
            module_id: ID du module demandeur
            allowed_keys: Ensemble de clés ou patterns (avec wildcards)
            ttl_ms: TTL en millisecondes
            correlation_id: ID de corrélation optionnel
        """
        token_id = secrets.token_urlsafe(16)

        token = CapabilityToken(
            module_id=module_id,
            allowed_keys=allowed_keys,
            expires_at=time.time() + (ttl_ms / 1000.0),
            issued_at=time.time(),
            correlation_id=correlation_id,
        )

        async with self._lock:
            self._tokens[token_id] = token

        return token_id

    def _approx_size(self, value: Any) -> int:
        """Estimation robuste de la taille mémoire en bytes

        Gère tous les types Python, même non-sérialisables JSON
        """
        if value is None:
            return 0
        elif isinstance(value, (bytes, bytearray)):
            return len(value)
        elif isinstance(value, str):
            return len(value.encode("utf-8"))
        elif isinstance(value, bool):
            return 1
        elif isinstance(value, (int, float)):
            return 8  # Taille standard 64 bits
        elif isinstance(value, dict):
            size = 240  # Overhead dict Python
            for k, v in value.items():
                size += self._approx_size(k) + self._approx_size(v)
            return size
        elif isinstance(value, (list, tuple)):
            size = 56 if isinstance(value, list) else 48  # Overhead
            for item in value:
                size += self._approx_size(item)
            return size
        elif isinstance(value, set):
            size = 224  # Overhead set Python
            for item in value:
                size += self._approx_size(item)
            return size
        else:
            # Fallback pour objets custom (classes, etc.)
            try:
                # Utilise repr pour estimation
                return len(repr(value).encode("utf-8"))
            except:
                # Si même repr échoue, estimation fixe
                return 256

    async def write(
        self,
        correlation_id: str,  # TOUJOURS correlation_id, jamais session_id !
        key: str,
        value: Any,
        capability_token: str,
        ttl_ms: int | None = None,
        dedup_key: str | None = None,
    ) -> bool:
        """Write avec gestion mémoire robuste"""

        async with self._lock:
            # Vérifier capability si fournie
            if capability_token:
                if capability_token not in self._tokens:
                    return False
                token = self._tokens[capability_token]
                if not token.can_access(key):
                    return False

            # Déduplication SCOPÉE par correlation_id
            if dedup_key:
                # Clé scopée = correlation:dedup pour isolation
                scoped_key = f"{correlation_id}:{dedup_key}"

                if self._dedup_bloom.contains(scoped_key):
                    self._stats["dedup_blocked"] += 1
                    return False  # Déjà traité dans cette corrélation

                # Ajouter au bloom filter
                self._dedup_bloom.add(scoped_key)

                # Track pour cleanup ultérieur
                if correlation_id not in self._dedup_keys_by_correlation:
                    self._dedup_keys_by_correlation[correlation_id] = set()
                self._dedup_keys_by_correlation[correlation_id].add(dedup_key)

            # Calcul incrémental de la mémoire AVANT écriture
            entry_key = (correlation_id, key)
            new_size = self._approx_size(value)
            old_size = self._entry_bytes.get(entry_key, 0)

            # Update compteurs
            self._entry_bytes[entry_key] = new_size
            self._total_bytes += new_size - old_size

            # Vérifier limite mémoire et évincer si nécessaire
            while self._total_bytes > self._max_memory_mb * 1024 * 1024:
                await self._evict_oldest()
                self._stats["memory_drops"] += 1

            # Initialiser si nécessaire
            if correlation_id not in self._storage:
                self._storage[correlation_id] = {}
                self._metadata[correlation_id] = {
                    "created_at": time.time(),
                    "last_access": time.time(),
                    "access_count": 0,
                    "ttl_ms": ttl_ms or 120000,
                }

            # Écrire
            self._storage[correlation_id][key] = value
            self._metadata[correlation_id]["last_access"] = time.time()
            self._stats["writes"] += 1

            # Invalider hot cache si nécessaire
            cache_key = f"{correlation_id}:{key}"
            if cache_key in self._hot_cache:
                del self._hot_cache[cache_key]

            return True

    async def read(
        self, correlation_id: str, key: str | None = None, capability_token: str | None = None
    ) -> Any | None:
        """Read avec vérification sécurité AVANT cache

        Args:
            correlation_id: ID de corrélation
            key: Clé optionnelle à lire
            capability_token: Token de capacité optionnel pour vérification
        """

        # CRITIQUE: Vérifier le token AVANT tout accès aux données
        if capability_token:
            token = self._tokens.get(capability_token)
            if token is None:
                self._stats["security_denied"] += 1
                return None

            # Interdire la lecture globale avec un token (limite surface d'attaque)
            if key is None:
                self._stats["security_denied"] += 1
                return None

            # Vérifier que le token a accès à cette clé spécifique
            if not token.can_access(key):
                self._stats["security_denied"] += 1
                return None

        # MAINTENANT on peut consulter le cache en sécurité
        if key:
            cache_key = f"{correlation_id}:{key}"
            if cache_key in self._hot_cache:
                self._stats["hits"] += 1
                return self._hot_cache[cache_key]

        # Accès au storage principal
        async with self._lock:
            if correlation_id not in self._storage:
                self._stats["misses"] += 1
                return None

            # Vérifier TTL
            metadata = self._metadata[correlation_id]
            if time.time() - metadata["created_at"] > self._ttl:
                # Expiré
                del self._storage[correlation_id]
                del self._metadata[correlation_id]
                self._stats["misses"] += 1
                return None

            # Mettre à jour métadonnées
            metadata["last_access"] = time.time()
            metadata["access_count"] += 1

            if key:
                value = self._storage[correlation_id].get(key)
                if value is not None:
                    # Ajouter au hot cache pour accès futur
                    cache_key = f"{correlation_id}:{key}"
                    self._hot_cache[cache_key] = value
                    # Gérer taille du cache LRU simple
                    if len(self._hot_cache) > 100:
                        # Supprimer le plus ancien (FIFO simple)
                        first_key = next(iter(self._hot_cache))
                        del self._hot_cache[first_key]
                self._stats["hits"] += 1
                return value
            else:
                # Retourner toutes les clés si pas de key spécifique
                self._stats["hits"] += 1
                return dict(self._storage[correlation_id])

    async def _periodic_cleanup(self):
        """Nettoie périodiquement les entrées expirées"""
        while True:
            try:
                await asyncio.sleep(30)  # Toutes les 30 secondes
                await self._cleanup()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"❌ Erreur cleanup blackboard: {e}")

    async def _cleanup_expired(self):
        """Cleanup périodique avec gestion mémoire"""
        now = time.time()
        expired_correlations = []

        for corr_id, meta in list(self._metadata.items()):
            created_at = meta.get("created_at", now)
            ttl_ms = meta.get("ttl_ms", 120000)

            if (now - created_at) * 1000 > ttl_ms:
                expired_correlations.append(corr_id)

        # Nettoyer avec update mémoire
        for corr_id in expired_correlations:
            if corr_id in self._storage:
                for key in list(self._storage[corr_id].keys()):
                    entry_key = (corr_id, key)
                    size = self._entry_bytes.get(entry_key, 0)
                    self._total_bytes -= size
                    self._entry_bytes.pop(entry_key, None)

            # Nettoyer les clés de dédup
            if corr_id in self._dedup_keys_by_correlation:
                for dedup_key in self._dedup_keys_by_correlation[corr_id]:
                    scoped_key = f"{corr_id}:{dedup_key}"
                    if hasattr(self._dedup_bloom, "remove"):
                        try:
                            self._dedup_bloom.remove(scoped_key)
                            self._stats["dedup_cleaned"] += 1
                        except:
                            pass

                del self._dedup_keys_by_correlation[corr_id]

            self._storage.pop(corr_id, None)
            self._metadata.pop(corr_id, None)

    async def _cleanup(self):
        """Nettoie les entrées expirées et tokens"""
        async with self._lock:
            # Cleanup des corrélations expirées
            await self._cleanup_expired()

            # Nettoyer tokens
            expired_tokens = []
            for token_id, token in self._tokens.items():
                if not token.is_valid():
                    expired_tokens.append(token_id)

            for token_id in expired_tokens:
                del self._tokens[token_id]

            # Limiter nombre d'entrées
            if len(self._storage) > self._max_entries:
                # Supprimer les moins récemment accédés
                sorted_by_access = sorted(self._metadata.items(), key=lambda x: x[1]["last_access"])
                to_remove = len(self._storage) - self._max_entries
                for corr_id, _ in sorted_by_access[:to_remove]:
                    # Utiliser _evict_oldest pour cleanup correct de la mémoire
                    if corr_id in self._storage:
                        for key in list(self._storage[corr_id].keys()):
                            entry_key = (corr_id, key)
                            size = self._entry_bytes.get(entry_key, 0)
                            self._total_bytes -= size
                            self._entry_bytes.pop(entry_key, None)

                    if corr_id in self._dedup_keys_by_correlation:
                        del self._dedup_keys_by_correlation[corr_id]

                    del self._storage[corr_id]
                    del self._metadata[corr_id]

    async def _evict_oldest(self):
        """Éviction avec update correct des compteurs mémoire"""
        if not self._metadata:
            return

        # Trouver la corrélation la plus ancienne
        oldest_corr = min(self._metadata.items(), key=lambda x: x[1].get("last_access", 0))
        corr_id = oldest_corr[0]

        # Décrémenter la mémoire pour toutes les clés évincées
        if corr_id in self._storage:
            for key in list(self._storage[corr_id].keys()):
                entry_key = (corr_id, key)
                size = self._entry_bytes.get(entry_key, 0)
                self._total_bytes -= size
                self._entry_bytes.pop(entry_key, None)

        # Cleanup dédup pour la corrélation évincée
        if corr_id in self._dedup_keys_by_correlation:
            for dedup_key in self._dedup_keys_by_correlation[corr_id]:
                scoped_key = f"{corr_id}:{dedup_key}"
                if hasattr(self._dedup_bloom, "remove"):
                    try:
                        self._dedup_bloom.remove(scoped_key)
                        self._stats["dedup_cleaned"] += 1
                    except:
                        pass

            del self._dedup_keys_by_correlation[corr_id]

        # Nettoyer storage et metadata
        self._storage.pop(corr_id, None)
        self._metadata.pop(corr_id, None)
        self._stats["evictions"] += 1

    def _estimate_memory_usage(self) -> int:
        """Estime l'usage mémoire (approximatif)"""
        # Sérialiser pour estimer (coûteux, à optimiser)
        try:
            return len(json.dumps(self._storage))
        except:
            return 0

    def get_stats(self) -> dict:
        """Statistiques détaillées"""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0

        return {
            **self._stats,
            "hit_rate": hit_rate,
            "entries": len(self._storage),
            "tokens": len(self._tokens),
            "hot_cache_size": len(self._hot_cache),
        }
