"""
Cache LRU avec TTL adaptatif et metadata pour future évolution neurale
"""

import hashlib
import json
import time
from collections import OrderedDict
from typing import Any

import numpy as np


def _canonicalize(obj: Any) -> str:
    """Sérialisation canonique pour clé stable"""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def make_cache_key(prompt: str, *, context: dict, emotion: str) -> str:
    """Génère clé de cache déterministe et privacy-safe"""
    payload = _canonicalize({"p": prompt, "ctx": context, "emo": emotion})
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cosine(u: np.ndarray, v: np.ndarray) -> float:
    """Cosine similarity sans sklearn"""
    u = u.astype(float)
    v = v.astype(float)
    denom = (np.linalg.norm(u) * np.linalg.norm(v)) + 1e-12
    return float(np.dot(u, v) / denom)


class NeuralLRUCache:
    """Cache LRU avec support metadata pour adaptation neurale future"""

    def __init__(self, max_size: int = 100, ttl_s: int = 3600):
        self.max_size = max_size
        self.base_ttl = ttl_s
        self._store = OrderedDict()  # key -> (timestamp, value, meta, access_count)
        self._embeddings = {}  # Pour fuzzy invalidation future

    def get(self, key: str) -> Any | None:
        """Récupère avec adaptation TTL selon confidence"""
        item = self._store.get(key)
        if not item:
            return None

        ts, value, meta, access_count = item

        # TTL adaptatif selon confidence (si présente)
        ttl = self.base_ttl
        if meta and "confidence" in meta:
            # Réduire TTL si faible confidence
            confidence = meta["confidence"]
            ttl = int(ttl * (0.5 + confidence * 0.5))  # 50% à 100% du TTL

        if (time.time() - ts) > ttl:
            self._store.pop(key, None)
            return None

        # Incrémenter compteur d'accès (pour future priorisation)
        self._store[key] = (ts, value, meta, access_count + 1)
        self._store.move_to_end(key)
        return value

    def set(
        self,
        key: str,
        value: Any,
        meta: dict | None = None,
        embedding: np.ndarray | None = None,
    ):
        """Stocke avec metadata et embedding optionnel pour fuzzy matching"""
        self._store[key] = (time.time(), value, meta or {}, 1)
        self._store.move_to_end(key)

        if embedding is not None:
            self._embeddings[key] = embedding

        # Éviction LRU avec pondération access_count
        if len(self._store) > self.max_size:
            # Trouve l'item avec le score le plus bas (old + peu accédé)
            min_score = float("inf")
            min_key = None
            now = time.time()

            for k, (ts, _, _, count) in self._store.items():
                age = now - ts
                score = count / (age + 1)  # Plus c'est vieux et peu utilisé, plus le score est bas
                if score < min_score:
                    min_score = score
                    min_key = k

            if min_key:
                self._store.pop(min_key)
                self._embeddings.pop(min_key, None)

    def invalidate_fuzzy(self, embedding: np.ndarray, threshold: float = 0.7):
        """Invalide les entrées similaires (cosine > threshold)"""
        if not self._embeddings or embedding is None:
            return

        to_remove = []
        for key, cached_emb in self._embeddings.items():
            try:
                sim = _cosine(embedding, cached_emb)
                if sim > threshold:
                    to_remove.append(key)
            except Exception:
                continue

        for k in to_remove:
            self._store.pop(k, None)
            self._embeddings.pop(k, None)

    def clear(self):
        """Efface tout le cache"""
        self._store.clear()
        self._embeddings.clear()

    def get_stats(self) -> dict:
        """Statistiques du cache pour monitoring"""
        if not self._store:
            return {"size": 0, "hit_rate": 0, "avg_confidence": 0}

        total_access = sum(item[3] for item in self._store.values())
        avg_confidence = np.mean([item[2].get("confidence", 0.5) for item in self._store.values()])

        return {
            "size": len(self._store),
            "total_access": total_access,
            "avg_confidence": round(avg_confidence, 3),
            "has_embeddings": len(self._embeddings),
        }
