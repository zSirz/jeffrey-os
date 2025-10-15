"""
Memory Bridge - Pont entre l'ancien et le nouveau système de mémoire

Ce module fournit une couche de compatibilité qui permet à l'orchestrateur
d'utiliser le nouveau système UnifiedMemory tout en gardant l'API existante.

Architecture :
- MemoryBridge : Classe principale qui délègue au nouveau UnifiedMemory
- Singleton global : Une seule instance partagée partout
- Pass-through explicite : Toutes les méthodes publiques exposées
- Fallback automatique : __getattr__ pour méthodes non définies
"""

import logging
from collections.abc import Iterable
from typing import Any

from .unified_memory import UnifiedMemory as NewUnifiedMemory

logger = logging.getLogger(__name__)

# Singleton global pour le bridge
_bridge_singleton = None


class MemoryBridge:
    """
    Pont de compatibilité : expose l'API attendue par l'orchestrateur,
    en la déléguant au nouveau UnifiedMemory.

    Toutes les méthodes sont forwardées au système NewUnifiedMemory.
    L'instance est partagée via un singleton global.
    """

    def __init__(self):
        """Initialise le bridge avec une instance du nouveau UnifiedMemory."""
        self._new = NewUnifiedMemory(
            enable_vector=True,  # S'activera auto si embeddings présents
            temporal_mode="recent_bias",
            default_limit=10,
            log=lambda m: logger.debug(m),
        )
        logger.info("✅ MemoryBridge initialisé — nouveau UnifiedMemory actif")

    # ==========================================
    # PASS-THROUGH EXPLICITE (API PUBLIQUE)
    # ==========================================

    def add_memory(self, mem: dict[str, Any]) -> dict[str, Any]:
        """
        Ajoute une mémoire au système.

        Args:
            mem: Dictionnaire avec au minimum {"user_id", "content"}

        Returns:
            Mémoire normalisée et persistée avec "id" généré
        """
        return self._new.add_memory(mem)

    def batch_add(self, memories: Iterable[dict[str, Any]]) -> int:
        """
        Ajoute un lot de mémoires (optimisé).

        Args:
            memories: Liste de dictionnaires de mémoires

        Returns:
            Nombre de mémoires ajoutées
        """
        return self._new.batch_add(memories)

    def search_memories(
        self,
        user_id: str,
        query: str | list[str] | None = None,
        *,
        filters: dict[str, Any] | None = None,
        temporal_weight: str | None = None,
        semantic_search: bool = False,
        exact_match_boost: bool = True,
        min_relevance: float = 0.15,  # Défaut raisonnable (Tweak #3)
        limit: int | None = None,
        explain: bool = True,
        query_emotion: str | None = None,  # Tweak #2
    ) -> list[dict[str, Any]]:
        """
        Recherche hybride avec scoring multi-critères.

        Args:
            user_id: ID utilisateur
            query: Requête textuelle (str) ou liste de mots-clés
            filters: Filtres structurés (emotion, date_range, memory_types, min_importance)
            temporal_weight: "recent_bias"|"stable"|"distant_focus"
            semantic_search: Activer recherche sémantique (Phase 2)
            exact_match_boost: Boost si mots-clés exacts trouvés
            min_relevance: Score minimum (0.0-1.0)
            limit: Nombre max de résultats
            explain: Inclure explication des scores
            query_emotion: Émotion du contexte query (Tweak #2)

        Returns:
            Liste de {"memory": {...}, "relevance": 0.85, "explanation": {...}}
        """
        return self._new.search_memories(
            user_id=user_id,
            query=query,
            filters=filters,
            temporal_weight=temporal_weight,
            semantic_search=semantic_search,
            exact_match_boost=exact_match_boost,
            min_relevance=min_relevance,
            limit=limit,
            explain=explain,
            query_emotion=query_emotion,
        )

    def get_all_memories(
        self, user_id: str, *, offset: int = 0, limit: int = 1000, redact: bool = False
    ) -> list[dict[str, Any]]:
        """
        Export paginé de toutes les mémoires d'un user.
        Utile pour GDPR, backup, debug.

        Args:
            user_id: ID utilisateur
            offset: Index de départ (pagination)
            limit: Nombre max de résultats
            redact: Masquer le contenu complet (preview uniquement)

        Returns:
            Liste de mémoires triées par date décroissante
        """
        return self._new.get_all_memories(user_id=user_id, offset=offset, limit=limit, redact=redact)

    def stats(self, user_id: str | None = None) -> dict[str, Any]:
        """
        Statistiques du système de mémoire.

        Args:
            user_id: Stats pour un user spécifique (ou global si None)

        Returns:
            {
                "storage": {"total": 150, "users": 5},
                "inverted_index": {"terms": 487},
                "vector_index": {"enabled": True, "vectors": 150}
            }
        """
        return self._new.stats(user_id)

    # ==========================================
    # COMPATIBILITÉ AVEC ANCIEN SYSTÈME
    # ==========================================

    def save_fact(self, user_id: str, category: str, fact: str):
        """
        Sauvegarde un fait (compatibilité API ancienne).

        Args:
            user_id: ID utilisateur
            category: Catégorie du fait (ex: "animal_chien")
            fact: Contenu du fait (ex: "Max")
        """
        self._new.add_memory(
            {
                "user_id": user_id,
                "content": f"{category}: {fact}",
                "type": "fact",
                "tags": [category],
                "importance": 0.7,
            }
        )
        logger.debug(f"[Compat] Saved fact for {user_id}: {category} = {fact}")

    def get_emotional_summary(self, user_id: str) -> dict[str, Any]:
        """
        Résumé émotionnel (compatibilité API ancienne).

        Returns:
            {
                "dominant_emotions": ["joie", "neutre"],
                "emotional_stability": 0.8,
                "recent_mood": "joie",
                "relationship_depth": 0.5
            }
        """
        # Récupère les mémoires récentes
        recent = self._new.search_memories(user_id=user_id, query=None, temporal_weight="recent_bias", limit=20)

        # Extrait les émotions
        from collections import Counter

        emotions = [r["memory"].get("emotion", "neutre") for r in recent if r["memory"].get("emotion")]

        emotion_counts = Counter(emotions)
        dominant = [e for e, _ in emotion_counts.most_common(3)]

        return {
            "dominant_emotions": dominant,
            "emotional_stability": 0.7,  # Placeholder
            "recent_mood": dominant[0] if dominant else "neutre",
            "relationship_depth": 0.5,  # Placeholder
        }

    def get_context_summary(self) -> str:
        """
        Résumé textuel du contexte (compatibilité API ancienne).

        Returns:
            Texte résumant le contexte actuel
        """
        # Pour l'instant, retourne un placeholder
        return "Contexte chargé depuis le nouveau système UnifiedMemory."

    # ==========================================
    # MÉTHODES LEGACY POUR ORCHESTRATEUR
    # ==========================================

    def update(self, message: str, emotion_state: dict, metadata: dict = None):
        """
        Met à jour la mémoire avec un nouveau message (ancienne API).

        Args:
            message: Le message de l'utilisateur ou de Jeffrey
            emotion_state: État émotionnel actuel
            metadata: Métadonnées additionnelles (user_id, timestamp, etc.)
        """
        metadata = metadata or {}
        user_id = metadata.get("user_id", "default")

        # Convertir vers le nouveau format
        memory_data = {
            "user_id": user_id,
            "content": message,
            "emotion": emotion_state.get("primary_emotion", "neutre"),
            "type": "conversation",
            "importance": min(1.0, emotion_state.get("intensity", 0.5)),
            "meta": metadata,
        }

        # Ajouter au nouveau système
        self._new.add_memory(memory_data)

    def retrieve(self, query_type: str = "all", limit: int = 10) -> list[dict]:
        """
        Récupère des informations de la mémoire (ancienne API).

        Args:
            query_type: Type de requête (all, context, emotional, patterns)
            limit: Nombre maximum de résultats

        Returns:
            Liste des entrées mémoire pertinentes
        """
        # Convertir vers recherche du nouveau système
        if query_type == "context":
            # Recherche récente sans query spécifique
            results = self._new.search_memories(
                user_id="default", query=None, temporal_weight="recent_bias", limit=limit
            )
        elif query_type == "emotional":
            # Recherche par émotions fortes
            results = self._new.search_memories(
                user_id="default", query=None, filters={"min_importance": 0.6}, limit=limit
            )
        else:  # "all" ou autres
            # Recherche générale
            results = self._new.search_memories(user_id="default", query=None, limit=limit)

        # Convertir vers l'ancien format attendu
        old_format = []
        for r in results:
            memory = r["memory"]
            old_format.append(
                {
                    "message": memory.get("content", ""),
                    "emotion": {"primary_emotion": memory.get("emotion", "neutre")},
                    "timestamp": memory.get("created_at", ""),
                    "metadata": memory.get("meta", {}),
                }
            )

        return old_format

    def update_relationship(self, user_id: str, interaction_quality: float):
        """Met à jour la relation avec un utilisateur (ancienne API)."""
        # Sauvegarder comme métadonnée dans le nouveau système
        memory_data = {
            "user_id": user_id,
            "content": f"Qualité d'interaction: {interaction_quality}",
            "type": "relationship",
            "importance": interaction_quality,
            "meta": {"interaction_quality": interaction_quality},
        }

        self._new.add_memory(memory_data)

    def get_all_memories_legacy(self, user_id: str) -> list[str]:
        """Récupère tous les souvenirs d'un utilisateur en format simple (ancienne API)."""
        # Utiliser le nouveau système pour récupérer
        all_memories = self._new.get_all_memories(user_id=user_id, limit=20)

        # Convertir en format simple
        return [m["content"] for m in all_memories]

    def save_persistent_data(self):
        """Sauvegarde pour compatibilité (le nouveau système auto-sauvegarde)."""
        pass  # Le nouveau système gère automatiquement la persistance

    def clear_old_data(self, days: int = 30):
        """Nettoyage pour compatibilité."""
        logger.info(f"Clear old data appelé (days={days}) - géré automatiquement")

    # ==========================================
    # NOUVELLES FONCTIONNALITÉS AVANCÉES
    # ==========================================

    def advanced_search(
        self,
        user_id: str,
        query: str,
        query_emotion: str | None = None,
        filters: dict[str, Any] | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """
        Recherche avancée avec toutes les nouvelles fonctionnalités.

        Cette méthode expose directement l'API du nouveau système.
        """
        return self._new.search_memories(
            user_id=user_id, query=query, query_emotion=query_emotion, filters=filters, **kwargs
        )

    def get_advanced_stats(self, user_id: str | None = None) -> dict[str, Any]:
        """Statistiques avancées du nouveau système."""
        return self._new.stats(user_id)

    def batch_add_memories(self, memories: list[dict[str, Any]]) -> int:
        """Ajout en lot optimisé."""
        return self._new.batch_add(memories)

    # ==========================================
    # FALLBACK AUTOMATIQUE
    # ==========================================

    def __getattr__(self, name: str):
        """
        Si une méthode n'est pas définie ci-dessus, on délègue au nouveau système.
        Permet d'ajouter de nouvelles méthodes sans modifier le bridge.
        """
        return getattr(self._new, name)


# ==========================================
# SINGLETON GLOBAL
# ==========================================


def get_unified_memory() -> MemoryBridge:
    """
    Fournit l'instance singleton du bridge (qui lui-même encapsule le nouveau système).

    Important : L'orchestrateur, les tests et le life-sim partagent la MÊME instance.
    Cela garantit que tous utilisent le même système de mémoire.

    Returns:
        Instance unique de MemoryBridge
    """
    global _bridge_singleton
    if _bridge_singleton is None:
        _bridge_singleton = MemoryBridge()
    return _bridge_singleton


def reset_unified_memory():
    """Réinitialise le singleton (utile pour les tests)."""
    global _bridge_singleton
    _bridge_singleton = None
    logger.info("🔄 MemoryBridge singleton réinitialisé")


# ==========================================
# ALIAS POUR COMPATIBILITÉ
# ==========================================

# Alias pour l'ancien nom (si utilisé quelque part)
UnifiedMemory = MemoryBridge

# Export explicite pour clarté
__all__ = [
    "MemoryBridge",
    "get_unified_memory",
    "reset_unified_memory",
    "UnifiedMemory",  # alias
]
