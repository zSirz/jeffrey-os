"""
Unified Memory System - Mémoire de Classe Mondiale pour Jeffrey OS

Architecture hybride :
- Index inversé pour présélection rapide
- Scoring multi-critères (MCDM) pour pertinence
- Explainability totale (XAI)
- Apprentissage actif (fréquence)
- Extensible (embeddings Phase 2)

Version : 1.0.0 (Phase 1 - Foundation) avec 7 micro-tweaks optimisés
"""

from __future__ import annotations

import datetime as dt
import json
import math
import re
import threading
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from typing import Any, Protocol

from .clustering import ClusterEngine

# =========================
# PUBLIC API (STABLE)
# =========================


class UnifiedMemory:
    """
    Système de mémoire unifié pour Jeffrey OS.

    Features Phase 1 :
    - Scoring multi-critères (texte, émotion, temps, fréquence, importance)
    - Explainability (décomposition des scores)
    - Index inversé (performance)
    - Filtres avancés (date, type, émotion, importance)
    - Apprentissage actif (fréquence d'usage)
    - Thread-safe

    API Principale :
    - add_memory(mem: Dict) -> Dict
    - batch_add(memories: List[Dict]) -> int
    - search_memories(user_id, query, **filters) -> List[Dict]
    - get_all_memories(user_id, **pagination) -> List[Dict]
    - stats(user_id) -> Dict
    """

    def __init__(
        self,
        storage: StorageAdapter | None = None,
        enable_vector: bool = True,
        temporal_mode: str = "recent_bias",
        default_limit: int = 10,
        log: Callable[[str], None] | None = None,
    ):
        """
        Initialise le système de mémoire.

        Args:
            storage: Adapter de stockage (InMemoryStorage par défaut)
            enable_vector: Activer index vectoriel si embeddings disponibles
            temporal_mode: "recent_bias"|"stable"|"distant_focus"
            default_limit: Nombre max de résultats par défaut
            log: Fonction de logging custom (optionnel)
        """
        self.log = log or (lambda m: None)
        self.store = storage or InMemoryStorage()
        self.temporal_mode = temporal_mode
        self.default_limit = default_limit

        # Index inversé pour présélection rapide
        self._inv = InvertedIndex()

        # Index vectoriel optionnel (lazy load si embeddings disponibles)
        self._vec = VectorIndex.auto(enable=enable_vector, logger=self.log)

        # Thread safety
        self._lock = threading.RLock()

        # Clustering
        self._cluster = ClusterEngine()
        self._cluster_threshold = 50  # Minimum de mémoires pour activer
        self._cluster_every = 100  # Re-cluster tous les N ajouts
        self._user_add_counter = {}  # user_id -> compteur d'ajouts

        # Learning-to-rank user weights storage
        if not hasattr(self.store, "_meta_by_user"):
            self.store._meta_by_user = {}

        # Warm-up : reconstruction des index depuis le storage
        self._warm_indexes()

    # ---------- API PRINCIPALE ----------

    def add_memory(self, mem: dict[str, Any]) -> dict[str, Any]:
        """
        Ajoute une mémoire au système.

        Args:
            mem: Dictionnaire avec au minimum {"user_id", "content"}
                 Champs optionnels : "type", "emotion", "importance", "tags"

        Returns:
            Mémoire normalisée et persistée avec "id" généré

        Example:
            >>> memory = um.add_memory({
            ...     "user_id": "david",
            ...     "content": "J'adore le jazz",
            ...     "emotion": "joie",
            ...     "importance": 0.7,
            ...     "tags": ["musique", "jazz"]
            ... })
        """
        with self._lock:
            # Normalisation (ajout champs par défaut)
            m = normalize_memory(mem)

            # Persistance
            m = self.store.upsert(m)

            # Indexation
            self._inv.add(m)
            if self._vec:
                self._vec.add(m)

            # Incrémenter compteur et vérifier clustering
            user_id = m["user_id"]
            self._user_add_counter[user_id] = self._user_add_counter.get(user_id, 0) + 1

            # Déclencher re-clustering si nécessaire
            if self._should_recluster(user_id):
                self._recluster_user_async(user_id)

            return m

    def batch_add(self, memories: Iterable[dict[str, Any]]) -> int:
        """
        Ajoute un lot de mémoires (optimisé).

        Args:
            memories: Liste de dictionnaires de mémoires

        Returns:
            Nombre de mémoires ajoutées

        Example:
            >>> count = um.batch_add([
            ...     {"user_id": "david", "content": "Test 1"},
            ...     {"user_id": "david", "content": "Test 2"},
            ... ])
        """
        count = 0
        with self._lock:
            # Normalisation batch
            normalized = [normalize_memory(m) for m in memories]

            # Persistance batch
            saved = self.store.batch_upsert(normalized)

            # Indexation batch
            for m in saved:
                self._inv.add(m)
            if self._vec:
                self._vec.batch_add(saved)

            count = len(saved)
        return count

    def search_memories(
        self,
        user_id: str,
        query: str | list[str] | None = None,
        *,
        queries: list[str] | None = None,  # NOUVEAU: Multi-query
        combine_strategy: str = "union",  # NOUVEAU: union|intersection
        field_boosts: dict[str, float] | None = None,  # NOUVEAU: field boosts
        filters: dict[str, Any] | None = None,
        temporal_weight: str | None = None,
        semantic_search: bool | None = None,  # None = auto-detect
        exact_match_boost: bool = True,
        boost_exact_phrase: float = 0.15,  # NOUVEAU: exact phrase boost value
        min_relevance: float = 0.15,  # Tweak #3: Seuil par défaut raisonnable
        limit: int | None = None,
        explain: bool = True,
        query_emotion: str | None = None,  # Tweak #2: Score émotionnel contextualisé
        cluster_results: bool = False,  # NOUVEAU: Si True, retourne résultats groupés
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """
        Recherche hybride avec scoring multi-critères et features avancées Phase 3.

        Args:
            user_id: ID utilisateur
            query: Requête textuelle (str) ou liste de mots-clés (rétro-compatible)
            queries: Liste de requêtes (nouveau) - prend priorité sur query
            combine_strategy: "union" (OR) | "intersection" (AND) pour multi-query
            field_boosts: Boost pour champs spécifiques {"tags": 0.2, "type:preference": 0.1}
            filters: Filtres structurés (existant)
            temporal_weight: "recent_bias"|"stable"|"distant_focus"
            semantic_search: Activer recherche sémantique (Phase 2)
            exact_match_boost: Boost si mots-clés exacts trouvés
            boost_exact_phrase: Valeur du boost pour phrase exacte
            min_relevance: Score minimum (0.0-1.0)
            limit: Nombre max de résultats
            explain: Inclure explication enrichie des scores
            query_emotion: Émotion de la requête pour scoring contextualisé
            cluster_results: Si True, retourne {"flat": [...], "clusters": {...}}

        Returns:
            Liste de résultats OU dict avec regroupement par cluster si cluster_results=True
            Format enrichi avec contributions, weights, et reasons dans explanation

        Example:
            >>> results = um.search_memories(
            ...     user_id="david",
            ...     queries=["musique", "voyage"],
            ...     combine_strategy="union",
            ...     field_boosts={"tags": 0.2},
            ...     cluster_results=True
            ... )
        """
        limit = limit or self.default_limit
        temporal_mode = temporal_weight or self.temporal_mode

        # Normalisation: queries prend priorité sur query
        if queries is None:
            queries = [query] if query else []

        if not queries:
            queries = [""]  # Query vide mais permet de continuer

        # 1) Charger mémoires du user
        mems = self.store.list_by_user(user_id)

        # 2) Pré-filtrage structuré (date, type, émotion, importance)
        mems = apply_structured_filters(mems, filters)

        # 3) Tokenisation multi-query
        all_tokens = []
        for q in queries:
            tokens = tokenize(" ".join(q) if isinstance(q, list) else (q or ""))[:64]
            all_tokens.extend(tokens)

        # Unique tokens pour éviter doublons
        q_tokens = list(set(all_tokens))

        # 4) Présélection rapide par index inversé (si query textuelle)
        candidate_ids = None
        if q_tokens:
            # Pour multi-query union, utiliser OR logic
            if len(queries) > 1 and combine_strategy == "union":
                candidate_ids = set()
                for q in queries:
                    q_toks = tokenize(" ".join(q) if isinstance(q, list) else (q or ""))[:64]
                    if q_toks:
                        candidate_ids.update(self._inv.lookup(q_toks))
            else:
                candidate_ids = self._inv.lookup(q_tokens)

            if candidate_ids:
                mems = [m for m in mems if m["id"] in candidate_ids]

        # 5) Scoring multi-critères
        # Auto-activer si embeddings disponibles et pas explicitement désactivé
        if semantic_search is None:
            use_semantic = bool(self._vec and self._vec.available)
        else:
            use_semantic = bool(semantic_search and self._vec and self._vec.available)
        results: list[dict[str, Any]] = []

        for m in mems:
            # Calcul du score sémantique si disponible
            sem_score = self._vec.similarity(q_tokens, m) if use_semantic else None

            # Scoring multi-critères avec émotion contextualisée et features Phase 3
            score, detail = self.score_memory(
                memory=m,
                query_tokens=q_tokens,
                user_id=user_id,  # NOUVEAU: pour poids personnalisés
                temporal_mode=temporal_mode,
                exact_match_boost=exact_match_boost,
                boost_exact_phrase=boost_exact_phrase,
                semantic_score=sem_score,
                query_emotion=query_emotion,
                field_boosts=field_boosts,  # NOUVEAU: field boosts
                filters=filters,
            )

            # Filtrage par score minimum
            if score >= min_relevance:
                item = {"memory": m, "relevance": round(score, 4)}
                if explain:
                    item["explanation"] = detail
                results.append(item)

        # 6) Tri par pertinence décroissante + tie-break déterministe (Tweak #1)
        results.sort(
            key=lambda r: (
                r["relevance"],
                r["memory"].get("importance", 0.0),
                r["memory"].get("last_access", ""),
                r["memory"].get("created_at", ""),
            ),
            reverse=True,
        )
        results = results[:limit]

        # 7) Apprentissage : incrémenter fréquence d'usage (Tweak #5: Apprentissage doux)
        for i, r in enumerate(results):
            self._increment_usage(r["memory"]["id"], boost_importance=(i == 0))

        # 8) Si cluster_results demandé, grouper par thème
        if cluster_results:
            return self._group_by_clusters(results)

        return results

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
        mems = self.store.list_by_user(user_id)
        mems.sort(key=lambda m: m.get("created_at", ""), reverse=True)
        page = mems[offset : offset + limit]
        if redact:
            page = [redact_memory(m) for m in page]
        return page

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
        return {
            "storage": self.store.stats(user_id),
            "inverted_index": self._inv.stats(),
            "vector_index": self._vec.stats() if self._vec else {"enabled": False},
        }

    # ---------- NOUVELLES API PHASE 3 ----------

    def get_clusters(self, user_id: str) -> dict[int, dict[str, Any]]:
        """
        Obtenir les clusters thématiques d'un utilisateur.

        Returns:
            {
                cluster_id: {
                    "theme": str,
                    "size": int
                }
            }
        """
        memories = self.store.list_by_user(user_id)

        aggregation = {}
        for m in memories:
            cluster_id = m.get("cluster")
            if cluster_id is None:
                continue

            theme = m.get("cluster_theme", f"cluster_{cluster_id}")

            if cluster_id not in aggregation:
                aggregation[cluster_id] = {"theme": theme, "size": 0}

            aggregation[cluster_id]["size"] += 1

        return aggregation

    def feedback(self, user_id: str, shown_ids: list[str], clicked_ids: list[str]):
        """
        Apprentissage par feedback utilisateur (bandit multi-armed).

        Args:
            user_id: ID de l'utilisateur
            shown_ids: Liste des IDs affichés (ordre: du meilleur au pire)
            clicked_ids: Liste des IDs cliqués/sélectionnés

        Logic:
            - Si click sur top-1: renforce text + temporal
            - Si click plus bas: diminue text/temporal, augmente importance/emotion
            - Clamp [0.05, 0.70] et renormalise
        """
        if not shown_ids or not clicked_ids:
            return

        weights = self._get_user_weights(user_id)

        top1 = shown_ids[0]
        clicked_top = clicked_ids[0]

        lr = 0.05  # Learning rate

        if clicked_top == top1:
            # Bon ranking: renforcer les critères qui ont bien marché
            weights["text"] += lr * 0.5
            weights["temporal"] += lr * 0.2
        else:
            # Mauvais ranking: ajuster
            weights["text"] -= lr * 0.3
            weights["temporal"] -= lr * 0.1
            weights["importance"] += lr * 0.2
            weights["emotion"] += lr * 0.2

        # Clamp dans [0.05, 0.70]
        for k in weights:
            weights[k] = max(0.05, min(0.70, weights[k]))

        # Renormaliser pour sum = 1.0
        total = sum(weights.values())
        for k in weights:
            weights[k] /= total

        self._set_user_weights(user_id, weights)

        self.log(f"[Learning] Updated weights for {user_id}: {weights}")

    def score_memory(
        self,
        memory: dict[str, Any],
        query_tokens: list[str],
        user_id: str,  # NOUVEAU: pour poids personnalisés
        temporal_mode: str = "recent_bias",
        exact_match_boost: bool = True,
        boost_exact_phrase: float = 0.15,
        semantic_score: float | None = None,
        query_emotion: str | None = None,
        field_boosts: dict[str, float] | None = None,
        filters: dict = None,
        **kwargs,
    ) -> tuple[float, dict]:
        """
        Scoring multi-critères enrichi Phase 3 avec poids personnalisés.
        """
        # Récupérer les poids personnalisés
        weights = self._get_user_weights(user_id)

        # 1) Score textuel (overlap)
        m_tokens = tokenize(memory.get("content", ""))
        s_text = overlap_score(query_tokens, m_tokens)

        # Boost si correspondance exacte de phrase
        if exact_match_boost and query_tokens:
            q_str = " ".join(query_tokens)
            if q_str and q_str in (memory.get("content", "").lower()):
                s_text = min(1.0, s_text + boost_exact_phrase)

        # Fusion avec score sémantique si disponible
        SEMANTIC_WEIGHT = 0.6
        LEXICAL_WEIGHT = 0.4
        if semantic_score is not None:
            s_text = clamp(SEMANTIC_WEIGHT * semantic_score + LEXICAL_WEIGHT * s_text)

        # Appliquer field_boosts
        if field_boosts:
            boost_total = 0.0
            for field_spec, boost_value in field_boosts.items():
                # Format: "field" ou "field:value"
                if ":" in field_spec:
                    field_name, expected_value = field_spec.split(":", 1)
                else:
                    field_name = field_spec
                    expected_value = None

                # Vérifier le match
                field_content = memory.get(field_name)
                if field_content is None:
                    continue

                # Pour les listes (tags)
                if isinstance(field_content, list):
                    if any(qt in str(item).lower() for item in field_content for qt in query_tokens):
                        if expected_value is None:
                            boost_total += boost_value

                # Pour les strings
                elif isinstance(field_content, str):
                    matches_value = (expected_value is None) or (field_content == expected_value)
                    matches_query = any(qt in field_content.lower() for qt in query_tokens)

                    if matches_value and matches_query:
                        boost_total += boost_value

            # Ajouter le boost au score textuel
            s_text = min(1.0, s_text + boost_total)

        # 2) Scores autres critères
        s_em = emotion_score(query_emotion, memory.get("emotion"))
        s_tmp = temporal_score(memory.get("created_at", ""), temporal_mode)
        s_frq = frequency_score(memory.get("access_count", 0))
        s_imp = importance_score(memory.get("importance", 0.0))

        # 3) Aggregation avec poids personnalisés
        final_score = (
            weights["text"] * s_text
            + weights["emotion"] * s_em
            + weights["temporal"] * s_tmp
            + weights["frequency"] * s_frq
            + weights["importance"] * s_imp
        )

        # 4) Générer les raisons lisibles
        reasons = []

        # Raison 1: Match exact de phrase
        if boost_exact_phrase > 0 and query_tokens:
            q_str = " ".join(query_tokens)
            if q_str and q_str in memory.get("content", "").lower():
                reasons.append(f"Exact phrase match: '{q_str}'")

        # Raison 2: Match de tags
        matched_tags = [tag for tag in (memory.get("tags", []) or []) if any(qt in tag.lower() for qt in query_tokens)]
        if matched_tags:
            reasons.append(f"Tag matches: {', '.join(matched_tags)}")

        # Raison 3: Récence
        try:
            created = dt.datetime.fromisoformat(memory.get("created_at", "").replace("Z", "+00:00"))
            days_ago = (dt.datetime.now(dt.UTC) - created).days
            if days_ago <= 7:
                reasons.append(f"Recent: {days_ago} day(s) ago")
        except:
            pass

        # Raison 4: Fréquence
        access_count = memory.get("access_count", 0)
        if access_count >= 3:
            reasons.append(f"Frequently accessed: {access_count} times")

        # Raison 5: Importance
        if memory.get("importance", 0) >= 0.7:
            reasons.append(f"High importance: {memory['importance']}")

        # Raison 6: Emotion match
        if filters and "emotion" in filters and memory.get("emotion") == filters.get("emotion"):
            reasons.append(f"Emotion match: {filters['emotion']}")

        # 5) Explication enrichie
        detail = {
            # Existant
            "text_score": round(s_text, 3),
            "emotion_score": round(s_em, 3),
            "temporal_score": round(s_tmp, 3),
            "frequency_score": round(s_frq, 3),
            "importance_score": round(s_imp, 3),
            # NOUVEAU: Contributions par critère
            "criterion_contributions": {
                "text": round(weights["text"] * s_text, 3),
                "emotion": round(weights["emotion"] * s_em, 3),
                "temporal": round(weights["temporal"] * s_tmp, 3),
                "frequency": round(weights["frequency"] * s_frq, 3),
                "importance": round(weights["importance"] * s_imp, 3),
            },
            # NOUVEAU: Poids utilisés
            "weights_used": weights,
            # NOUVEAU: Raisons lisibles
            "reasons": reasons,
            # NOUVEAU: Si clustering activé
            "cluster_theme": memory.get("cluster_theme"),
            # Ancien main_reason pour rétro-compatibilité
            "main_reason": max(
                [("text", s_text), ("emotion", s_em), ("temporal", s_tmp), ("frequency", s_frq), ("importance", s_imp)],
                key=lambda x: x[1],
            )[0],
        }

        if semantic_score is not None:
            detail["semantic_score"] = round(semantic_score, 3)

        return float(final_score), detail

    # ---------- MÉTHODES INTERNES ----------

    def _warm_indexes(self):
        """Reconstruction des index en mémoire depuis le storage (au boot)."""
        with self._lock:
            all_mems = self.store.list_all()
            self._inv.rebuild(all_mems)
            if self._vec:
                self._vec.rebuild(all_mems)
            self.log(f"[UnifiedMemory] Indexes warmed: {len(all_mems)} memories")

    def _increment_usage(self, mem_id: str, boost_importance: bool = False):
        """
        Apprentissage simple : augmente access_count et met à jour last_access.
        Tweak #5: Apprentissage doux avec boost d'importance pour le Top-1.
        Hook appelé automatiquement après chaque recherche réussie.
        """
        try:
            m = self.store.get(mem_id)
            if not m:
                return
            m["access_count"] = int(m.get("access_count", 0)) + 1

            # Boost d'importance pour le Top-1 (proxy "clicked/utile")
            if boost_importance:
                m["importance"] = clamp(float(m.get("importance", 0.0)) + 0.02)

            m["last_access"] = now_iso()
            self.store.upsert(m)
        except Exception as e:
            # Best-effort : ne pas casser la recherche si le hook échoue
            self.log(f"[Warning] Failed to increment usage for {mem_id}: {e}")

    # ---------- MÉTHODES CLUSTERING ----------

    def _should_recluster(self, user_id: str) -> bool:
        """Décide si on doit re-clusterer cet utilisateur."""
        count = self._user_add_counter.get(user_id, 0)
        total = len(self.store.list_by_user(user_id))

        # Première fois: quand on atteint le seuil
        if total >= self._cluster_threshold and count >= self._cluster_threshold:
            return True

        # Ensuite: tous les N ajouts
        if count >= self._cluster_every:
            return True

        return False

    def _recluster_user_async(self, user_id: str):
        """Lance le clustering en arrière-plan (best-effort)."""

        def _worker():
            try:
                self._recluster_user(user_id)
            except Exception as e:
                self.log(f"[Cluster] Async clustering failed: {e}")

        # Thread daemon (ne bloque pas l'arrêt)
        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def _recluster_user(self, user_id: str):
        """
        Re-clusterer tous les souvenirs d'un utilisateur.
        Met à jour les champs 'cluster' et 'cluster_theme' dans le store.
        """
        try:
            # Récupérer toutes les mémoires
            memories = self.store.list_by_user(user_id)

            if len(memories) < self._cluster_threshold:
                return

            # Clustering
            mapping, themes = self._cluster.fit_user(memories)

            if not mapping:
                return  # Clustering disabled ou échoué

            # Mettre à jour le store avec thread safety
            with self._lock:
                for m in memories:
                    mem_id = m["id"]
                    cluster_id = mapping.get(mem_id)

                    if cluster_id is not None:
                        m["cluster"] = cluster_id
                        if themes:
                            m["cluster_theme"] = themes.get(cluster_id, f"cluster_{cluster_id}")
                        self.store.upsert(m)

            # Reset compteur
            self._user_add_counter[user_id] = 0

            self.log(f"[Cluster] Updated {len(mapping)} memories for {user_id}")

        except Exception as e:
            self.log(f"[Cluster] recluster_user failed for {user_id}: {e}")

    def _group_by_clusters(self, results: list[dict]) -> dict:
        """
        Regroupe les résultats par cluster thématique.

        Returns:
            {
                "flat": [...],  # Tous les résultats triés
                "clusters": {
                    "theme1": [...],
                    "theme2": [...]
                }
            }
        """
        grouped = {}

        for result in results:
            theme = result["memory"].get("cluster_theme")

            if theme:
                if theme not in grouped:
                    grouped[theme] = []
                grouped[theme].append(result)

        return {"flat": results, "clusters": grouped}

    # ---------- MÉTHODES LEARNING-TO-RANK ----------

    def _get_user_weights(self, user_id: str) -> dict[str, float]:
        """
        Récupère les poids de scoring personnalisés pour un utilisateur.
        Fallback sur les poids par défaut.
        """
        # Accès au meta store
        if not hasattr(self.store, "_meta_by_user"):
            self.store._meta_by_user = {}

        meta = self.store._meta_by_user.get(user_id, {})
        weights = meta.get("weights")

        if not weights:
            # Poids par défaut
            weights = {"text": 0.40, "emotion": 0.20, "temporal": 0.20, "frequency": 0.10, "importance": 0.10}

        return weights.copy()

    def _set_user_weights(self, user_id: str, weights: dict[str, float]):
        """Sauvegarde les poids de scoring personnalisés."""
        if not hasattr(self.store, "_meta_by_user"):
            self.store._meta_by_user = {}

        if user_id not in self.store._meta_by_user:
            self.store._meta_by_user[user_id] = {}

        self.store._meta_by_user[user_id]["weights"] = weights


# =========================
# STORAGE ADAPTERS
# =========================


class StorageAdapter(Protocol):
    """Interface pour adapter différents backends (Redis, PostgreSQL, etc.)"""

    def upsert(self, memory: dict[str, Any]) -> dict[str, Any]: ...
    def batch_upsert(self, memories: Iterable[dict[str, Any]]) -> list[dict[str, Any]]: ...
    def get(self, mem_id: str) -> dict[str, Any] | None: ...
    def list_all(self) -> list[dict[str, Any]]: ...
    def list_by_user(self, user_id: str) -> list[dict[str, Any]]: ...
    def stats(self, user_id: str | None = None) -> dict[str, Any]: ...


class InMemoryStorage:
    """
    Stockage simple en RAM avec auto-génération d'ID.
    Remplacez par Redis/PostgreSQL pour production.
    """

    def __init__(self):
        self._d: dict[str, dict[str, Any]] = {}
        self._by_user: dict[str, set[str]] = defaultdict(set)
        self._lock = threading.RLock()
        self._next = 1

    def upsert(self, memory: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if not memory.get("id"):
                memory["id"] = f"m_{self._next}"
                self._next += 1
            self._d[memory["id"]] = memory
            self._by_user[memory["user_id"]].add(memory["id"])
            return memory

    def batch_upsert(self, memories: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        saved = []
        with self._lock:
            for m in memories:
                if not m.get("id"):
                    m["id"] = f"m_{self._next}"
                    self._next += 1
                self._d[m["id"]] = m
                self._by_user[m["user_id"]].add(m["id"])
                saved.append(m)
        return saved

    def get(self, mem_id: str) -> dict[str, Any] | None:
        return self._d.get(mem_id)

    def list_all(self) -> list[dict[str, Any]]:
        return list(self._d.values())

    def list_by_user(self, user_id: str) -> list[dict[str, Any]]:
        ids = self._by_user.get(user_id, set())
        return [self._d[i] for i in ids if i in self._d]

    def stats(self, user_id: str | None = None) -> dict[str, Any]:
        if user_id:
            n = len(self._by_user.get(user_id, []))
            return {"user_id": user_id, "count": n}
        return {"total": len(self._d), "users": len(self._by_user)}


# =========================
# INDEXES
# =========================

_WORD = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tokenisation simple par regex Unicode."""
    return [t.lower() for t in _WORD.findall(text or "") if t]


class InvertedIndex:
    """
    Index inversé pour présélection rapide.
    Complexité : O(log n) lookup sur tokens.
    """

    def __init__(self):
        self._postings: dict[str, set[str]] = defaultdict(set)
        self._lock = threading.RLock()

    def rebuild(self, memories: Iterable[dict[str, Any]]):
        """Reconstruit l'index depuis une liste de mémoires."""
        with self._lock:
            self._postings.clear()
            for m in memories:
                self.add(m)

    def add(self, memory: dict[str, Any]):
        """Indexe une mémoire (content + tags)."""
        content = f"{memory.get('content', '')} {' '.join(memory.get('tags', []) or [])}"
        for tok in set(tokenize(content)):
            self._postings[tok].add(memory["id"])

    def lookup(self, q_tokens: list[str]) -> set[str]:
        """
        Recherche par tokens.
        Stratégie : AND (intersection) puis fallback à OR (union) si vide.
        """
        if not q_tokens:
            return set()

        sets = [self._postings.get(t, set()) for t in set(q_tokens)]
        if not sets:
            return set()

        # Intersection (AND sémantique)
        res = sets[0].copy()
        for s in sets[1:]:
            res &= s

        # Si vide, fallback à UNION (tolérant)
        return res if res else set().union(*sets)

    def stats(self) -> dict[str, Any]:
        return {"terms": len(self._postings)}


class VectorIndex:
    """
    Index vectoriel optionnel pour recherche sémantique (Phase 2).
    S'active automatiquement si sentence-transformers disponible.
    Fallback silencieux sinon.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", logger: Callable[[str], None] | None = None):
        self.logger = logger or (lambda m: None)
        self.available = False
        self._lock = threading.RLock()
        self._model = None
        self._mem_emb: dict[str, list[float]] = {}

        # Cache disque pour cold-start rapide
        from pathlib import Path

        self._cache_dir = Path(".cache/unified_memory") / model_name.replace("/", "_")
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._map_path = self._cache_dir / "id_to_vec.json"

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
            self.available = True
            self.logger(f"[VectorIndex] Loaded: {model_name}")

            # Charger le cache au démarrage
            self._load_cache()
        except Exception as e:
            self.logger(f"[VectorIndex] Disabled (embeddings unavailable): {e}")

    @classmethod
    def auto(cls, enable: bool, logger: Callable[[str], None] | None = None) -> VectorIndex | None:
        """Factory method avec lazy loading."""
        if not enable:
            return None
        return cls(logger=logger)

    def _load_cache(self):
        """Charge les embeddings depuis le cache disque."""
        try:
            if self._map_path.exists():
                import json

                with open(self._map_path, encoding="utf-8") as f:
                    self._mem_emb = json.load(f)
                self.logger(f"[VectorIndex] Cache loaded: {len(self._mem_emb)} vectors")
        except Exception as e:
            self.logger(f"[VectorIndex] Cache load failed: {e}")

    def _save_cache(self):
        """Sauvegarde les embeddings dans le cache disque."""
        try:
            import json

            with open(self._map_path, "w", encoding="utf-8") as f:
                json.dump(self._mem_emb, f)
            self.logger(f"[VectorIndex] Cache saved: {len(self._mem_emb)} vectors")
        except Exception as e:
            self.logger(f"[VectorIndex] Cache save failed: {e}")

    def rebuild(self, memories: Iterable[dict[str, Any]]):
        """Reconstruit les embeddings pour toutes les mémoires."""
        if not self.available:
            return

        with self._lock:
            self._mem_emb.clear()
            corpus = []
            ids = []
            for m in memories:
                text = embed_text_of(m)
                corpus.append(text)
                ids.append(m["id"])

            if not corpus:
                return

            vecs = self._model.encode(corpus, normalize_embeddings=True)
            for i, vid in enumerate(ids):
                self._mem_emb[vid] = vecs[i].tolist()

            # Sauvegarder le cache
            self._save_cache()

    def add(self, memory: dict[str, Any]):
        """Ajoute l'embedding d'une mémoire."""
        if not self.available:
            return

        with self._lock:
            text = embed_text_of(memory)
            vec = self._model.encode([text], normalize_embeddings=True)[0]
            self._mem_emb[memory["id"]] = vec.tolist()

            # Sauvegarder le cache
            self._save_cache()

    def batch_add(self, memories: Iterable[dict[str, Any]]):
        """Ajoute les embeddings d'un lot de mémoires."""
        if not self.available:
            return

        # Guardrail : limiter la taille du batch pour éviter OOM
        MAX_BATCH = 1024
        memories_list = list(memories)

        with self._lock:
            for batch_start in range(0, len(memories_list), MAX_BATCH):
                batch = memories_list[batch_start : batch_start + MAX_BATCH]

                corpus, ids = [], []
                for m in batch:
                    corpus.append(embed_text_of(m))
                    ids.append(m["id"])

                if not corpus:
                    continue

                vecs = self._model.encode(corpus, normalize_embeddings=True)
                for i, vid in enumerate(ids):
                    self._mem_emb[vid] = vecs[i].tolist()

            # Sauvegarder après tout le batch
            self._save_cache()

    def similarity(self, q_tokens: list[str], memory: dict[str, Any]) -> float | None:
        """
        Calcule la similarité cosinus entre query et mémoire.
        Returns None si embeddings indisponibles.
        """
        if not self.available:
            return None

        try:
            q = " ".join(q_tokens)
            qv = self._model.encode([q], normalize_embeddings=True)[0]

            mv = self._mem_emb.get(memory["id"])
            if mv is None:
                # Lazy compute si embedding manquant
                mv = self._model.encode([embed_text_of(memory)], normalize_embeddings=True)[0]
                self._mem_emb[memory["id"]] = mv.tolist()

            return cosine(qv, mv)
        except Exception:
            return None

    def stats(self) -> dict[str, Any]:
        return {"enabled": self.available, "vectors": len(self._mem_emb)}


def embed_text_of(m: dict[str, Any]) -> str:
    """Extrait le texte à embedder d'une mémoire (content + tags)."""
    tags = " ".join(m.get("tags", []) or [])
    return f"{m.get('content', '')} {tags}".strip()


# =========================
# SCORING ENGINE (MCDM)
# =========================


def overlap_score(q_tokens: list[str], m_tokens: list[str]) -> float:
    """Score de chevauchement normalisé (BM25-lite)."""
    if not q_tokens or not m_tokens:
        return 0.0
    q = Counter(q_tokens)
    m = Counter(m_tokens)
    inter = sum((q & m).values())
    return inter / math.sqrt(len(q_tokens) * len(m_tokens))


def emotion_score(query_emotion: str | None, mem_emotion: str | None) -> float:
    """Score émotionnel (1.0 si match exact, 0.5 si neutre)."""
    if not query_emotion or query_emotion == "neutre":
        return 0.5
    return 1.0 if (str(mem_emotion or "neutre").lower() == query_emotion.lower()) else 0.0


def temporal_score(created_iso: str, mode: str = "recent_bias") -> float:
    """
    Score temporel avec stratégies multiples.

    Modes :
    - recent_bias : Privilégie les souvenirs récents (décroissance exponentielle)
    - distant_focus : Privilégie les souvenirs anciens
    - stable : Pas de biais temporel
    """
    try:
        created = dt.datetime.fromisoformat(created_iso.replace("Z", "+00:00"))
        age_days = max(0.0, (dt.datetime.now(dt.UTC) - created).total_seconds() / 86400.0)
    except Exception:
        return 0.5

    if mode == "recent_bias":
        return 1.0 / (1.0 + 0.1 * age_days)
    elif mode == "distant_focus":
        return min(age_days / 30.0, 1.0)
    else:  # stable
        return 1.0


def frequency_score(access_count: int) -> float:
    """Score de fréquence (scaling logarithmique)."""
    return min(1.0, math.log1p(max(0, access_count)) / math.log(10))


def importance_score(importance: Any) -> float:
    """Score d'importance (clamp 0.0-1.0)."""
    try:
        v = float(importance or 0.0)
        return clamp(v)
    except Exception:
        return 0.0


def cosine(a, b) -> float:
    """Similarité cosinus entre deux vecteurs."""
    try:
        import numpy as np

        a = np.asarray(a)
        b = np.asarray(b)
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
        return float(np.dot(a, b) / denom)
    except ImportError:
        # Fallback si numpy absent
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(y * y for y in b))
        return dot / (mag_a * mag_b) if (mag_a * mag_b) > 0 else 0.0


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp une valeur entre lo et hi."""
    return max(lo, min(hi, x))


# =========================
# FILTERS & NORMALIZATION
# =========================


def apply_structured_filters(mems: list[dict[str, Any]], filters: dict[str, Any] | None) -> list[dict[str, Any]]:
    """
    Applique des filtres structurés sur métadonnées.

    Filtres supportés :
    - emotion : ["joie", "neutre"]
    - memory_types : ["fact", "preference"]
    - min_importance : 0.5
    - date_range : ("2025-01-01", "2025-12-31")
    """
    if not filters:
        return mems

    out = mems

    # Filtre émotion
    if "emotion" in filters:
        allowed = set([str(e).lower() for e in filters["emotion"]])
        out = [m for m in out if str(m.get("emotion", "neutre")).lower() in allowed]

    # Filtre type
    if "memory_types" in filters:
        allowed = set([str(t).lower() for t in filters["memory_types"]])
        out = [m for m in out if str(m.get("type", "")).lower() in allowed]

    # Filtre importance minimum
    if "min_importance" in filters:
        th = float(filters["min_importance"])
        out = [m for m in out if importance_score(m.get("importance")) >= th]

    # Filtre date range
    if "date_range" in filters:
        start_iso, end_iso = filters["date_range"]
        start = parse_iso(start_iso) if start_iso else None
        end = parse_iso(end_iso) if end_iso else None

        def in_range(iso: str) -> bool:
            try:
                t = parse_iso(iso)
                if start and t < start:
                    return False
                if end and t > end:
                    return False
                return True
            except Exception:
                return False

        out = [m for m in out if in_range(m.get("created_at", ""))]

    return out


def normalize_memory(m: dict[str, Any]) -> dict[str, Any]:
    """
    Normalise une mémoire (ajout champs par défaut).

    Schéma complet :
    - id : auto-généré
    - user_id : "default"
    - type : "note"
    - content : ""
    - emotion : "neutre"
    - importance : 0.0
    - created_at : now()
    - last_access : created_at
    - access_count : 0
    - tags : []
    - meta : {}
    """
    nm = dict(m)
    nm.setdefault("user_id", "default")
    nm.setdefault("type", "note")
    nm.setdefault("content", "")
    nm.setdefault("emotion", "neutre")
    nm.setdefault("importance", 0.0)
    nm.setdefault("created_at", now_iso())
    nm.setdefault("last_access", nm["created_at"])
    nm.setdefault("access_count", 0)
    nm.setdefault("tags", [])
    nm.setdefault("meta", {})
    return nm


def redact_memory(m: dict[str, Any]) -> dict[str, Any]:
    """Masque le contenu complet (GDPR/privacy)."""
    rd = {k: v for k, v in m.items() if k not in ("content", "meta")}
    content = m.get("content", "")
    if len(content) > 120:
        rd["content_preview"] = content[:120] + "…"
    else:
        rd["content_preview"] = content
    return rd


def now_iso() -> str:
    """Timestamp ISO 8601 UTC."""
    return dt.datetime.now(dt.UTC).isoformat()


def parse_iso(s: str) -> dt.datetime:
    """Parse un timestamp ISO 8601."""
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


# =========================
# SELF-TEST (OPTIONNEL)
# =========================

if __name__ == "__main__":
    """Test de validation rapide."""
    print("=== UnifiedMemory Self-Test ===\n")

    # Init
    um = UnifiedMemory()
    uid = "david"

    # Ajout batch
    um.batch_add(
        [
            {
                "user_id": uid,
                "content": "J'adore la musique jazz et les concerts.",
                "emotion": "joie",
                "importance": 0.7,
                "tags": ["musique", "jazz"],
            },
            {
                "user_id": uid,
                "content": "Le projet Jeffrey OS avance, j'ai optimisé le bridge émotionnel.",
                "type": "fact",
                "importance": 0.9,
                "tags": ["projet", "jeffrey"],
            },
            {
                "user_id": uid,
                "content": "Hier j'étais stressé par la deadline, mais ça va mieux.",
                "emotion": "peur",
                "importance": 0.4,
                "tags": ["travail"],
            },
            {
                "user_id": uid,
                "content": "Je veux voyager au Japon l'an prochain.",
                "type": "preference",
                "importance": 0.8,
                "tags": ["voyage"],
            },
        ]
    )

    # Test 1 : Recherche simple
    print("== Test 1 : Search 'jazz' ==")
    res = um.search_memories(uid, query="jazz", semantic_search=False, explain=True, limit=5)
    print(json.dumps(res, ensure_ascii=False, indent=2))

    # Test 2 : Filtres avancés
    print("\n== Test 2 : Filters (type=preference, min_importance=0.7) ==")
    res = um.search_memories(uid, query=None, filters={"memory_types": ["preference"], "min_importance": 0.7})
    print(json.dumps(res, ensure_ascii=False, indent=2))

    # Test 3 : Stats
    print("\n== Test 3 : Stats ==")
    print(json.dumps(um.stats(uid), ensure_ascii=False, indent=2))

    print("\n✅ Self-test passed!")
