"""
Système de mémoire à 3 niveaux inspiré du cerveau humain
Working Memory → Episodic Memory → Semantic Memory
"""

import asyncio
import hashlib
import json
import pickle
import sqlite3
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Pour les embeddings et recherche vectorielle
try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    logger.warning("FAISS not available, using numpy fallback")
    FAISS_AVAILABLE = False

from datasketch import MinHash, MinHashLSH
from sklearn.feature_extraction.text import HashingVectorizer

from jeffrey.utils.logger import get_logger

logger = get_logger("TripleMemory")


@dataclass
class Episode:
    """Une expérience/interaction mémorisée"""

    id: str
    timestamp: float
    intent: str
    text_in: str
    text_out: str
    embedding_in: np.ndarray | None
    embedding_out: np.ndarray | None
    quality_score: float
    tags: list[str]
    metadata: dict[str, Any]


class LocalEmbedder:
    """Embedder local simple sans dépendance externe"""

    def __init__(self, dim=384):
        self.vec = HashingVectorizer(n_features=dim, alternate_sign=False, norm="l2")

    def encode(self, text: str) -> np.ndarray:
        """Encode un texte en vecteur"""
        return self.vec.transform([text]).toarray()[0].astype("float32")


class WorkingMemory:
    """
    Mémoire de travail volatile (RAM)
    Garde les N derniers items pour contexte immédiat
    """

    def __init__(self, capacity: int = 128, ttl_seconds: float = 120):
        self.capacity = capacity
        self.ttl = ttl_seconds
        self.items = deque(maxlen=capacity)
        self.timestamps = deque(maxlen=capacity)

    def add(self, item: Any):
        """Ajoute un item à la mémoire de travail"""
        self.items.append(item)
        self.timestamps.append(time.time())
        self._cleanup()

    def _cleanup(self):
        """Supprime les items expirés"""
        current_time = time.time()
        while self.timestamps and (current_time - self.timestamps[0]) > self.ttl:
            self.items.popleft()
            self.timestamps.popleft()

    def get_recent(self, n: int = 10) -> list[Any]:
        """Récupère les N items les plus récents"""
        self._cleanup()
        return list(self.items)[-n:]

    def clear(self):
        """Vide la mémoire de travail"""
        self.items.clear()
        self.timestamps.clear()


class EpisodicMemory:
    """
    Mémoire épisodique avec index vectoriel
    Stockage SQLite + FAISS/LSH pour recherche rapide
    """

    def __init__(self, db_path: str = "data/episodic.db", dim: int = 384):
        self.db_path = db_path
        self.dim = dim

        # Créer le dossier si nécessaire
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        # SQLite pour métadonnées
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

        # FAISS pour recherche vectorielle (avec fallback NumPy)
        if FAISS_AVAILABLE:
            self.index = faiss.IndexFlatL2(dim)  # L2 distance
            self.index = faiss.IndexIDMap(self.index)  # Pour mapping ID
        else:
            self.index = None
            self.vectors = {}  # Fallback: dict of id -> vector

        # LSH pour recherche approximative rapide
        self.lsh = MinHashLSH(threshold=0.72, num_perm=128)

        # Embedder local
        self.embedder = LocalEmbedder(dim=dim)

        # Charger index existant
        self._load_indices()

        logger.info(f"📚 Episodic memory initialized: {self.get_size()} episodes")

    def _init_db(self):
        """Initialise le schéma de base de données"""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                timestamp REAL,
                intent TEXT,
                text_in TEXT,
                text_out TEXT,
                quality_score REAL,
                tags TEXT,
                metadata TEXT,
                embedding_in BLOB,
                embedding_out BLOB
            )
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_timestamp ON episodes(timestamp)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_intent ON episodes(intent)
        """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_quality ON episodes(quality_score)
        """
        )
        # Table de mapping FAISS
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS faiss_map (
                faiss_id INTEGER PRIMARY KEY,
                episode_id TEXT UNIQUE
            )
        """
        )
        self.conn.commit()

    async def store_episode(self, episode: Episode):
        """Stocke un épisode en mémoire"""
        cursor = self.conn.cursor()

        # Générer embeddings si nécessaire
        if episode.embedding_in is None and episode.text_in:
            episode.embedding_in = self.embedder.encode(episode.text_in)
        if episode.embedding_out is None and episode.text_out:
            episode.embedding_out = self.embedder.encode(episode.text_out)

        # Sérialiser les embeddings
        emb_in = pickle.dumps(episode.embedding_in) if episode.embedding_in is not None else None
        emb_out = pickle.dumps(episode.embedding_out) if episode.embedding_out is not None else None

        cursor.execute(
            """
            INSERT OR REPLACE INTO episodes
            (id, timestamp, intent, text_in, text_out, quality_score, tags, metadata, embedding_in, embedding_out)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                episode.id,
                episode.timestamp,
                episode.intent,
                episode.text_in,
                episode.text_out,
                episode.quality_score,
                json.dumps(episode.tags),
                json.dumps(episode.metadata),
                emb_in,
                emb_out,
            ),
        )

        # Ajouter à l'index FAISS avec mapping correct
        if episode.embedding_in is not None:
            faiss_id = int(hashlib.md5(episode.id.encode()).hexdigest()[:8], 16)

            if FAISS_AVAILABLE and self.index:
                cursor.execute(
                    "INSERT OR REPLACE INTO faiss_map(faiss_id, episode_id) VALUES (?,?)",
                    (faiss_id, episode.id),
                )
                self.index.add_with_ids(episode.embedding_in.reshape(1, -1).astype("float32"), np.array([faiss_id]))
            else:
                # Fallback NumPy
                self.vectors[episode.id] = episode.embedding_in

        # Ajouter à LSH
        minhash = self._text_to_minhash(episode.text_in)
        self.lsh.insert(episode.id, minhash)

        self.conn.commit()

        # Sauvegarder les indices
        self._save_indices()

        logger.debug(f"Stored episode: {episode.id}")

    async def search_similar(self, query_embedding: np.ndarray, k: int = 5) -> list[Episode]:
        """
        Recherche les k épisodes les plus similaires
        """
        if FAISS_AVAILABLE and self.index and self.index.ntotal > 0:
            # Recherche FAISS avec mapping correct
            distances, faiss_ids = self.index.search(
                query_embedding.reshape(1, -1).astype("float32"), min(k, self.index.ntotal)
            )

            # Récupérer les episode_ids depuis le mapping
            cursor = self.conn.cursor()
            valid_ids = [int(fid) for fid in faiss_ids[0] if fid != -1]

            if not valid_ids:
                return []

            placeholders = ",".join(["?"] * len(valid_ids))
            cursor.execute(f"SELECT episode_id FROM faiss_map WHERE faiss_id IN ({placeholders})", valid_ids)
            episode_ids = [r[0] for r in cursor.fetchall()]

            if not episode_ids:
                return []

            # Récupérer les épisodes
            placeholders = ",".join(["?"] * len(episode_ids))
            cursor.execute(f"SELECT * FROM episodes WHERE id IN ({placeholders})", episode_ids)
        else:
            # Fallback: calcul cosine similarity avec NumPy
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM episodes WHERE embedding_in IS NOT NULL LIMIT 100")
            all_episodes = [self._row_to_episode(row) for row in cursor.fetchall()]

            if not all_episodes:
                return []

            # Calculer les similarités
            similarities = []
            for ep in all_episodes:
                if ep.embedding_in is not None:
                    sim = np.dot(query_embedding, ep.embedding_in) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(ep.embedding_in) + 1e-8
                    )
                    similarities.append((sim, ep))

            # Trier et retourner top-k
            similarities.sort(key=lambda x: x[0], reverse=True)
            return [ep for _, ep in similarities[:k]]

        return [self._row_to_episode(row) for row in cursor.fetchall()]

    async def search_by_text(self, text: str, k: int = 5) -> list[Episode]:
        """
        Recherche approximative par texte (LSH)
        """
        minhash = self._text_to_minhash(text)

        # Query LSH
        result_ids = self.lsh.query(minhash)

        if not result_ids:
            return []

        # Récupérer depuis DB
        cursor = self.conn.cursor()
        placeholders = ",".join("?" * len(result_ids))
        cursor.execute(f"SELECT * FROM episodes WHERE id IN ({placeholders})", list(result_ids))

        return [self._row_to_episode(row) for row in cursor.fetchall()]

    def _text_to_minhash(self, text: str) -> MinHash:
        """Convertit un texte en MinHash pour LSH"""
        minhash = MinHash(num_perm=128)
        for word in text.lower().split():
            minhash.update(word.encode("utf-8"))
        return minhash

    def _row_to_episode(self, row) -> Episode:
        """Convertit une ligne DB en Episode"""
        return Episode(
            id=row[0],
            timestamp=row[1],
            intent=row[2],
            text_in=row[3],
            text_out=row[4],
            quality_score=row[5],
            tags=json.loads(row[6]),
            metadata=json.loads(row[7]),
            embedding_in=pickle.loads(row[8]) if row[8] else None,
            embedding_out=pickle.loads(row[9]) if row[9] else None,
        )

    def get_size(self) -> int:
        """Retourne le nombre d'épisodes"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM episodes")
        return cursor.fetchone()[0]

    def _save_indices(self):
        """Sauvegarde les indices sur disque"""
        try:
            if FAISS_AVAILABLE and self.index:
                faiss.write_index(self.index, "data/faiss.index")

            with open("data/lsh.pkl", "wb") as f:
                pickle.dump(self.lsh, f)
        except Exception as e:
            logger.debug(f"Could not save indices: {e}")

    def _load_indices(self):
        """Charge les indices depuis le disque si existants"""
        try:
            if FAISS_AVAILABLE and Path("data/faiss.index").exists():
                self.index = faiss.read_index("data/faiss.index")

            if Path("data/lsh.pkl").exists():
                with open("data/lsh.pkl", "rb") as f:
                    self.lsh = pickle.load(f)
        except Exception as e:
            logger.debug(f"Could not load indices: {e}")


class SemanticMemory:
    """
    Mémoire sémantique : patterns consolidés et connaissances
    """

    def __init__(self, patterns_dir: str = "data/patterns"):
        self.patterns_dir = Path(patterns_dir)
        self.patterns_dir.mkdir(parents=True, exist_ok=True)

        # Cache en mémoire des patterns
        self.patterns_cache = {}
        self.templates = {}

        # Charger patterns existants
        self._load_patterns()

    def _load_patterns(self):
        """Charge les patterns depuis le disque"""
        for pattern_file in self.patterns_dir.glob("*.json"):
            try:
                with open(pattern_file) as f:
                    pattern_data = json.load(f)
                    category = pattern_file.stem
                    self.patterns_cache[category] = pattern_data
            except Exception as e:
                logger.debug(f"Could not load pattern {pattern_file}: {e}")

        logger.info(f"📖 Loaded {len(self.patterns_cache)} pattern categories")

    async def consolidate(self, episodes: list[Episode]):
        """
        Consolide des épisodes en patterns
        (Process de consolidation nocturne)
        """
        # Clustering des épisodes similaires
        clusters = self._cluster_episodes(episodes)

        for cluster_id, cluster_episodes in clusters.items():
            # Extraire le pattern commun
            pattern = self._extract_pattern(cluster_episodes)

            if pattern:
                # Sauvegarder le pattern
                await self._save_pattern(pattern)

        logger.info(f"✨ Consolidated {len(clusters)} patterns")

    def _cluster_episodes(self, episodes: list[Episode]) -> dict[str, list[Episode]]:
        """
        Groupe les épisodes similaires
        """
        # Simplification : grouper par intent
        from collections import defaultdict

        clusters = defaultdict(list)
        for episode in episodes:
            clusters[episode.intent].append(episode)
        return dict(clusters)

    def _extract_pattern(self, episodes: list[Episode]) -> dict[str, Any] | None:
        """
        Extrait un pattern depuis un cluster d'épisodes
        """
        if len(episodes) < 3:  # Besoin d'au moins 3 exemples
            return None

        # Extraire les éléments communs
        pattern = {
            "intent": episodes[0].intent,
            "examples": [
                {"input": ep.text_in, "output": ep.text_out}
                for ep in episodes[:5]  # Garder max 5 exemples
            ],
            "quality_mean": np.mean([ep.quality_score for ep in episodes]),
            "count": len(episodes),
            "last_updated": time.time(),
        }

        return pattern

    async def _save_pattern(self, pattern: dict[str, Any]):
        """Sauvegarde un pattern sur disque"""
        category = pattern["intent"]
        pattern_file = self.patterns_dir / f"{category}.json"

        # Charger patterns existants ou créer nouveau
        if pattern_file.exists():
            with open(pattern_file) as f:
                patterns = json.load(f)
        else:
            patterns = []

        # Ajouter le nouveau pattern
        patterns.append(pattern)

        # Limiter à 100 patterns par catégorie
        if len(patterns) > 100:
            # Garder les meilleurs (par qualité et récence)
            patterns = sorted(
                patterns,
                key=lambda p: p["quality_mean"] * 0.7 + (1 - (time.time() - p["last_updated"]) / 86400) * 0.3,
                reverse=True,
            )[:100]

        # Sauvegarder
        with open(pattern_file, "w") as f:
            json.dump(patterns, f, indent=2)

        # Mettre à jour le cache
        self.patterns_cache[category] = patterns

    def get_patterns(self, intent: str = None) -> list[dict[str, Any]]:
        """Récupère les patterns pour un intent"""
        if intent and intent in self.patterns_cache:
            return self.patterns_cache[intent]
        elif intent is None:
            # Retourner tous les patterns
            all_patterns = []
            for patterns in self.patterns_cache.values():
                all_patterns.extend(patterns)
            return all_patterns
        else:
            return []


class TripleMemorySystem:
    """
    Système de mémoire complet intégrant les 3 niveaux
    """

    def __init__(self):
        self.working = WorkingMemory()
        self.episodic = EpisodicMemory()
        self.semantic = SemanticMemory()

        # Tâche de consolidation périodique
        self.consolidation_task = None
        self.consolidation_interval = 900  # 15 minutes

        logger.info("💾 Triple Memory System initialized")

    async def remember(
        self,
        text_in: str,
        text_out: str,
        intent: str,
        quality_score: float,
        embedding_in: np.ndarray = None,
        embedding_out: np.ndarray = None,
        metadata: dict = None,
    ) -> str:
        """
        Enregistre une interaction dans la mémoire
        """
        # Générer ID unique
        episode_id = hashlib.md5(f"{text_in}{text_out}{time.time()}".encode()).hexdigest()

        # Créer l'épisode
        episode = Episode(
            id=episode_id,
            timestamp=time.time(),
            intent=intent,
            text_in=text_in,
            text_out=text_out,
            embedding_in=embedding_in,
            embedding_out=embedding_out,
            quality_score=quality_score,
            tags=metadata.get("tags", []) if metadata else [],
            metadata=metadata or {},
        )

        # Ajouter à la mémoire de travail
        self.working.add(episode)

        # Stocker dans la mémoire épisodique
        await self.episodic.store_episode(episode)

        return episode_id

    async def recall_similar(self, query: str, embedding: np.ndarray = None, k: int = 5) -> list[Episode]:
        """
        Rappelle des épisodes similaires
        """
        # D'abord vérifier la mémoire de travail
        recent = self.working.get_recent(k)
        working_matches = [ep for ep in recent if isinstance(ep, Episode) and query.lower() in ep.text_in.lower()][:k]

        # Puis chercher dans la mémoire épisodique
        if embedding is not None:
            episodic_matches = await self.episodic.search_similar(embedding, k)
        else:
            episodic_matches = await self.episodic.search_by_text(query, k)

        # Combiner et dédupliquer
        all_matches = working_matches + episodic_matches
        seen_ids = set()
        unique_matches = []

        for episode in all_matches:
            if episode.id not in seen_ids:
                seen_ids.add(episode.id)
                unique_matches.append(episode)
                if len(unique_matches) >= k:
                    break

        return unique_matches

    async def get_patterns(self, intent: str = None) -> list[dict[str, Any]]:
        """Récupère les patterns consolidés"""
        return self.semantic.get_patterns(intent)

    async def start_consolidation_loop(self):
        """Démarre la boucle de consolidation périodique"""

        async def consolidate():
            while True:
                try:
                    await asyncio.sleep(self.consolidation_interval)

                    # Récupérer les épisodes récents de haute qualité
                    cursor = self.episodic.conn.cursor()
                    cursor.execute(
                        """
                        SELECT * FROM episodes
                        WHERE quality_score > 0.7
                        AND timestamp > ?
                        ORDER BY timestamp DESC
                        LIMIT 100
                    """,
                        (time.time() - 86400,),
                    )  # Dernières 24h

                    episodes = [self.episodic._row_to_episode(row) for row in cursor.fetchall()]

                    if episodes:
                        await self.semantic.consolidate(episodes)
                        logger.info(f"✨ Consolidated {len(episodes)} episodes")

                except Exception as e:
                    logger.error(f"Consolidation error: {e}")

        self.consolidation_task = asyncio.create_task(consolidate())

    async def shutdown(self):
        """Arrête proprement le système de mémoire"""
        if self.consolidation_task:
            self.consolidation_task.cancel()
            try:
                await self.consolidation_task
            except asyncio.CancelledError:
                pass

    def get_stats(self) -> dict[str, Any]:
        """Retourne les statistiques de mémoire"""
        return {
            "working_memory_size": len(self.working.items),
            "episodic_memory_size": self.episodic.get_size(),
            "semantic_patterns": len(self.semantic.patterns_cache),
            "index_size": self.episodic.index.ntotal
            if FAISS_AVAILABLE and self.episodic.index
            else len(self.episodic.vectors)
            if hasattr(self.episodic, "vectors")
            else 0,
            "consolidation_running": self.consolidation_task is not None and not self.consolidation_task.done(),
        }
