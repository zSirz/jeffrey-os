#!/usr/bin/env python3
"""
🗄️ Jeffrey V2.1.0 - Database Manager SQLite
Remplace le stockage JSON par une base SQLite performante et thread-safe

Migration automatique des données JSON existantes vers SQLite
Support complet des conversations, préférences et patterns appris
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Exception personnalisée pour les erreurs de base de données"""

    pass


class DatabaseManager:
    """
    Gestionnaire de base de données SQLite pour Jeffrey

    Remplace les fichiers JSON par une base SQLite thread-safe avec :
    - Tables conversations, user_preferences, learned_patterns
    - Migration automatique depuis JSON
    - Connection pooling par thread
    - Optimisations de performance
    """

    def __init__(self, db_path: str = "data/jeffrey_memory.db") -> None:
        """
        Initialise le gestionnaire de base de données

        Args:
            db_path: Chemin vers le fichier SQLite
        """
        self.db_path = db_path
        self._ensure_directory()
        self._local = threading.local()  # Thread-safe connections
        self._lock = threading.RLock()

        # Statistiques
        self.stats = {
            "total_conversations": 0,
            "total_users": 0,
            "total_patterns": 0,
            "db_size_bytes": 0,
            "last_vacuum": None,
        }

        # Configuration
        self.max_conversations_per_user = 1000
        self.vacuum_interval_hours = 24
        self.connection_timeout = 30

        # Initialisation
        self._init_database()
        self._migrate_json_if_exists()
        self._update_stats()

        logger.info(f"✅ DatabaseManager initialized: {self.db_path}")

    def _ensure_directory(self):
        """Crée le répertoire data/ si nécessaire"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _get_connection(self, readonly: bool = False):
        """
        Context manager pour les connexions thread-safe

        Args:
            readonly: Si True, ouvre en lecture seule pour optimiser
        """
        if not hasattr(self._local, "connection"):
            try:
                # Configuration optimisée
                if readonly:
                    connection_uri = f"file:{self.db_path}?mode=ro"
                    self._local.connection = sqlite3.connect(
                        connection_uri,
                        uri=True,
                        timeout=self.connection_timeout,
                        check_same_thread=False,
                    )
                else:
                    self._local.connection = sqlite3.connect(
                        self.db_path,
                        timeout=self.connection_timeout,
                        check_same_thread=False,
                    )

                # Optimisations SQLite
                self._local.connection.execute("PRAGMA journal_mode=WAL")
                self._local.connection.execute("PRAGMA synchronous=NORMAL")
                self._local.connection.execute("PRAGMA cache_size=10000")
                self._local.connection.execute("PRAGMA temp_store=MEMORY")

                # Row factory pour dict-like access
                self._local.connection.row_factory = sqlite3.Row

            except sqlite3.Error as e:
                raise DatabaseError(f"Failed to connect to database: {e}")

        try:
            yield self._local.connection
        except sqlite3.Error as e:
            self._local.connection.rollback()
            raise DatabaseError(f"Database operation failed: {e}")

    def _init_database(self):
        """Initialise les tables de la base de données"""
        with self._get_connection() as conn:
            try:
                # Table conversations (remplace conversation_history.json)
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        emotion TEXT,
                        metadata TEXT,  -- JSON string
                        importance_score REAL DEFAULT 1.0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )

                # Index pour les performances
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_user_timestamp
                    ON conversations(user_id, timestamp DESC)
                """
                )

                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_user_created
                    ON conversations(user_id, created_at DESC)
                """
                )

                # Table user_preferences (remplace user_preferences.json)
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS user_preferences (
                        user_id TEXT PRIMARY KEY,
                        preferences TEXT NOT NULL,  -- JSON string
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )

                # Table learned_patterns (remplace learned_patterns.json)
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS learned_patterns (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pattern_type TEXT NOT NULL,
                        pattern_data TEXT NOT NULL,  -- JSON string
                        frequency INTEGER DEFAULT 1,
                        confidence REAL DEFAULT 1.0,
                        last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )

                # Index pour patterns
                conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_pattern_type_freq
                    ON learned_patterns(pattern_type, frequency DESC)
                """
                )

                # Table pour métadonnées système
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS system_metadata (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """
                )

                conn.commit()
                logger.info("✅ Database tables initialized successfully")

            except sqlite3.Error as e:
                conn.rollback()
                raise DatabaseError(f"Failed to initialize database: {e}")

    # ==================== CONVERSATIONS ====================

    def add_conversation(
        self,
        user_id: str,
        role: str,
        content: str,
        emotion: str = None,
        metadata: dict = None,
        importance_score: float = 1.0,
    ) -> int:
        """
        Ajoute une conversation à la base

        Args:
            user_id: ID utilisateur
            role: Role (user/assistant)
            content: Contenu du message
            emotion: Emotion détectée
            metadata: Métadonnées additionnelles
            importance_score: Score d'importance (0.0-1.0)

        Returns:
            ID de la conversation créée
        """
        timestamp = datetime.now().isoformat()
        metadata_json = json.dumps(metadata or {})

        with self._lock:
            with self._get_connection() as conn:
                try:
                    cursor = conn.execute(
                        """
                        INSERT INTO conversations
                        (user_id, role, content, timestamp, emotion, metadata, importance_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            user_id,
                            role,
                            content,
                            timestamp,
                            emotion,
                            metadata_json,
                            importance_score,
                        ),
                    )

                    conversation_id = cursor.lastrowid
                    conn.commit()

                    # Nettoyage automatique si trop de conversations
                    self._cleanup_old_conversations(user_id, conn)

                    logger.debug(f"Added conversation {conversation_id} for user {user_id}")
                    return conversation_id

                except sqlite3.Error as e:
                    conn.rollback()
                    raise DatabaseError(f"Failed to add conversation: {e}")

    def get_conversation_history(self, user_id: str, limit: int = 100) -> list[dict]:
        """
        Récupère l'historique des conversations d'un utilisateur

        Args:
            user_id: ID utilisateur
            limit: Nombre maximum de messages

        Returns:
            Liste des conversations ordonnées par timestamp DESC
        """
        with self._get_connection(readonly=True) as conn:
            try:
                cursor = conn.execute(
                    """
                    SELECT id, user_id, role, content, timestamp, emotion,
                           metadata, importance_score, created_at
                    FROM conversations
                    WHERE user_id = ?
                    ORDER BY timestamp DESC, created_at DESC
                    LIMIT ?
                """,
                    (user_id, limit),
                )

                conversations = []
                for row in cursor.fetchall():
                    conv = dict(row)
                    # Parse metadata JSON
                    try:
                        conv["metadata"] = json.loads(conv["metadata"] or "{}")
                    except json.JSONDecodeError:
                        conv["metadata"] = {}
                    conversations.append(conv)

                logger.debug(f"Retrieved {len(conversations)} conversations for user {user_id}")
                return conversations

            except sqlite3.Error as e:
                raise DatabaseError(f"Failed to get conversation history: {e}")

    def get_recent_context(self, user_id: str, limit: int = 10) -> list[dict]:
        """
        Récupère le contexte récent pour un utilisateur (format compatible MemoryManager)

        Args:
            user_id: ID utilisateur
            limit: Nombre de messages récents

        Returns:
            Liste des messages dans l'ordre chronologique (ancien -> récent)
        """
        conversations = self.get_conversation_history(user_id, limit)

        # Convertir au format MemoryManager (inversé pour avoir ancien -> récent)
        context = []
        for conv in reversed(conversations):
            context_entry = {
                "timestamp": conv["timestamp"],
                "user_id": conv["user_id"],
                "type": "conversation",
                "metadata": conv["metadata"],
            }

            if conv["role"] == "user":
                context_entry["user_message"] = conv["content"]
            else:
                context_entry["ai_response"] = conv["content"]
                if conv["emotion"]:
                    context_entry["emotion"] = conv["emotion"]

            context.append(context_entry)

        return context

    def _cleanup_old_conversations(self, user_id: str, conn: sqlite3.Connection):
        """Nettoie les anciennes conversations si on dépasse la limite"""
        try:
            # Compter les conversations pour cet utilisateur
            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM conversations WHERE user_id = ?
            """,
                (user_id,),
            )

            count = cursor.fetchone()[0]

            if count > self.max_conversations_per_user:
                # Supprimer les plus anciennes
                to_delete = count - self.max_conversations_per_user
                conn.execute(
                    """
                    DELETE FROM conversations
                    WHERE user_id = ?
                    AND id IN (
                        SELECT id FROM conversations
                        WHERE user_id = ?
                        ORDER BY timestamp ASC, created_at ASC
                        LIMIT ?
                    )
                """,
                    (user_id, user_id, to_delete),
                )

                logger.info(f"Cleaned up {to_delete} old conversations for user {user_id}")

        except sqlite3.Error as e:
            logger.error(f"Failed to cleanup conversations: {e}")

    # ==================== USER PREFERENCES ====================

    def save_user_preferences(self, user_id: str, preferences: dict) -> bool:
        """
        Sauvegarde les préférences utilisateur

        Args:
            user_id: ID utilisateur
            preferences: Dictionnaire des préférences

        Returns:
            True si succès
        """
        preferences_json = json.dumps(preferences, ensure_ascii=False)

        with self._lock:
            with self._get_connection() as conn:
                try:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO user_preferences
                        (user_id, preferences, updated_at)
                        VALUES (?, ?, CURRENT_TIMESTAMP)
                    """,
                        (user_id, preferences_json),
                    )

                    conn.commit()
                    logger.debug(f"Saved preferences for user {user_id}")
                    return True

                except sqlite3.Error as e:
                    conn.rollback()
                    raise DatabaseError(f"Failed to save user preferences: {e}")

    def get_user_preferences(self, user_id: str) -> dict:
        """
        Récupère les préférences utilisateur

        Args:
            user_id: ID utilisateur

        Returns:
            Dictionnaire des préférences (vide si aucune)
        """
        with self._get_connection(readonly=True) as conn:
            try:
                cursor = conn.execute(
                    """
                    SELECT preferences FROM user_preferences WHERE user_id = ?
                """,
                    (user_id,),
                )

                row = cursor.fetchone()
                if row:
                    try:
                        return json.loads(row["preferences"])
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON in preferences for user {user_id}")
                        return {}

                return {}

            except sqlite3.Error as e:
                raise DatabaseError(f"Failed to get user preferences: {e}")

    # ==================== LEARNED PATTERNS ====================

    def save_learned_pattern(self, pattern_type: str, pattern_data: dict, confidence: float = 1.0) -> int:
        """
        Sauvegarde un pattern appris

        Args:
            pattern_type: Type de pattern (ex: "frequent_question")
            pattern_data: Données du pattern
            confidence: Niveau de confiance (0.0-1.0)

        Returns:
            ID du pattern créé
        """
        pattern_json = json.dumps(pattern_data, ensure_ascii=False)

        with self._lock:
            with self._get_connection() as conn:
                try:
                    cursor = conn.execute(
                        """
                        INSERT INTO learned_patterns
                        (pattern_type, pattern_data, confidence)
                        VALUES (?, ?, ?)
                    """,
                        (pattern_type, pattern_json, confidence),
                    )

                    pattern_id = cursor.lastrowid
                    conn.commit()

                    logger.debug(f"Saved pattern {pattern_id} of type {pattern_type}")
                    return pattern_id

                except sqlite3.Error as e:
                    conn.rollback()
                    raise DatabaseError(f"Failed to save learned pattern: {e}")

    def get_learned_patterns(self, pattern_type: str = None, limit: int = 100) -> list[dict]:
        """
        Récupère les patterns appris

        Args:
            pattern_type: Type spécifique de pattern (optionnel)
            limit: Nombre maximum de patterns

        Returns:
            Liste des patterns trouvés
        """
        with self._get_connection(readonly=True) as conn:
            try:
                if pattern_type:
                    cursor = conn.execute(
                        """
                        SELECT id, pattern_type, pattern_data, frequency, confidence,
                               last_used, created_at
                        FROM learned_patterns
                        WHERE pattern_type = ?
                        ORDER BY frequency DESC, confidence DESC
                        LIMIT ?
                    """,
                        (pattern_type, limit),
                    )
                else:
                    cursor = conn.execute(
                        """
                        SELECT id, pattern_type, pattern_data, frequency, confidence,
                               last_used, created_at
                        FROM learned_patterns
                        ORDER BY frequency DESC, confidence DESC
                        LIMIT ?
                    """,
                        (limit,),
                    )

                patterns = []
                for row in cursor.fetchall():
                    pattern = dict(row)
                    try:
                        pattern["pattern_data"] = json.loads(pattern["pattern_data"])
                    except json.JSONDecodeError:
                        pattern["pattern_data"] = {}
                    patterns.append(pattern)

                logger.debug(f"Retrieved {len(patterns)} patterns of type {pattern_type}")
                return patterns

            except sqlite3.Error as e:
                raise DatabaseError(f"Failed to get learned patterns: {e}")

    def increment_pattern_frequency(self, pattern_id: int) -> bool:
        """
        Incrémente la fréquence d'utilisation d'un pattern

        Args:
            pattern_id: ID du pattern

        Returns:
            True si succès
        """
        with self._lock:
            with self._get_connection() as conn:
                try:
                    cursor = conn.execute(
                        """
                        UPDATE learned_patterns
                        SET frequency = frequency + 1, last_used = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """,
                        (pattern_id,),
                    )

                    conn.commit()

                    if cursor.rowcount > 0:
                        logger.debug(f"Incremented frequency for pattern {pattern_id}")
                        return True
                    else:
                        logger.warning(f"Pattern {pattern_id} not found for frequency increment")
                        return False

                except sqlite3.Error as e:
                    conn.rollback()
                    raise DatabaseError(f"Failed to increment pattern frequency: {e}")

    # ==================== MIGRATION JSON ====================

    def _migrate_json_if_exists(self):
        """Migre automatiquement les fichiers JSON existants vers SQLite"""
        data_dir = Path(self.db_path).parent

        json_files = {
            "conversation_history.json": self._migrate_conversations,
            "user_preferences.json": self._migrate_preferences,
            "learned_patterns.json": self._migrate_patterns,
        }

        migrated_any = False

        for filename, migrate_func in json_files.items():
            json_path = data_dir / filename
            if json_path.exists():
                logger.info(f"📦 Migration de {filename} vers SQLite...")
                try:
                    count = migrate_func(json_path)
                    logger.info(f"✅ Migré {count} entrées de {filename}")

                    # Renommer le fichier pour backup
                    backup_path = json_path.with_suffix(".json.backup")
                    json_path.rename(backup_path)
                    logger.info(f"📋 Backup créé: {backup_path}")

                    migrated_any = True

                except Exception as e:
                    logger.error(f"❌ Erreur migration {filename}: {e}")

        if migrated_any:
            logger.info("🎉 Migration JSON vers SQLite terminée avec succès!")

    def _migrate_conversations(self, json_path: Path) -> int:
        """Migre conversation_history.json vers la table conversations"""
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)

            count = 0

            with self._get_connection() as conn:
                for user_id, conversations in data.items():
                    for conv in conversations:
                        # Détecter le format et convertir
                        if "user_message" in conv:
                            # Message utilisateur
                            self._insert_conversation_raw(
                                conn,
                                user_id,
                                "user",
                                conv["user_message"],
                                conv.get("timestamp"),
                                None,
                                conv.get("metadata", {}),
                            )
                            count += 1

                        if "ai_response" in conv:
                            # Réponse IA
                            self._insert_conversation_raw(
                                conn,
                                user_id,
                                "assistant",
                                conv["ai_response"],
                                conv.get("timestamp"),
                                conv.get("emotion"),
                                conv.get("metadata", {}),
                            )
                            count += 1

                conn.commit()

            return count

        except Exception as e:
            raise DatabaseError(f"Failed to migrate conversations: {e}")

    def _migrate_preferences(self, json_path: Path) -> int:
        """Migre user_preferences.json vers la table user_preferences"""
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)

            count = 0

            with self._get_connection() as conn:
                for user_id, preferences in data.items():
                    self.save_user_preferences(user_id, preferences)
                    count += 1

                conn.commit()

            return count

        except Exception as e:
            raise DatabaseError(f"Failed to migrate preferences: {e}")

    def _migrate_patterns(self, json_path: Path) -> int:
        """Migre learned_patterns.json vers la table learned_patterns"""
        try:
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)

            count = 0

            # Les patterns peuvent être dans différents formats
            if isinstance(data, dict):
                for pattern_type, pattern_data in data.items():
                    if isinstance(pattern_data, dict):
                        self.save_learned_pattern(pattern_type, pattern_data)
                        count += 1
                    elif isinstance(pattern_data, list):
                        for item in pattern_data:
                            self.save_learned_pattern(pattern_type, item)
                            count += 1

            return count

        except Exception as e:
            raise DatabaseError(f"Failed to migrate patterns: {e}")

    def _insert_conversation_raw(
        self,
        conn: sqlite3.Connection,
        user_id: str,
        role: str,
        content: str,
        timestamp: str = None,
        emotion: str = None,
        metadata: dict = None,
    ):
        """Insert direct en base sans validation (pour migration)"""
        if not timestamp:
            timestamp = datetime.now().isoformat()

        metadata_json = json.dumps(metadata or {})

        conn.execute(
            """
            INSERT INTO conversations
            (user_id, role, content, timestamp, emotion, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (user_id, role, content, timestamp, emotion, metadata_json),
        )

    # ==================== MAINTENANCE & STATS ====================

    def _update_stats(self):
        """Met à jour les statistiques de la base"""
        with self._get_connection(readonly=True) as conn:
            try:
                # Conversations
                cursor = conn.execute("SELECT COUNT(*) FROM conversations")
                self.stats["total_conversations"] = cursor.fetchone()[0]

                # Utilisateurs uniques
                cursor = conn.execute("SELECT COUNT(DISTINCT user_id) FROM conversations")
                self.stats["total_users"] = cursor.fetchone()[0]

                # Patterns
                cursor = conn.execute("SELECT COUNT(*) FROM learned_patterns")
                self.stats["total_patterns"] = cursor.fetchone()[0]

                # Taille de la base
                if os.path.exists(self.db_path):
                    self.stats["db_size_bytes"] = os.path.getsize(self.db_path)

            except sqlite3.Error as e:
                logger.error(f"Failed to update stats: {e}")

    def vacuum_database(self):
        """Optimise la base de données (VACUUM)"""
        with self._lock:
            with self._get_connection() as conn:
                try:
                    logger.info("🧹 Starting database vacuum...")
                    conn.execute("VACUUM")
                    conn.commit()

                    self.stats["last_vacuum"] = datetime.now().isoformat()
                    self._update_stats()

                    logger.info("✅ Database vacuum completed")

                except sqlite3.Error as e:
                    raise DatabaseError(f"Failed to vacuum database: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Retourne les statistiques de la base"""
        self._update_stats()
        return self.stats.copy()

    def close(self):
        """Ferme les connexions"""
        if hasattr(self._local, "connection"):
            try:
                self._local.connection.close()
                del self._local.connection
            except:
                pass

        logger.info("Database connections closed")

    def __del__(self):
        """Destructeur - ferme les connexions"""
        self.close()


# ==================== HELPERS POUR TESTS ====================


def create_test_database(db_path: str = ":memory:") -> DatabaseManager:
    """Crée une base de test en mémoire"""
    return DatabaseManager(db_path)


def backup_database(source_db: str, backup_path: str) -> bool:
    """Crée un backup de la base de données"""
    try:
        import shutil

        shutil.copy2(source_db, backup_path)
        logger.info(f"Database backup created: {backup_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to backup database: {e}")
        return False


if __name__ == "__main__":
    # Test rapide
    print("🗄️ Jeffrey DatabaseManager - Test rapide")

    # Créer une base de test
    db = create_test_database()

    # Test conversation
    conv_id = db.add_conversation("test_user", "user", "Bonjour Jeffrey!")
    print(f"✅ Conversation créée: {conv_id}")

    db.add_conversation("test_user", "assistant", "Bonjour ! Comment allez-vous ?", emotion="friendly")
    print("✅ Réponse ajoutée")

    # Test récupération
    history = db.get_conversation_history("test_user")
    print(f"✅ Historique récupéré: {len(history)} messages")

    context = db.get_recent_context("test_user")
    print(f"✅ Contexte récupéré: {len(context)} entrées")

    # Test préférences
    db.save_user_preferences("test_user", {"language": "fr", "tone": "friendly"})
    prefs = db.get_user_preferences("test_user")
    print(f"✅ Préférences sauvegardées: {prefs}")

    # Test patterns
    pattern_id = db.save_learned_pattern("greeting", {"pattern": "bonjour", "response": "salut"})
    patterns = db.get_learned_patterns("greeting")
    print(f"✅ Pattern sauvegardé: {len(patterns)} trouvé(s)")

    # Stats
    stats = db.get_stats()
    print(f"📊 Stats: {stats}")

    print("🎉 Test DatabaseManager terminé avec succès!")
