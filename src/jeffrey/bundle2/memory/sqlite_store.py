"""
SQLite Memory Store - Version corrigée avec retour du context
Inclut tous les fixes GPT
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any


class SQLiteMemoryStore:
    """Store de mémoire avec toutes les corrections"""

    def __init__(self, db_path: str = "data/jeffrey_memory.db"):
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self):
        """S'assurer que la DB existe avec le bon schéma"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        con = sqlite3.connect(self.db_path)
        cur = con.cursor()

        # Créer la table avec TOUTES les colonnes nécessaires
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                context TEXT,
                emotions TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                importance REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                last_accessed DATETIME,
                session_id TEXT,
                source TEXT,
                vector BLOB
            )
        """
        )

        # Créer les index
        cur.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_session ON memories(session_id)")

        con.commit()
        con.close()

    def _conn(self):
        """Créer une connexion"""
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row  # Pour accès par nom de colonne
        return con

    async def store(self, content: Any, metadata: dict = None) -> str:
        """Stocker une mémoire"""
        metadata = metadata or {}

        def _store():
            con = self._conn()
            cur = con.cursor()

            mid = f"mem_{int(time.time() * 1000000)}"

            content_str = json.dumps(content) if not isinstance(content, str) else content
            context = metadata.get("context", {})

            # Extraire session_id du context si présent
            session_id = context.get("session_id") if isinstance(context, dict) else None

            cur.execute(
                """
                INSERT OR REPLACE INTO memories
                (id, content, context, emotions, importance, session_id, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    mid,
                    content_str,
                    json.dumps(context),
                    json.dumps(metadata.get("emotions", {})),
                    float(metadata.get("importance", 0.5)),
                    session_id or metadata.get("session_id", "default"),
                    metadata.get("source", "unknown"),
                ),
            )

            con.commit()
            con.close()
            return mid

        try:
            return await asyncio.to_thread(_store)
        except AttributeError:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _store)

    async def recall(self, memory_id: str) -> dict | None:
        """Rappeler une mémoire par ID"""

        def _recall():
            con = self._conn()
            cur = con.cursor()

            cur.execute(
                """
                SELECT id, content, context, emotions, importance,
                       timestamp, access_count, last_accessed, session_id, source
                FROM memories WHERE id = ?
            """,
                (memory_id,),
            )

            row = cur.fetchone()

            if row:
                # Mettre à jour l'accès
                cur.execute(
                    """
                    UPDATE memories
                    SET access_count = access_count + 1,
                        last_accessed = CURRENT_TIMESTAMP
                    WHERE id = ?
                """,
                    (memory_id,),
                )
                con.commit()

            con.close()

            if not row:
                return None

            # Retourner avec tous les champs
            return {
                "id": row["id"],
                "content": json.loads(row["content"])
                if row["content"].startswith("[") or row["content"].startswith("{")
                else row["content"],
                "context": json.loads(row["context"]) if row["context"] else {},
                "emotions": json.loads(row["emotions"]) if row["emotions"] else {},
                "importance": row["importance"],
                "timestamp": row["timestamp"],
                "access_count": row["access_count"],
                "last_accessed": row["last_accessed"],
                "session_id": row["session_id"],
                "source": row["source"],
            }

        try:
            return await asyncio.to_thread(_recall)
        except AttributeError:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _recall)

    async def search(self, query: str, limit: int = 10) -> list[dict]:
        """Recherche avec retour du context (FIX GPT BONUS)"""
        q = f"%{query.lower()}%"

        def _search():
            con = self._conn()
            cur = con.cursor()

            # Recherche dans content, context ET emotions
            cur.execute(
                """
                SELECT id, content, context, emotions, importance,
                       timestamp, access_count, session_id, source
                FROM memories
                WHERE lower(content) LIKE ?
                   OR lower(context) LIKE ?
                   OR lower(emotions) LIKE ?
                ORDER BY importance DESC, timestamp DESC
                LIMIT ?
            """,
                (q, q, q, limit),
            )

            results = []
            for row in cur.fetchall():
                content = row["content"]
                if content and (content.startswith("[") or content.startswith("{")):
                    content = json.loads(content)

                # IMPORTANT: Retourner le context (FIX GPT BONUS)
                results.append(
                    {
                        "id": row["id"],
                        "content": content,
                        "context": json.loads(row["context"]) if row["context"] else {},
                        "emotions": json.loads(row["emotions"]) if row["emotions"] else {},
                        "importance": row["importance"],
                        "timestamp": row["timestamp"],
                        "access_count": row["access_count"] or 0,
                        "session_id": row["session_id"],
                        "source": row["source"],
                    }
                )

            con.close()
            return results

        try:
            return await asyncio.to_thread(_search)
        except AttributeError:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _search)

    async def update_access(self, memory_id: str) -> None:
        """Mettre à jour le compteur d'accès (FIX GPT #3)"""

        def _update():
            con = self._conn()
            cur = con.cursor()

            cur.execute(
                """
                UPDATE memories
                SET access_count = access_count + 1,
                    last_accessed = CURRENT_TIMESTAMP
                WHERE id = ?
            """,
                (memory_id,),
            )

            con.commit()
            con.close()

        try:
            await asyncio.to_thread(_update)
        except AttributeError:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _update)

    async def consolidate(self) -> dict:
        """Consolidation des mémoires"""

        def _consolidate():
            con = self._conn()
            cur = con.cursor()

            # Stats basiques
            cur.execute("SELECT COUNT(*) FROM memories")
            total = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM memories WHERE importance > 0.7")
            important = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM memories WHERE access_count > 5")
            accessed = cur.fetchone()[0]

            con.close()

            return {
                "total_memories": total,
                "important_memories": important,
                "frequently_accessed": accessed,
                "status": "consolidated",
            }

        try:
            return await asyncio.to_thread(_consolidate)
        except AttributeError:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _consolidate)

    def health_check(self):
        """Health check du module"""
        try:
            os.makedirs("data", exist_ok=True)
            con = sqlite3.connect(self.db_path)
            con.execute("SELECT 1")
            con.close()
            return {"status": "healthy", "module": __name__, "db": self.db_path}
        except Exception as e:
            return {"status": "unhealthy", "module": __name__, "error": str(e)}


# Pour import direct
def health_check():
    store = SQLiteMemoryStore()
    return store.health_check()
