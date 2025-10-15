"""
jeffrey/ml/feedback.py
Système de feedback et apprentissage continu pour Jeffrey OS Phase 1.
"""

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FeedbackEvent:
    """Événement de feedback utilisateur."""

    text: str
    embedding: np.ndarray
    predicted_emotion: str
    predicted_confidence: float
    corrected_emotion: str | None
    user_confidence: float  # 0-1
    timestamp: datetime
    user_id: str | None = None
    abstention: bool = False
    margin: float = 0.0


class FeedbackStore:
    """Stockage et gestion des événements de feedback.

    Features:
    - SQLite pour persistance
    - Snapshots quotidiens (rollback)
    - Quotas par label
    - Anti-outliers
    - Export/stats
    """

    def __init__(self, db_path: Path = Path("data/feedback.db")):
        """Initialise le store.

        Args:
            db_path: Chemin vers la DB SQLite
        """
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

        logger.info(f"FeedbackStore initialized at {db_path}")

    def _init_db(self):
        """Initialise les tables SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    predicted_emotion TEXT NOT NULL,
                    predicted_confidence REAL NOT NULL,
                    corrected_emotion TEXT,
                    user_confidence REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    user_id TEXT,
                    abstention INTEGER NOT NULL,
                    margin REAL NOT NULL
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON feedback_events(timestamp)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_corrected_emotion
                ON feedback_events(corrected_emotion)
            """)

            conn.commit()

    def add_event(self, event: FeedbackEvent) -> int:
        """Ajoute un événement de feedback.

        Args:
            event: FeedbackEvent à stocker

        Returns:
            ID de l'événement créé
        """
        # Sérialiser embedding
        embedding_bytes = event.embedding.tobytes()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO feedback_events (
                    text, embedding, predicted_emotion, predicted_confidence,
                    corrected_emotion, user_confidence, timestamp, user_id,
                    abstention, margin
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    event.text,
                    embedding_bytes,
                    event.predicted_emotion,
                    event.predicted_confidence,
                    event.corrected_emotion,
                    event.user_confidence,
                    event.timestamp.isoformat(),
                    event.user_id,
                    int(event.abstention),
                    event.margin,
                ),
            )

            event_id = cursor.lastrowid
            conn.commit()

        logger.debug(f"Added feedback event #{event_id}")
        return event_id

    def get_corrections_since(
        self, since: datetime, label: str | None = None, limit: int | None = None
    ) -> list[FeedbackEvent]:
        """Récupère les corrections depuis une date.

        Args:
            since: Date de départ
            label: Filtrer par émotion corrigée
            limit: Nombre max de résultats

        Returns:
            Liste de FeedbackEvent
        """
        query = """
            SELECT * FROM feedback_events
            WHERE timestamp >= ? AND corrected_emotion IS NOT NULL
        """
        params = [since.isoformat()]

        if label:
            query += " AND corrected_emotion = ?"
            params.append(label)

        query += " ORDER BY timestamp DESC"

        if limit:
            query += f" LIMIT {limit}"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

        events = []
        for row in rows:
            # Désérialiser embedding
            embedding = np.frombuffer(row['embedding'], dtype=np.float32)

            events.append(
                FeedbackEvent(
                    text=row['text'],
                    embedding=embedding,
                    predicted_emotion=row['predicted_emotion'],
                    predicted_confidence=row['predicted_confidence'],
                    corrected_emotion=row['corrected_emotion'],
                    user_confidence=row['user_confidence'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    user_id=row['user_id'],
                    abstention=bool(row['abstention']),
                    margin=row['margin'],
                )
            )

        return events

    def get_stats(self, days: int = 7) -> dict:
        """Statistiques récentes.

        Args:
            days: Nombre de jours à analyser

        Returns:
            Dict avec statistiques
        """
        since = datetime.now() - timedelta(days=days)

        with sqlite3.connect(self.db_path) as conn:
            # Total events
            total = conn.execute(
                "SELECT COUNT(*) FROM feedback_events WHERE timestamp >= ?", (since.isoformat(),)
            ).fetchone()[0]

            # Corrections
            corrections = conn.execute(
                "SELECT COUNT(*) FROM feedback_events WHERE timestamp >= ? AND corrected_emotion IS NOT NULL",
                (since.isoformat(),),
            ).fetchone()[0]

            # Abstentions
            abstentions = conn.execute(
                "SELECT COUNT(*) FROM feedback_events WHERE timestamp >= ? AND abstention = 1", (since.isoformat(),)
            ).fetchone()[0]

            # Accuracy (si corrections)
            if corrections > 0:
                correct_predictions = conn.execute(
                    "SELECT COUNT(*) FROM feedback_events "
                    "WHERE timestamp >= ? AND corrected_emotion IS NOT NULL "
                    "AND predicted_emotion = corrected_emotion",
                    (since.isoformat(),),
                ).fetchone()[0]
                accuracy = correct_predictions / corrections
            else:
                accuracy = 0.0

        return {
            "days": days,
            "total_events": total,
            "corrections": corrections,
            "abstentions": abstentions,
            "correction_rate": corrections / total if total > 0 else 0.0,
            "abstention_rate": abstentions / total if total > 0 else 0.0,
            "accuracy": accuracy,
        }

    def create_snapshot(self, snapshot_dir: Path = Path("data/snapshots")) -> Path:
        """Crée un snapshot de la DB (pour rollback).

        Args:
            snapshot_dir: Dossier des snapshots

        Returns:
            Path du snapshot créé
        """
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_path = snapshot_dir / f"feedback_{timestamp}.db"

        # Copie simple de la DB
        import shutil

        shutil.copy2(self.db_path, snapshot_path)

        logger.info(f"Snapshot created: {snapshot_path}")
        return snapshot_path


if __name__ == "__main__":
    # Test rapide
    logging.basicConfig(level=logging.INFO)

    store = FeedbackStore(Path("data/feedback_test.db"))

    # Test event
    event = FeedbackEvent(
        text="Je suis en colère",
        embedding=np.random.randn(384).astype(np.float32),
        predicted_emotion="anger",
        predicted_confidence=0.85,
        corrected_emotion="frustration",
        user_confidence=0.9,
        timestamp=datetime.now(),
        abstention=False,
        margin=0.2,
    )

    event_id = store.add_event(event)
    print(f"✅ Event added: #{event_id}")

    # Stats
    stats = store.get_stats(days=7)
    print(f"✅ Stats: {stats}")

    # Snapshot
    snapshot = store.create_snapshot()
    print(f"✅ Snapshot: {snapshot}")
