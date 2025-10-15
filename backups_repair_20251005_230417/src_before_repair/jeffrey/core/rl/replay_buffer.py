"""
ReplayBuffer for stable Q-learning
Phase 2.3 implementation
"""

import logging
import pickle
import random
from collections import deque
from pathlib import Path

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """Buffer circulaire pour stabiliser Q-learning"""

    def __init__(self, capacity: int = 10000, persist_path: Path | None = None):
        self.buffer = deque(maxlen=capacity)
        self.persist_path = persist_path or Path("data/replay_buffer.pkl")
        self.total_added = 0
        self.total_sampled = 0
        self.load()

    def add(self, state: str, action: str, reward: float, next_state: str, done: bool = False):
        """Ajoute une transition au buffer"""
        self.buffer.append((state, action, reward, next_state, done))
        self.total_added += 1

    def sample(self, batch_size: int = 32) -> list[tuple]:
        """Échantillonne aléatoirement des transitions"""
        if len(self.buffer) == 0:
            return []
        sample_size = min(batch_size, len(self.buffer))
        self.total_sampled += sample_size
        return random.sample(self.buffer, sample_size)

    def save(self):
        """Persiste le buffer sur disque"""
        try:
            self.persist_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.persist_path, "wb") as f:
                pickle.dump(
                    {
                        "buffer": list(self.buffer),
                        "total_added": self.total_added,
                        "total_sampled": self.total_sampled,
                    },
                    f,
                )
            logger.debug(f"Replay buffer saved: {len(self.buffer)} items")
        except Exception as e:
            logger.error(f"Failed to save replay buffer: {e}")

    def load(self):
        """Charge le buffer depuis le disque"""
        if self.persist_path.exists():
            try:
                with open(self.persist_path, "rb") as f:
                    data = pickle.load(f)
                    self.buffer = deque(data["buffer"], maxlen=self.buffer.maxlen)
                    self.total_added = data.get("total_added", 0)
                    self.total_sampled = data.get("total_sampled", 0)
                logger.info(f"Replay buffer loaded: {len(self.buffer)} items")
            except Exception as e:
                logger.error(f"Failed to load replay buffer: {e}")

    def clear(self):
        """Vide le buffer"""
        self.buffer.clear()

    def get_stats(self) -> dict:
        """Statistiques du buffer"""
        return {
            "size": len(self.buffer),
            "capacity": self.buffer.maxlen,
            "utilization": len(self.buffer) / self.buffer.maxlen,
            "total_added": self.total_added,
            "total_sampled": self.total_sampled,
        }

    def __len__(self):
        return len(self.buffer)

    def __repr__(self):
        return f"ReplayBuffer(size={len(self.buffer)}/{self.buffer.maxlen})"
