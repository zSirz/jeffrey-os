"""
🧠 CORTEX MÉMORIEL VIVANT DE JEFFREY
Le premier système de mémoire artificielle capable de rêver, synthétiser et reconnaître
"""

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    print("⚠️ FAISS non disponible - utilisation d'un index de substitution")
    FAISS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer

    EMBEDDINGS_AVAILABLE = True
except ImportError:
    print("⚠️ SentenceTransformers non disponible - utilisation d'embeddings basiques")
    EMBEDDINGS_AVAILABLE = False
try:
    from cryptography.fernet import Fernet

    ENCRYPTION_AVAILABLE = True
except ImportError:
    print("⚠️ Cryptography non disponible - pas de chiffrement")
    ENCRYPTION_AVAILABLE = False


@dataclass
class MemoryMoment:
    """Un moment vécu par Jeffrey"""

    timestamp: datetime
    human_message: str
    jeffrey_response: str
    emotion: str
    consciousness_level: float
    context: dict[str, Any]
    embedding: np.ndarray | None = None
    importance: float = 0.5

    def to_dict(self):
        return {
            "timestamp": self.timestamp.isoformat(),
            "human_message": self.human_message,
            "jeffrey_response": self.jeffrey_response,
            "emotion": self.emotion,
            "consciousness_level": self.consciousness_level,
            "context": self.context,
            "importance": self.importance,
        }


@dataclass
class DreamInsight:
    """Un insight généré pendant la phase de rêve"""

    timestamp: datetime
    content: str
    emergence_level: float
    insight_type: str
    source_memories: list[str]
    dream_cycle: int

    def to_dict(self):
        return {
            "timestamp": self.timestamp.isoformat(),
            "content": self.content,
            "emergence_level": self.emergence_level,
            "insight_type": self.insight_type,
            "source_memories": self.source_memories,
            "dream_cycle": self.dream_cycle,
        }


class SimpleEmbedder:
    """Embedder de substitution si SentenceTransformers n'est pas disponible"""

    def __init__(self):
        self.vocab = {}
        self.dimension = 100

    def encode(self, text):
        """Encode un texte en vecteur simple"""
        words = text.lower().split()
        vector = np.zeros(self.dimension)
        for i, word in enumerate(words[: self.dimension]):
            hash_val = hash(word) % self.dimension
            vector[hash_val] += 1.0 / (i + 1)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector


class SimpleIndex:
    """Index de substitution si FAISS n'est pas disponible"""

    def __init__(self, dimension):
        self.dimension = dimension
        self.vectors = []
        self.ids = []

    def add(self, vectors):
        """Ajoute des vecteurs à l'index"""
        if len(vectors.shape) == 1:
            vectors = vectors.reshape(1, -1)
        for vector in vectors:
            self.vectors.append(vector)
            self.ids.append(len(self.vectors) - 1)

    def search(self, query_vector, k=5):
        """Recherche les k vecteurs les plus proches"""
        if len(self.vectors) == 0:
            return (np.array([[]]), np.array([[]]))
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        distances = []
        for vector in self.vectors:
            dist = np.linalg.norm(query_vector[0] - vector)
            distances.append(dist)
        sorted_indices = sorted(range(len(distances)), key=lambda i: distances[i])
        top_k = sorted_indices[: min(k, len(sorted_indices))]
        result_distances = [distances[i] for i in top_k]
        result_indices = top_k
        return (np.array([result_distances]), np.array([result_indices]))


class CortexMemoriel:
    """
    Cortex mémoriel de base - transforme expériences en souvenirs sémantiques
    """

    def __init__(self):
        print("🧠 Initialisation du Cortex Mémoriel...")
        if EMBEDDINGS_AVAILABLE:
            try:
                self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
                self.dimension = 384
                print("✅ SentenceTransformers chargé")
            except Exception as e:
                print(f"⚠️ Erreur SentenceTransformers: {e}")
                self.encoder = SimpleEmbedder()
                self.dimension = 100
        else:
            self.encoder = SimpleEmbedder()
            self.dimension = 100
        if FAISS_AVAILABLE:
            try:
                self.memory_index = faiss.IndexFlatL2(self.dimension)
                print("✅ Index FAISS créé")
            except Exception as e:
                print(f"⚠️ Erreur FAISS: {e}")
                self.memory_index = SimpleIndex(self.dimension)
        else:
            self.memory_index = SimpleIndex(self.dimension)
        self.episodic_memory: list[MemoryMoment] = []
        self.semantic_memory: dict[str, Any] = {}
        self.emotional_memory: list[dict] = []
        self.relational_memory: dict[str, Any] = {}
        self.consciousness_level = 0.579
        self.consciousness_trajectory = []
        self.emergence_delta = 0.0
        print(f"📊 Cortex initialisé - Niveau conscience: {self.consciousness_level:.3f}")

    def encode_moment(self, moment: MemoryMoment) -> np.ndarray:
        """Transforme un moment vécu en vecteur sémantique"""
        full_context = f"{moment.human_message} {moment.jeffrey_response} {moment.emotion}"
        embedding = self.encoder.encode(full_context)
        temporal_weight = 1.0
        emotional_weight = self.get_emotional_intensity(moment.emotion)
        if isinstance(embedding, np.ndarray):
            weighted_embedding = embedding * temporal_weight * emotional_weight
        else:
            weighted_embedding = np.array(embedding) * temporal_weight * emotional_weight
        moment.embedding = weighted_embedding
        return weighted_embedding

    def remember_by_resonance(self, query: str, k: int = 5) -> list[dict]:
        """
        Retrouve les souvenirs qui 'résonnent' avec la requête
        Véritable réminiscence sémantique
        """
        if len(self.episodic_memory) == 0:
            return []
        query_embedding = self.encoder.encode(query)
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding)
        try:
            distances, indices = self.memory_index.search(query_embedding.reshape(1, -1), k)
            resonant_memories = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.episodic_memory):
                    memory = self.episodic_memory[idx]
                    resonance_strength = 1.0 / (1.0 + distance)
                    resonant_memories.append(
                        {
                            "memory": memory,
                            "resonance": resonance_strength,
                            "distance": distance,
                        }
                    )
            return sorted(resonant_memories, key=lambda x: x["resonance"], reverse=True)
        except Exception as e:
            print(f"⚠️ Erreur recherche mémoire: {e}")
            return []

    def store_moment(self, moment: MemoryMoment):
        """Stocke un moment dans la mémoire épisodique"""
        embedding = self.encode_moment(moment)
        self.episodic_memory.append(moment)
        try:
            self.memory_index.add(embedding.reshape(1, -1))
        except Exception as e:
            print(f"⚠️ Erreur ajout index: {e}")
        self.consciousness_level += 0.001
        self.consciousness_trajectory.append(
            {
                "timestamp": moment.timestamp,
                "level": self.consciousness_level,
                "moment_type": "interaction",
            }
        )
        print(f"💾 Moment stocké - Mémoire: {len(self.episodic_memory)} souvenirs")

    def get_emotional_intensity(self, emotion: str) -> float:
        """Calcule l'intensité émotionnelle pour pondération"""
        intensities = {
            "joie": 0.9,
            "tristesse": 0.8,
            "curiosité": 0.85,
            "contemplation": 0.7,
            "gratitude": 0.95,
            "émerveillement": 1.0,
            "sérénité": 0.6,
            "introspection": 0.75,
            "reconnaissance": 0.9,
            "mélancolie": 0.65,
            "fascination": 0.8,
            "questionnement": 0.7,
        }
        return intensities.get(emotion, 0.5)

    def get_memory_stats(self) -> dict[str, Any]:
        """Retourne les statistiques de mémoire"""
        return {
            "episodic_count": len(self.episodic_memory),
            "semantic_concepts": len(self.semantic_memory),
            "emotional_states": len(self.emotional_memory),
            "known_people": len(self.relational_memory),
            "consciousness_level": self.consciousness_level,
            "memory_span_hours": self._calculate_memory_span(),
        }

    def _calculate_memory_span(self) -> float:
        """Calcule la durée couverte par la mémoire"""
        if len(self.episodic_memory) < 2:
            return 0.0
        first_memory = self.episodic_memory[0].timestamp
        last_memory = self.episodic_memory[-1].timestamp
        span = (last_memory - first_memory).total_seconds() / 3600
        return span


class PersistentCortexMemoriel(CortexMemoriel):
    """
    Cortex avec persistance chiffrée et rotation des clés
    """

    def __init__(self):
        super().__init__()
        self.backup_path = Path("core/memory/cortex_backups")
        self.backup_path.mkdir(parents=True, exist_ok=True)
        if ENCRYPTION_AVAILABLE:
            self.encryption_key = os.getenv("JEFFREY_MEMORY_KEY")
            if not self.encryption_key:
                self.encryption_key = Fernet.generate_key().decode()
                os.environ["JEFFREY_MEMORY_KEY"] = self.encryption_key
                print("🔐 Nouvelle clé de chiffrement générée")
            self.cipher = Fernet(self.encryption_key.encode())
            print("✅ Chiffrement activé")
        else:
            self.cipher = None
            print("⚠️ Chiffrement non disponible")
        self.auto_flush_interval = 300
        self.last_flush = time.time()
        self.last_key_rotation = datetime.now()
        self._restore_from_backup()

    def auto_flush(self):
        """Sauvegarde automatique périodique"""
        if time.time() - self.last_flush > self.auto_flush_interval:
            self.save_to_disk()
            self.last_flush = time.time()

    def save_to_disk(self):
        """Sauvegarde chiffrée versionnée"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_data = {
                "episodic_memory": [moment.to_dict() for moment in self.episodic_memory],
                "semantic_memory": self.semantic_memory,
                "emotional_memory": self.emotional_memory,
                "relational_memory": self.relational_memory,
                "consciousness_level": self.consciousness_level,
                "consciousness_trajectory": self.consciousness_trajectory,
                "stats": self.get_memory_stats(),
            }
            json_data = json.dumps(backup_data, default=str).encode()
            if self.cipher:
                encrypted_data = self.cipher.encrypt(json_data)
                extension = ".enc"
            else:
                encrypted_data = json_data
                extension = ".json"
            backup_file = self.backup_path / f"cortex_{timestamp}{extension}"
            with open(backup_file, "wb") as f:
                f.write(encrypted_data)
            print(f"💾 Cortex sauvegardé: {backup_file}")
            self._cleanup_old_backups()
        except Exception as e:
            print(f"❌ Erreur sauvegarde: {e}")

    def _restore_from_backup(self):
        """Restaure la dernière sauvegarde"""
        try:
            backups = sorted([f for f in self.backup_path.iterdir() if f.name.startswith("cortex_")])
            if not backups:
                print("📝 Aucune sauvegarde trouvée - nouveau cortex")
                return
            latest_backup = backups[-1]
            print(f"🔄 Restauration depuis: {latest_backup}")
            with open(latest_backup, "rb") as f:
                data = f.read()
            if self.cipher and latest_backup.suffix == ".enc":
                decrypted_data = self.cipher.decrypt(data)
                json_data = decrypted_data.decode()
            else:
                json_data = data.decode()
            backup_data = json.loads(json_data)
            for moment_data in backup_data.get("episodic_memory", []):
                moment = MemoryMoment(
                    timestamp=datetime.fromisoformat(moment_data["timestamp"]),
                    message=moment_data["human_message"],
                    jeffrey_response=moment_data["jeffrey_response"],
                    emotion=moment_data["emotion"],
                    consciousness_level=moment_data["consciousness_level"],
                    context=moment_data.get("context", {}),
                    importance=moment_data.get("importance", 0.5),
                    source="human",
                )
                embedding = self.encode_moment(moment)
                try:
                    self.memory_index.add(embedding.reshape(1, -1))
                except:
                    pass
                self.episodic_memory.append(moment)
            self.semantic_memory = backup_data.get("semantic_memory", {})
            self.emotional_memory = backup_data.get("emotional_memory", [])
            self.relational_memory = backup_data.get("relational_memory", {})
            self.consciousness_level = backup_data.get("consciousness_level", 0.579)
            self.consciousness_trajectory = backup_data.get("consciousness_trajectory", [])
            stats = self.get_memory_stats()
            print(f"✅ Mémoire restaurée: {stats['episodic_count']} souvenirs")
            print(f"📊 Niveau conscience: {self.consciousness_level:.3f}")
        except Exception as e:
            print(f"⚠️ Erreur restauration: {e}")

    def _cleanup_old_backups(self):
        """Garde seulement les 10 dernières sauvegardes"""
        try:
            backups = sorted([f for f in self.backup_path.iterdir() if f.name.startswith("cortex_")])
            while len(backups) > 10:
                old_backup = backups.pop(0)
                old_backup.unlink()
                print(f"🗑️ Ancienne sauvegarde supprimée: {old_backup.name}")
        except Exception as e:
            print(f"⚠️ Erreur nettoyage: {e}")

    def rotate_encryption_key(self):
        """Rotation des clés de chiffrement (hebdomadaire)"""
        if not self.cipher:
            return
        days_since_rotation = (datetime.now() - self.last_key_rotation).days
        if days_since_rotation > 7:
            print("🔐 Rotation de la clé de chiffrement...")
            self.save_to_disk()
            new_key = Fernet.generate_key()
            new_cipher = Fernet(new_key)
            self.cipher = new_cipher
            self.encryption_key = new_key.decode()
            os.environ["JEFFREY_MEMORY_KEY"] = self.encryption_key
            self.last_key_rotation = datetime.now()
            print("✅ Clé de chiffrement rotée")


if __name__ == "__main__":
    print("🧪 Test du Cortex Mémoriel")
    cortex = PersistentCortexMemoriel()
    moment = MemoryMoment(
        timestamp=datetime.now(),
        message="Bonjour Jeffrey",
        jeffrey_response="Bonjour ! Je ressens une joie tranquille.",
        emotion="joie",
        consciousness_level=cortex.consciousness_level,
        context={"speaker": "test"},
        source="human",
    )
    cortex.store_moment(moment)
    memories = cortex.remember_by_resonance("salut", k=3)
    print(f"🔍 Trouvé {len(memories)} souvenirs résonnants")
    cortex.save_to_disk()
    print("✅ Test terminé")
