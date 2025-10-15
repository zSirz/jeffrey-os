"""
JEFFREY OS - Tests Phase 1 : Mémoire Unifiée
==============================================

Teste les fonctionnalités de base de la mémoire unifiée :
- Ajout de messages
- Récupération d'historique
- Persistance et chargement
- Gestion des métadonnées émotionnelles
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

# ===============================================================================
# FIXTURES
# ===============================================================================


@pytest.fixture
def temp_memory_dir(tmp_path):
    """Crée un répertoire temporaire pour les tests de mémoire"""
    memory_dir = tmp_path / "jeffrey_memory"
    memory_dir.mkdir()
    return memory_dir


@pytest.fixture
def sample_message() -> dict[str, Any]:
    """Crée un message type pour les tests"""
    return {
        "role": "user",
        "content": "Bonjour Jeffrey, comment vas-tu ?",
        "timestamp": datetime.now().isoformat(),
        "metadata": {"emotion": "neutral", "confidence": 0.8, "importance": 0.5},
    }


# ===============================================================================
# CLASSE SIMPLIFIÉE UnifiedMemory POUR LES TESTS
# ===============================================================================


class UnifiedMemory:
    """
    Version simplifiée de la mémoire unifiée pour tests unitaires.

    Dans le vrai Jeffrey OS, cette classe sera bien plus complexe
    avec clustering, embeddings, recherche sémantique, etc.
    """

    def __init__(self, memory_dir: Path):
        self.memory_dir = Path(memory_dir)
        self.memory_file = self.memory_dir / "unified_memory.json"
        self.messages: list[dict[str, Any]] = []

        # Créer le fichier de mémoire s'il n'existe pas
        if not self.memory_file.exists():
            self._save()

    def add_message(self, message: dict[str, Any]) -> None:
        """Ajoute un message à la mémoire"""
        # Validation basique
        if "role" not in message or "content" not in message:
            raise ValueError("Message doit contenir 'role' et 'content'")

        # Ajout du timestamp si absent
        if "timestamp" not in message:
            message["timestamp"] = datetime.now().isoformat()

        self.messages.append(message)
        self._save()

    def get_recent(self, n: int = 10) -> list[dict[str, Any]]:
        """Récupère les n derniers messages"""
        return self.messages[-n:]

    def get_all(self) -> list[dict[str, Any]]:
        """Récupère tous les messages"""
        return self.messages.copy()

    def search(self, query: str) -> list[dict[str, Any]]:
        """Recherche simple par contenu (pour Phase 1, simplifié)"""
        results = []
        query_lower = query.lower()
        for msg in self.messages:
            if query_lower in msg["content"].lower():
                results.append(msg)
        return results

    def clear(self) -> None:
        """Vide la mémoire"""
        self.messages = []
        self._save()

    def load(self) -> None:
        """Charge la mémoire depuis le disque"""
        if self.memory_file.exists():
            with open(self.memory_file, encoding='utf-8') as f:
                data = json.load(f)
                self.messages = data.get("messages", [])

    def _save(self) -> None:
        """Sauvegarde la mémoire sur disque"""
        data = {"messages": self.messages, "last_updated": datetime.now().isoformat()}
        with open(self.memory_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# ===============================================================================
# TESTS PHASE 1
# ===============================================================================


def test_memory_initialization(temp_memory_dir):
    """Teste l'initialisation de la mémoire"""
    memory = UnifiedMemory(temp_memory_dir)
    assert memory.memory_dir == temp_memory_dir
    assert memory.memory_file.exists()
    assert len(memory.messages) == 0


def test_add_single_message(temp_memory_dir, sample_message):
    """Teste l'ajout d'un message unique"""
    memory = UnifiedMemory(temp_memory_dir)
    memory.add_message(sample_message)

    assert len(memory.messages) == 1
    assert memory.messages[0]["content"] == sample_message["content"]
    assert memory.messages[0]["role"] == sample_message["role"]


def test_add_multiple_messages(temp_memory_dir):
    """Teste l'ajout de plusieurs messages"""
    memory = UnifiedMemory(temp_memory_dir)

    messages = [
        {"role": "user", "content": "Message 1"},
        {"role": "assistant", "content": "Réponse 1"},
        {"role": "user", "content": "Message 2"},
    ]

    for msg in messages:
        memory.add_message(msg)

    assert len(memory.messages) == 3
    assert memory.messages[0]["content"] == "Message 1"
    assert memory.messages[2]["content"] == "Message 2"


def test_get_recent_messages(temp_memory_dir):
    """Teste la récupération des messages récents"""
    memory = UnifiedMemory(temp_memory_dir)

    # Ajouter 15 messages
    for i in range(15):
        memory.add_message({"role": "user", "content": f"Message {i}"})

    # Récupérer les 5 derniers
    recent = memory.get_recent(5)
    assert len(recent) == 5
    assert recent[-1]["content"] == "Message 14"
    assert recent[0]["content"] == "Message 10"


def test_message_persistence(temp_memory_dir, sample_message):
    """Teste la persistance des messages sur disque"""
    # Créer mémoire et ajouter message
    memory1 = UnifiedMemory(temp_memory_dir)
    memory1.add_message(sample_message)

    # Créer nouvelle instance et charger
    memory2 = UnifiedMemory(temp_memory_dir)
    memory2.load()

    assert len(memory2.messages) == 1
    assert memory2.messages[0]["content"] == sample_message["content"]


def test_search_messages(temp_memory_dir):
    """Teste la recherche dans les messages"""
    memory = UnifiedMemory(temp_memory_dir)

    memory.add_message({"role": "user", "content": "J'aime Python"})
    memory.add_message({"role": "user", "content": "J'aime JavaScript"})
    memory.add_message({"role": "user", "content": "Je déteste les bugs"})

    results = memory.search("aime")
    assert len(results) == 2
    assert "Python" in results[0]["content"] or "JavaScript" in results[0]["content"]


def test_clear_memory(temp_memory_dir, sample_message):
    """Teste la suppression de la mémoire"""
    memory = UnifiedMemory(temp_memory_dir)
    memory.add_message(sample_message)

    assert len(memory.messages) == 1

    memory.clear()
    assert len(memory.messages) == 0


def test_message_validation(temp_memory_dir):
    """Teste la validation des messages"""
    memory = UnifiedMemory(temp_memory_dir)

    # Message invalide (sans content)
    with pytest.raises(ValueError):
        memory.add_message({"role": "user"})

    # Message invalide (sans role)
    with pytest.raises(ValueError):
        memory.add_message({"content": "Test"})


def test_metadata_preservation(temp_memory_dir, sample_message):
    """Teste la préservation des métadonnées"""
    memory = UnifiedMemory(temp_memory_dir)
    memory.add_message(sample_message)

    retrieved = memory.messages[0]
    assert "metadata" in retrieved
    assert retrieved["metadata"]["emotion"] == "neutral"
    assert retrieved["metadata"]["confidence"] == 0.8


# ===============================================================================
# TESTS DE PERFORMANCE (BASIQUES)
# ===============================================================================


def test_large_memory_performance(temp_memory_dir):
    """Teste les performances avec beaucoup de messages"""
    memory = UnifiedMemory(temp_memory_dir)

    # Ajouter 1000 messages
    for i in range(1000):
        memory.add_message({"role": "user", "content": f"Message numéro {i}"})

    assert len(memory.messages) == 1000

    # Vérifier que la recherche reste rapide
    import time

    start = time.time()
    results = memory.search("500")
    elapsed = time.time() - start

    assert len(results) == 1
    assert elapsed < 0.1  # Moins de 100ms


# ===============================================================================
# EXÉCUTION DES TESTS
# ===============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
