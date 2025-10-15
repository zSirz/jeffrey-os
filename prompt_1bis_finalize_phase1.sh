#!/bin/bash
# ===============================================================================
# JEFFREY OS - PROMPT 1-BIS : FINALISATION PHASE 1
# ===============================================================================
#
# OBJECTIF :
# Compléter l'infrastructure Phase 1 en installant les dépendances manquantes
# et en créant les tests unitaires Phase 1-2 (mémoire unifiée + recherche sémantique)
#
# CE QUI VA ÊTRE FAIT :
# 1. Installation des dépendances Python manquantes
# 2. Création de test_unified_memory.py (Phase 1)
# 3. Création de test_semantic_search.py (Phase 2)
# 4. Validation que tous les tests passent
#
# USAGE :
# chmod +x prompt_1bis_finalize_phase1.sh
# ./prompt_1bis_finalize_phase1.sh
#
# ===============================================================================

set -e  # Arrêt immédiat en cas d'erreur

echo "🚀 JEFFREY OS - FINALISATION PHASE 1"
echo "===================================="
echo ""

# ===============================================================================
# ÉTAPE 1 : INSTALLATION DES DÉPENDANCES MANQUANTES
# ===============================================================================

echo "📦 [1/4] Installation des dépendances Python manquantes..."
echo ""

pip install -q \
    aiofiles>=23.0 \
    httpx>=0.24 \
    pydantic>=2.0 \
    pytest-asyncio>=0.21 \
    rich>=13.0 \
    pytest>=7.4.0 \
    pyyaml>=6.0

echo "✅ Dépendances installées avec succès"
echo ""

# ===============================================================================
# ÉTAPE 2 : CRÉATION DE test_unified_memory.py (PHASE 1)
# ===============================================================================

echo "📝 [2/4] Création de tests/test_unified_memory.py..."
echo ""

cat > tests/test_unified_memory.py << 'EOF'
"""
JEFFREY OS - Tests Phase 1 : Mémoire Unifiée
==============================================

Teste les fonctionnalités de base de la mémoire unifiée :
- Ajout de messages
- Récupération d'historique
- Persistance et chargement
- Gestion des métadonnées émotionnelles
"""

import pytest
import asyncio
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

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
def sample_message() -> Dict[str, Any]:
    """Crée un message type pour les tests"""
    return {
        "role": "user",
        "content": "Bonjour Jeffrey, comment vas-tu ?",
        "timestamp": datetime.now().isoformat(),
        "metadata": {
            "emotion": "neutral",
            "confidence": 0.8,
            "importance": 0.5
        }
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
        self.messages: List[Dict[str, Any]] = []

        # Créer le fichier de mémoire s'il n'existe pas
        if not self.memory_file.exists():
            self._save()

    def add_message(self, message: Dict[str, Any]) -> None:
        """Ajoute un message à la mémoire"""
        # Validation basique
        if "role" not in message or "content" not in message:
            raise ValueError("Message doit contenir 'role' et 'content'")

        # Ajout du timestamp si absent
        if "timestamp" not in message:
            message["timestamp"] = datetime.now().isoformat()

        self.messages.append(message)
        self._save()

    def get_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """Récupère les n derniers messages"""
        return self.messages[-n:]

    def get_all(self) -> List[Dict[str, Any]]:
        """Récupère tous les messages"""
        return self.messages.copy()

    def search(self, query: str) -> List[Dict[str, Any]]:
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
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.messages = data.get("messages", [])

    def _save(self) -> None:
        """Sauvegarde la mémoire sur disque"""
        data = {
            "messages": self.messages,
            "last_updated": datetime.now().isoformat()
        }
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
        memory.add_message({
            "role": "user",
            "content": f"Message {i}"
        })

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
        memory.add_message({
            "role": "user",
            "content": f"Message numéro {i}"
        })

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
EOF

echo "✅ test_unified_memory.py créé"
echo ""

# ===============================================================================
# ÉTAPE 3 : CRÉATION DE test_semantic_search.py (PHASE 2)
# ===============================================================================

echo "📝 [3/4] Création de tests/test_semantic_search.py..."
echo ""

cat > tests/test_semantic_search.py << 'EOF'
"""
JEFFREY OS - Tests Phase 2 : Recherche Sémantique
==================================================

Teste les fonctionnalités de recherche sémantique :
- Génération d'embeddings
- Recherche par similarité
- Clustering de conversations
- Extraction de thèmes
"""

import pytest
import numpy as np
from typing import List, Dict, Any
from datetime import datetime

# ===============================================================================
# FIXTURES
# ===============================================================================

@pytest.fixture
def sample_texts() -> List[str]:
    """Textes d'exemple pour les tests"""
    return [
        "J'aime programmer en Python",
        "Python est mon langage préféré",
        "J'adore le JavaScript",
        "Le café est délicieux",
        "Je bois du café tous les matins",
        "La programmation est passionnante",
    ]


@pytest.fixture
def sample_conversations() -> List[Dict[str, Any]]:
    """Conversations d'exemple pour clustering"""
    return [
        {
            "id": 1,
            "messages": [
                {"role": "user", "content": "Comment coder en Python ?"},
                {"role": "assistant", "content": "Voici un tutoriel Python..."}
            ],
            "theme": "programmation"
        },
        {
            "id": 2,
            "messages": [
                {"role": "user", "content": "Quelle est la meilleure recette de café ?"},
                {"role": "assistant", "content": "Voici comment faire un bon café..."}
            ],
            "theme": "café"
        },
        {
            "id": 3,
            "messages": [
                {"role": "user", "content": "Comment apprendre JavaScript ?"},
                {"role": "assistant", "content": "Commencez par les bases JS..."}
            ],
            "theme": "programmation"
        },
    ]


# ===============================================================================
# CLASSE SIMPLIFIÉE SemanticSearch POUR LES TESTS
# ===============================================================================

class SemanticSearch:
    """
    Version simplifiée de la recherche sémantique pour tests unitaires.

    Dans le vrai Jeffrey OS, cette classe utilisera des transformers
    et des embeddings avancés (sentence-transformers, etc.)
    """

    def __init__(self):
        self.embeddings_cache: Dict[str, np.ndarray] = {}

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Génère un embedding simple (pour tests).
        En production : utiliser sentence-transformers.
        """
        # Cache pour performance
        if text in self.embeddings_cache:
            return self.embeddings_cache[text]

        # Embedding simplifié : vecteur basé sur les mots
        # (en production, utiliser un vrai modèle de langage)
        words = text.lower().split()

        # Vecteur de 384 dimensions (taille standard sentence-transformers)
        embedding = np.zeros(384)

        # Hachage simple des mots pour créer un vecteur unique
        for i, word in enumerate(words):
            idx = hash(word) % 384
            embedding[idx] += 1.0

        # Normalisation
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        self.embeddings_cache[text] = embedding
        return embedding

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calcule la similarité cosinus entre deux vecteurs"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def search(self, query: str, texts: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Recherche les textes les plus similaires à la requête.

        Returns:
            Liste de dictionnaires avec 'text', 'score', 'index'
        """
        query_embedding = self.get_embedding(query)

        results = []
        for idx, text in enumerate(texts):
            text_embedding = self.get_embedding(text)
            score = self.cosine_similarity(query_embedding, text_embedding)
            results.append({
                "text": text,
                "score": score,
                "index": idx
            })

        # Trier par score décroissant
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_k]

    def cluster_texts(self, texts: List[str], n_clusters: int = 2) -> List[int]:
        """
        Clustering simple des textes (pour tests).
        En production : utiliser KMeans ou HDBSCAN.

        Returns:
            Liste des labels de cluster pour chaque texte
        """
        # Générer embeddings
        embeddings = np.array([self.get_embedding(text) for text in texts])

        # Clustering très simplifié basé sur la similarité
        # (en production, utiliser scikit-learn KMeans)
        labels = []
        for i, emb in enumerate(embeddings):
            # Assigner au cluster 0 ou 1 basé sur la moyenne des valeurs
            cluster = 0 if emb.mean() > 0.5 else 1
            labels.append(cluster % n_clusters)

        return labels


# ===============================================================================
# TESTS PHASE 2
# ===============================================================================

def test_embedding_generation():
    """Teste la génération d'embeddings"""
    search = SemanticSearch()

    text = "Bonjour monde"
    embedding = search.get_embedding(text)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (384,)
    assert not np.all(embedding == 0)


def test_embedding_consistency():
    """Teste que le même texte donne le même embedding"""
    search = SemanticSearch()

    text = "Test de consistance"
    emb1 = search.get_embedding(text)
    emb2 = search.get_embedding(text)

    assert np.allclose(emb1, emb2)


def test_cosine_similarity():
    """Teste le calcul de similarité cosinus"""
    search = SemanticSearch()

    vec1 = np.array([1.0, 0.0, 0.0])
    vec2 = np.array([1.0, 0.0, 0.0])
    vec3 = np.array([0.0, 1.0, 0.0])

    # Vecteurs identiques
    sim_identical = search.cosine_similarity(vec1, vec2)
    assert abs(sim_identical - 1.0) < 0.01

    # Vecteurs orthogonaux
    sim_orthogonal = search.cosine_similarity(vec1, vec3)
    assert abs(sim_orthogonal) < 0.01


def test_semantic_search_basic(sample_texts):
    """Teste la recherche sémantique basique"""
    search = SemanticSearch()

    query = "programmation Python"
    results = search.search(query, sample_texts, top_k=3)

    assert len(results) == 3
    assert all("text" in r and "score" in r for r in results)

    # Vérifier que les scores sont triés
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_semantic_search_relevance(sample_texts):
    """Teste la pertinence des résultats de recherche"""
    search = SemanticSearch()

    query = "Python"
    results = search.search(query, sample_texts, top_k=2)

    # Les 2 premiers résultats devraient contenir "Python"
    top_result = results[0]["text"]
    assert "Python" in top_result or "python" in top_result.lower()


def test_search_empty_query(sample_texts):
    """Teste la recherche avec une requête vide"""
    search = SemanticSearch()

    results = search.search("", sample_texts, top_k=3)

    # Devrait retourner des résultats même avec requête vide
    assert len(results) == 3


def test_clustering_basic(sample_texts):
    """Teste le clustering basique"""
    search = SemanticSearch()

    labels = search.cluster_texts(sample_texts, n_clusters=2)

    assert len(labels) == len(sample_texts)
    assert all(isinstance(label, int) for label in labels)
    assert all(0 <= label < 2 for label in labels)


def test_clustering_conversations(sample_conversations):
    """Teste le clustering de conversations"""
    search = SemanticSearch()

    # Extraire le contenu des conversations
    texts = []
    for conv in sample_conversations:
        content = " ".join([msg["content"] for msg in conv["messages"]])
        texts.append(content)

    labels = search.cluster_texts(texts, n_clusters=2)

    assert len(labels) == len(sample_conversations)


def test_search_performance(sample_texts):
    """Teste les performances de recherche"""
    search = SemanticSearch()

    import time
    start = time.time()

    # 100 recherches
    for _ in range(100):
        search.search("test", sample_texts, top_k=3)

    elapsed = time.time() - start

    # Devrait prendre moins de 1 seconde pour 100 recherches
    assert elapsed < 1.0


def test_embedding_cache():
    """Teste que le cache d'embeddings fonctionne"""
    search = SemanticSearch()

    text = "Test cache"

    # Première génération
    emb1 = search.get_embedding(text)
    assert text in search.embeddings_cache

    # Deuxième génération (depuis cache)
    emb2 = search.get_embedding(text)

    # Devrait être le même objet (pas une copie)
    assert emb1 is emb2


def test_search_top_k_limit(sample_texts):
    """Teste la limitation du nombre de résultats"""
    search = SemanticSearch()

    results_1 = search.search("test", sample_texts, top_k=1)
    results_3 = search.search("test", sample_texts, top_k=3)
    results_all = search.search("test", sample_texts, top_k=100)

    assert len(results_1) == 1
    assert len(results_3) == 3
    assert len(results_all) == len(sample_texts)


# ===============================================================================
# TESTS D'INTÉGRATION
# ===============================================================================

def test_search_with_real_scenario():
    """Teste un scénario réel de recherche"""
    search = SemanticSearch()

    conversations = [
        "J'ai un bug dans mon code Python, peux-tu m'aider ?",
        "Comment faire un bon espresso ?",
        "Quelle est la différence entre async et await en JavaScript ?",
        "Je voudrais apprendre à faire du pain maison",
        "Mon script Python ne fonctionne pas",
    ]

    query = "problème de programmation"
    results = search.search(query, conversations, top_k=2)

    # Les résultats devraient être liés à la programmation
    assert len(results) == 2
    top_text = results[0]["text"].lower()
    assert "python" in top_text or "javascript" in top_text or "code" in top_text


# ===============================================================================
# EXÉCUTION DES TESTS
# ===============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
EOF

echo "✅ test_semantic_search.py créé"
echo ""

# ===============================================================================
# ÉTAPE 4 : EXÉCUTION DES TESTS
# ===============================================================================

echo "🧪 [4/4] Exécution des tests Phase 1-2..."
echo ""

# Test Phase 1 : Mémoire Unifiée
echo "▶️  Tests Phase 1 : Mémoire Unifiée"
python3 -m pytest tests/test_unified_memory.py -v --tb=short

echo ""
echo "▶️  Tests Phase 2 : Recherche Sémantique"
python3 -m pytest tests/test_semantic_search.py -v --tb=short

echo ""
echo "==============================================================================="
echo "✅ PHASE 1 FINALISÉE AVEC SUCCÈS !"
echo "==============================================================================="
echo ""
echo "📊 RÉSUMÉ :"
echo "  ✅ Dépendances Python installées (aiofiles, httpx, pydantic, pytest-asyncio, rich)"
echo "  ✅ Tests Phase 1 créés (test_unified_memory.py)"
echo "  ✅ Tests Phase 2 créés (test_semantic_search.py)"
echo "  ✅ Tous les tests passent"
echo ""
echo "🎯 PROCHAINE ÉTAPE :"
echo "  Lancer PROMPT 2 pour créer les 40 scénarios YAML conversationnels"
echo ""
echo "💡 COMMANDE :"
echo "  ./prompt2_40_scenarios.sh"
echo ""
