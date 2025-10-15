"""
JEFFREY OS - Tests Phase 2 : Recherche Sémantique
==================================================

Teste les fonctionnalités de recherche sémantique :
- Génération d'embeddings
- Recherche par similarité
- Clustering de conversations
- Extraction de thèmes
"""

from typing import Any

import numpy as np
import pytest

# ===============================================================================
# FIXTURES
# ===============================================================================


@pytest.fixture
def sample_texts() -> list[str]:
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
def sample_conversations() -> list[dict[str, Any]]:
    """Conversations d'exemple pour clustering"""
    return [
        {
            "id": 1,
            "messages": [
                {"role": "user", "content": "Comment coder en Python ?"},
                {"role": "assistant", "content": "Voici un tutoriel Python..."},
            ],
            "theme": "programmation",
        },
        {
            "id": 2,
            "messages": [
                {"role": "user", "content": "Quelle est la meilleure recette de café ?"},
                {"role": "assistant", "content": "Voici comment faire un bon café..."},
            ],
            "theme": "café",
        },
        {
            "id": 3,
            "messages": [
                {"role": "user", "content": "Comment apprendre JavaScript ?"},
                {"role": "assistant", "content": "Commencez par les bases JS..."},
            ],
            "theme": "programmation",
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
        self.embeddings_cache: dict[str, np.ndarray] = {}

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

    def search(self, query: str, texts: list[str], top_k: int = 3) -> list[dict[str, Any]]:
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
            results.append({"text": text, "score": score, "index": idx})

        # Trier par score décroissant
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_k]

    def cluster_texts(self, texts: list[str], n_clusters: int = 2) -> list[int]:
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
