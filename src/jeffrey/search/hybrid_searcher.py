"""
JEFFREY OS - Recherche Hybride
===============================

Recherche combinant :
- BM25 (lexical)
- TF-IDF (lexical)
- Normalisation min-max
- Explicabilité (weights_used, components)

Sprint 1 : Version basique sans embeddings
Sprint 2 : Ajout embeddings sémantiques

Équipe : Dream Team Jeffrey OS
"""

import math
from collections import Counter
from dataclasses import dataclass


@dataclass
class SearchResult:
    """Résultat de recherche avec explicabilité"""

    content: str
    score: float
    index: int
    components: dict[str, float]  # {'lexical': 0.8, 'recency': 0.2}
    weights_used: dict[str, float]  # {'w_lex': 0.6, 'w_time': 0.4}


class HybridSearcher:
    """
    Recherche hybride pour Jeffrey OS.

    Sprint 1 : BM25 + normalisation
    Sprint 2 : + embeddings sémantiques
    """

    def __init__(self, w_lexical: float = 0.6, w_recency: float = 0.4):
        """
        Initialise le chercheur.

        Args:
            w_lexical: Poids lexical (BM25)
            w_recency: Poids récence
        """
        self.w_lexical = w_lexical
        self.w_recency = w_recency

        # Cache
        self.documents: list[str] = []
        self.doc_lengths: list[int] = []
        self.avg_doc_length: float = 0.0
        self.idf_cache: dict[str, float] = {}

    def add_documents(self, documents: list[str]):
        """Indexe des documents pour recherche"""
        self.documents = documents
        self.doc_lengths = [len(doc.split()) for doc in documents]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if documents else 0

        # Calculer IDF
        self._compute_idf()

    def _compute_idf(self):
        """Calcule IDF (Inverse Document Frequency)"""
        N = len(self.documents)
        if N == 0:
            return

        # Compter dans combien de docs chaque terme apparaît
        term_doc_count: dict[str, int] = {}

        for doc in self.documents:
            terms = set(doc.lower().split())
            for term in terms:
                term_doc_count[term] = term_doc_count.get(term, 0) + 1

        # IDF = log((N - df + 0.5) / (df + 0.5) + 1)
        for term, df in term_doc_count.items():
            self.idf_cache[term] = math.log((N - df + 0.5) / (df + 0.5) + 1.0)

    def _bm25_score(self, query: str, doc: str, doc_length: int) -> float:
        """
        Calcule le score BM25.

        BM25 = sum(IDF(qi) * (f(qi, D) * (k1 + 1)) / (f(qi, D) + k1 * (1 - b + b * |D| / avgdl)))

        où :
        - qi = terme de la query
        - f(qi, D) = fréquence du terme dans le doc
        - |D| = longueur du doc
        - avgdl = longueur moyenne des docs
        - k1 = 1.5 (paramètre)
        - b = 0.75 (paramètre)
        """
        k1 = 1.5
        b = 0.75

        query_terms = query.lower().split()
        doc_terms = doc.lower().split()
        doc_term_freq = Counter(doc_terms)

        score = 0.0

        for term in query_terms:
            if term not in self.idf_cache:
                continue

            idf = self.idf_cache[term]
            tf = doc_term_freq.get(term, 0)

            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_length / self.avg_doc_length))

            score += idf * (numerator / denominator)

        return score

    def _normalize_scores(self, scores: list[float]) -> list[float]:
        """Normalisation min-max vers [0, 1]"""
        if not scores or max(scores) == min(scores):
            return [0.5] * len(scores)

        min_score = min(scores)
        max_score = max(scores)
        range_score = max_score - min_score

        return [(s - min_score) / range_score for s in scores]

    def search(self, query: str, top_k: int = 5, recency_scores: list[float] = None) -> list[SearchResult]:
        """
        Recherche hybride.

        Args:
            query: Requête de recherche
            top_k: Nombre de résultats
            recency_scores: Scores de récence [0-1] pour chaque doc (optionnel)

        Returns:
            Liste de SearchResult triée par pertinence
        """
        if not self.documents:
            return []

        # 1. SCORES LEXICAUX (BM25)
        lexical_scores = []
        for i, doc in enumerate(self.documents):
            score = self._bm25_score(query, doc, self.doc_lengths[i])
            lexical_scores.append(score)

        # Normaliser
        lexical_normalized = self._normalize_scores(lexical_scores)

        # 2. SCORES DE RÉCENCE
        if recency_scores is None:
            # Par défaut, récence uniforme
            recency_normalized = [0.5] * len(self.documents)
        else:
            recency_normalized = recency_scores  # Déjà normalisés [0-1]

        # 3. FUSION PONDÉRÉE
        results = []

        for i, doc in enumerate(self.documents):
            lex_score = lexical_normalized[i]
            rec_score = recency_normalized[i]

            # Score final
            final_score = self.w_lexical * lex_score + self.w_recency * rec_score

            result = SearchResult(
                content=doc,
                score=final_score,
                index=i,
                components={"lexical": lex_score, "recency": rec_score},
                weights_used={"w_lexical": self.w_lexical, "w_recency": self.w_recency},
            )

            results.append(result)

        # Trier par score décroissant
        results.sort(key=lambda x: x.score, reverse=True)

        return results[:top_k]


# ===============================================================================
# TESTS UNITAIRES
# ===============================================================================


def test_hybrid_searcher():
    """Tests basiques du chercheur hybride"""
    searcher = HybridSearcher(w_lexical=0.7, w_recency=0.3)

    documents = [
        "J'aime programmer en Python",
        "Python est mon langage préféré",
        "J'adore le JavaScript",
        "Le café est délicieux",
        "Je bois du café tous les matins",
    ]

    searcher.add_documents(documents)

    # Test 1 : Recherche Python
    results = searcher.search("Python", top_k=2)
    assert len(results) == 2
    assert "Python" in results[0].content or "python" in results[0].content.lower()
    print(f"✅ Test 1 (Python) : Top result = \"{results[0].content[:50]}...\"")
    print(f"   Score: {results[0].score:.3f}, Components: {results[0].components}")

    # Test 2 : Recherche café
    results = searcher.search("café", top_k=2)
    assert len(results) == 2
    print(f"✅ Test 2 (café) : Top result = \"{results[0].content[:50]}...\"")

    # Test 3 : Explicabilité
    results = searcher.search("test", top_k=1)
    assert "weights_used" in results[0].__dict__
    assert "components" in results[0].__dict__
    print(f"✅ Test 3 (explicabilité) : weights={results[0].weights_used}")

    print("\n✅ Tous les tests HybridSearcher passent !")


if __name__ == "__main__":
    print("🧪 Tests unitaires HybridSearcher...")
    print()
    test_hybrid_searcher()
