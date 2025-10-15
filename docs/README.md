# 📚 JEFFREY OS - DOCUMENTATION COMPLÈTE

**Version** : 2.0.0
**Date** : Octobre 2025

## 🎯 VUE D'ENSEMBLE

Jeffrey OS est un orchestrateur d'IA avec mémoire émotionnelle avancée qui combine :
- Recherche hybride (keyword + semantic)
- Clustering thématique automatique
- Apprentissage adaptatif par feedback
- Explainability totale

## 🚀 INSTALLATION

```bash
# Dépendances de base
pip install -r requirements.txt

# Features avancées (recommandé)
pip install "sentence-transformers<3.0" "torch>=2.1,<2.4"
pip install "scikit-learn>=1.3,<1.6"
```

## 📖 UTILISATION

### Exemple minimal

```python
from jeffrey.memory.unified_memory import UnifiedMemory

# Init
memory = UnifiedMemory()

# Ajouter
memory.add_memory({
    "user_id": "alice",
    "content": "J'adore le jazz"
})

# Rechercher
results = memory.search_memories("alice", "musique")
print(results[0]['memory']['content'])
```

### Exemple complet

```python
from jeffrey.memory.unified_memory import UnifiedMemory

# Init avec semantic search (auto-détection)
memory = UnifiedMemory(enable_vector=None)

# Ajouter avec métadonnées
memory.add_memory({
    "user_id": "alice",
    "content": "Réunion importante avec le CEO demain 10h",
    "type": "reminder",
    "tags": ["travail", "urgent"],
    "importance": 0.9,
    "emotion": "neutral"
})

# Recherche avancée
results = memory.search_memories(
    user_id="alice",
    query="réunion importante",
    filters={"type": "reminder"},
    semantic_search=True,
    explain=True,
    limit=5
)

# Explorer résultats
for r in results:
    print(f"[{r['score']:.3f}] {r['memory']['content']}")
    print(f"  Raisons: {', '.join(r['explanation']['reasons'])}")
```

## 🔍 API REFERENCE

### UnifiedMemory

#### Constructeur

```python
UnifiedMemory(
    enable_vector: bool = None,    # None=auto, True=force, False=disable
    temporal_mode: str = "recent_bias",
    default_limit: int = 10
)
```

#### Méthodes principales

**add_memory(data: Dict) → str**

Ajoute un souvenir. Champs obligatoires : `user_id`, `content`.

```python
mem_id = memory.add_memory({
    "user_id": "alice",
    "content": "Texte du souvenir",
    "type": "note",              # optionnel
    "tags": ["tag1", "tag2"],    # optionnel
    "emotion": "joy",            # optionnel
    "importance": 0.7            # optionnel (0.0-1.0)
})
```

**search_memories(user_id, query, **kwargs) → List[Dict]**

Recherche de souvenirs.

```python
results = memory.search_memories(
    user_id="alice",
    query="projet",
    queries=["projet", "urgent"],        # multi-query
    combine_strategy="union",             # "union" | "intersection"
    filters={"type": "task"},
    field_boosts={"tags": 0.3},
    semantic_search=True,
    cluster_results=False,
    limit=10,
    explain=True
)
```

**get_clusters(user_id) → Dict**

Obtenir les clusters thématiques.

```python
clusters = memory.get_clusters("alice")
# {0: {"theme": "musique jazz", "size": 12}, ...}
```

**feedback(user_id, shown_ids, clicked_ids)**

Apprentissage par feedback.

```python
# Après affichage de résultats
shown = [r["memory"]["id"] for r in results]
clicked = [results[2]["memory"]["id"]]  # User clique sur le 3ème

memory.feedback("alice", shown_ids=shown, clicked_ids=clicked)
```

**stats(user_id) → Dict**

Statistiques du système.

```python
stats = memory.stats("alice")
# {"storage": {...}, "vector_index": {...}, "clustering": {...}}
```

## 🧪 TESTS

### Tests unitaires

```bash
PYTHONPATH=src python3 tests/test_unified_memory.py
PYTHONPATH=src python3 tests/test_semantic_search.py
PYTHONPATH=src python3 tests/test_phase3_advanced_memory.py
```

### Tests conversationnels

```bash
PYTHONPATH=src python3 tests/runner_convos.py
```

## 📊 ARCHITECTURE

### Composants

- **UnifiedMemory** : Système de mémoire hybride
- **VectorIndex** : Embeddings sémantiques (sentence-transformers)
- **ClusterEngine** : Découverte thématique (MiniBatchKMeans)
- **StorageAdapter** : Interface de stockage (extensible)

### Scoring MCDM

Chaque souvenir est évalué selon 5 critères :

1. **Text (40%)** : Pertinence textuelle + sémantique
2. **Emotion (20%)** : Correspondance émotionnelle
3. **Temporal (20%)** : Récence du souvenir
4. **Frequency (10%)** : Nombre d'accès
5. **Importance (10%)** : Importance déclarée

**Ces poids s'adaptent via feedback utilisateur !**

## 📈 MÉTRIQUES

- **Tests unitaires** : 20/20 ✅
- **Tests conversationnels** : 40+ scénarios
- **Performance** : < 50ms recherche (1000 mémoires)
- **Couverture** : 1000+ tours de conversation

## 🏆 PHASES COMPLÉTÉES

- ✅ Phase 1 : Système de base (MCDM, index inversé)
- ✅ Phase 2 : Embeddings sémantiques (sentence-transformers)
- ✅ Phase 3 : Clustering + Learning-to-Rank + Multi-Query

## 📝 LICENCE

MIT
