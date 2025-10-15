# Jeffrey Core - Architecture P2

## Structure des Modules

### 🧠 `/bus` - Neural Bus
Bus de messages événementiel basé sur NATS JetStream.
Gère la communication asynchrone entre tous les modules.

### 🎯 `/kernel` - Brain Kernel
Noyau central de décision et orchestration.
Implémente la logique AGI principale.

### 💭 `/consciousness` - Consciousness V3
Module de conscience et meta-cognition.
Migré depuis P1 avec améliorations.

### 💾 `/memory` - Memory Manager
Gestion de la mémoire épisodique et sémantique.
Cache L1/L2 avec invalidation événementielle.

### ❤️ `/emotions` - Emotional Core
Moteur émotionnel avec états persistants.
Influence les décisions et réponses.

### 📚 `/learning` - Learning Engine
Apprentissage continu et adaptation.
Intégration future avec knowledge graph.

### 💤 `/dream` - Dream Engine
Consolidation mémoire et créativité.
Processing offline et génération d'insights.

### 🔄 `/saga` - Saga Orchestrator
Pattern Saga pour transactions distribuées.
Gestion des workflows complexes multi-modules.

## Migration depuis P1

Les modules P1 sont préservés dans `/legacy` et seront migrés progressivement.
Utilisez le pattern Adapter pour la transition.

## Tests

- Unit tests: `pytest tests/unit/core/`
- Integration: `pytest tests/integration/`
- Chaos: `pytest tests/chaos/`
