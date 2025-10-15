#!/bin/bash
# Création de la structure P2 sans casser P1

set -e

echo "🏗️ Creating P2 structure..."

# Structure principale P2
mkdir -p src/jeffrey/{core,infrastructure,avatars,shared,tests,docs}

# Core modules (cerveau AGI)
mkdir -p src/jeffrey/core/{bus,kernel,consciousness,memory,emotions,learning,dream,saga}

# Knowledge Graph preparation (P3)
mkdir -p src/jeffrey/core/learning/kg

# Infrastructure
mkdir -p src/jeffrey/infrastructure/{nats,redis,security,monitoring,actors}

# Avatars (interfaces publiques)
mkdir -p src/jeffrey/avatars/{api,personas,vision,voice}

# Shared (utils communs)
mkdir -p src/jeffrey/shared/{contracts,utils,config,models}

# Tests structurés
mkdir -p tests/{unit,integration,e2e,chaos,benchmarks}

# Legacy - PRÉSERVER P1
mkdir -p src/jeffrey/legacy

# Copier (pas déplacer!) les modules P1 vers legacy pour transition progressive
if [ -d "src/jeffrey/consciousness" ]; then
    cp -r src/jeffrey/consciousness src/jeffrey/legacy/ 2>/dev/null || true
fi
if [ -d "src/jeffrey/memory_manager" ]; then
    cp -r src/jeffrey/memory_manager src/jeffrey/legacy/ 2>/dev/null || true
fi
if [ -d "src/jeffrey/emotional_core" ]; then
    cp -r src/jeffrey/emotional_core src/jeffrey/legacy/ 2>/dev/null || true
fi
if [ -d "src/jeffrey/dream_engine" ]; then
    cp -r src/jeffrey/dream_engine src/jeffrey/legacy/ 2>/dev/null || true
fi
if [ -d "src/jeffrey/symbiosis" ]; then
    cp -r src/jeffrey/symbiosis src/jeffrey/legacy/ 2>/dev/null || true
fi
if [ -d "src/jeffrey/brain_kernel" ]; then
    cp -r src/jeffrey/brain_kernel src/jeffrey/legacy/ 2>/dev/null || true
fi

# Créer les __init__.py avec imports compatibles
for dir in $(find src/jeffrey -type d); do
    if [ ! -f "$dir/__init__.py" ]; then
        touch "$dir/__init__.py"
    fi
done

# Adapter pattern pour migration progressive
cat > src/jeffrey/legacy/__init__.py << 'EOF'
"""
Legacy P1 modules - Migration progressive vers P2
Ces modules seront progressivement refactorisés et intégrés dans la nouvelle architecture
"""

# Réexporter pour compatibilité
try:
    from .consciousness import *
except ImportError:
    pass

try:
    from .memory_manager import *
except ImportError:
    pass

try:
    from .emotional_core import *
except ImportError:
    pass

try:
    from .dream_engine import *
except ImportError:
    pass

try:
    from .symbiosis import *
except ImportError:
    pass

try:
    from .brain_kernel import *
except ImportError:
    pass

print("✅ Legacy modules loaded for progressive migration")
EOF

# Créer README développeur
cat > src/jeffrey/core/README.md << 'EOF'
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
EOF

echo "✅ P2 structure created with legacy preservation"
