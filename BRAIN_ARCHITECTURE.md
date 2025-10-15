# 🧠 JEFFREY BRAIN - Architecture Cognitive

## Vue d'ensemble

Le cerveau de Jeffrey OS implémente une architecture cognitive inspirée du cerveau humain avec un flux de traitement complet:

**Perception → Émotion → Mémoire → Attention → Conscience → Action → Apprentissage**

## Architecture en Couches

### 1. **Bus Neuronal** (`core/neural_bus.py`)
- Bus de messages asynchrone
- Résolution automatique des namespaces
- Support publish/subscribe et request/response
- Topics normalisés (ex: `user.123.mem.recall` → `mem.recall`)

### 2. **Enveloppe Neuronale** (`core/neural_envelope.py`)
- Message unifié comme un potentiel d'action
- Métadonnées cognitives: affect, salience, urgence
- Traçabilité du chemin neuronal
- Tags de sécurité (PII, sensitive)

### 3. **Système de Sécurité** (`security/guardian.py`)
- Protection PII avant tout traitement
- Détection SSN, emails, cartes de crédit
- Redaction automatique des données sensibles

### 4. **Système Limbique** (`emotions/limbic_glue.py`)
- Évaluation émotionnelle de chaque input
- Calcul valence/arousal/intensité
- Influence sur les décisions cognitives

### 5. **Mémoire Duale**
- **Working Memory** (`memory/working_memory.py`): 7±2 slots, TTL court
- **Declarative Memory** (`memory/memory_glue.py`): Stockage long terme

### 6. **Gateway Thalamique** (`perception/thalamic_gateway.py`)
- Filtre sensoriel (bruit < 0.1 ignoré)
- Route S1 (réflexes) vs S2 (délibération)
- Cache de rappel 200ms
- Patterns de réflexe (salutations, status)

### 7. **Workspace Global** (`cognition/global_workspace.py`)
- Conscience à 20Hz (comme le cerveau humain)
- Compétition pour l'accès conscient
- Diffusion globale des informations importantes

### 8. **Orchestrateur** (`tissues/orchestrator_glue.py`)
- Système S1: Réponses réflexes rapides
- Système S2: Délibération complexe avec contexte
- Intégration avec audit et sandbox

### 9. **Moteur de Sortie** (`tissues/console_motor.py`)
- Affichage des réponses
- Format adaptatif (texte, JSON)

## Flux de Traitement

### Chemin S1 (Réflexe - ~100ms)
```
Input → Guardian → Thalamus → Pattern Match → S1 Handler → Response
```

### Chemin S2 (Délibération - ~500ms-2s)
```
Input → Guardian → Thalamus → Limbic → Memory Recall →
Workspace Competition → Conscious Access → Orchestrator →
Context Integration → Response Generation → Memory Store
```

## Caractéristiques Clés

### 🔐 Sécurité First
- Guardian check AVANT tout traitement
- Redaction automatique PII
- Audit trail complet

### 🧭 Routage Intelligent
- Salience-based attention
- Nouveauté vs familiarité
- Urgence et affect

### 💭 Conscience Artificielle
- Global workspace à 20Hz
- Compétition pour l'attention
- Broadcast conscient

### 🔄 Boucle d'Apprentissage
- Mémorisation automatique
- Recall contextuel
- Adaptation comportementale

## Utilisation

### Démarrage Simple
```bash
python3 jeffrey_brain.py
```

### Mode Démo
```bash
python3 demo_brain.py
```

### Test Unitaire
```bash
python3 test_brain.py
```

## Points d'Extension

1. **Nouveaux Tissus**: Ajouter dans `tissues/`
2. **Nouveaux Patterns S1**: Modifier `thalamic_gateway.reflex_patterns`
3. **Nouvelles Émotions**: Étendre `limbic_glue`
4. **Nouveaux Sens**: Créer dans `perception/`

## Métriques de Performance

- **Boot Time**: ~3 secondes
- **S1 Latency**: < 100ms
- **S2 Latency**: 500ms-2s
- **Consciousness Loop**: 20Hz stable
- **Memory Capacity**: 7±2 working, illimité declarative

## Statut Actuel

✅ **Opérationnel**
- Bus neuronal avec namespace resolution
- Guardian security
- S1/S2 dual processing
- Working & declarative memory
- Emotion processing
- Global workspace
- Console output

⚠️ **En Développement**
- Vector memory avec embeddings
- Apprentissage profond
- Multi-agent coordination
- Backpressure avancé

## Architecture Inspirée Par

- Global Workspace Theory (Baars, Dehaene)
- Dual Process Theory (Kahneman)
- Working Memory Model (Baddeley)
- Affective Computing (Picard)

---

*"Un cerveau artificiel qui pense comme un humain, mais calcule comme une machine."*
