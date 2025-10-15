# 🧠 Jeffrey OS Consciousness Loop - Production Final

## Architecture
```
Input → Perception → Compréhension (Wernicke) → [Mémoire || Émotion]
     → Conscience → Expression (Broca) → Output → Consolidation
         ↑______________________________________________|
```

## Corrections Critiques Appliquées
- ✅ Tous les dossiers créés avant utilisation
- ✅ Schéma DB vérifié/migré (last_accessed, session_id)
- ✅ Makefile idempotent (pas de doublons)
- ✅ Context retourné dans search()
- ✅ Circuit breaker pour modules défaillants
- ✅ Tests robustes sans faux positifs
- ✅ MemoryStub corrigé (évite .store.store ambiguïté)
- ✅ __init__.py garantis pour tous les packages

## Performance
- Cible: <250ms
- Optimal: <100ms avec cache
- P95: <150ms

## Lancement
```bash
# Standard
make -f Makefile_hardened launch-integrated

# Debug
JEFFREY_DEBUG=1 make -f Makefile_hardened launch-integrated

# Tests
make -f Makefile_hardened test-integration

# Benchmark
make -f Makefile_hardened benchmark
```

## Monitoring
- `status` - Vue d'ensemble
- `health` - JSON complet
- `regions` - Détail des régions
- `stats` - Statistiques performance

## Troubleshooting
- **Modules fail** : Vérifier les paths dans inventory_ultimate.json
- **DB errors** : Vérifier permissions sur data/
- **Slow** : Activer cache, réduire modules actifs
- **Import errors** : Vérifier tous les __init__.py existent
