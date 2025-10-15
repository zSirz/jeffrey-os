# Registre des Shims de Compatibilité - Jeffrey OS

Ce fichier liste tous les shims (alias) de compatibilité créés pour faciliter
la migration architecturale.

**⚠️ Ces modules doivent être progressivement supprimés ⚠️**

## Format

| Module Legacy | Module Cible | Raison | Date Création | Deadline Suppression | Équipe |
|--------------|--------------|--------|---------------|---------------------|--------|
| `jeffrey.core.unified_memory` | `jeffrey.core.memory.unified_memory` | Consolidation architecturale, 3 versions → 1 (Production Ready 33ko), imports cassés | 2025-10-07 | Sprint M+2 | Claude+GPT+Grok+Gemini |
