# 🔧 Guide de Restauration V2 - Jeffrey OS

**Version 2.0 - Ultra-Safe avec Dry-Run**

## 🎯 Nouveautés V2

✅ **Dry-run par défaut** - Aucune écriture sans `--apply`
✅ **Protection Git** - Refuse si repo non propre
✅ **Scan optimisé** - Phase unique pour imports + stubs
✅ **Contrats d'interface** - Extraction automatique des signatures
✅ **Obsolescence scoring** - Identifie les stubs à supprimer
✅ **Estimation d'effort** - Planning par sprints
✅ **Allow-list** - Tolère experimental/ et _archive/
✅ **Pre-commit hooks** - Anti-régression automatique

## 🚀 Démarrage Rapide (Safe)

```bash
# 1. Rendre exécutable
chmod +x run_restoration_safe.sh validate_strict.sh

# 2. Lancer en mode sûr (dry-run)
./run_restoration_safe.sh
# Choisir option 3 (Diagnostic + Contrats + Shims dry-run)

# 3. Consulter les rapports
cat PRIORITIZATION_REPORT_V2.md

# 4. Si tout OK, relancer avec --apply
./run_restoration_safe.sh
# Choisir option 4 (TOUT avec --apply)
```

## 📋 Scripts Disponibles

### 1. `comprehensive_diagnostic_v2.py`
**Scan unique optimisé** : imports + stubs + obsolescence + graphe deps

**Génère** : `COMPREHENSIVE_DIAGNOSTIC_V2.json`

### 2. `extract_interface_contracts.py`
**Extraction automatique** des signatures et usages

**Génère** : `interface_contracts/*.md` (un par module)

### 3. `create_shims_safe.py`
**Création sécurisée** avec dry-run et protection Git

```bash
# Dry-run (défaut)
python3 create_shims_safe.py

# Application réelle
python3 create_shims_safe.py --apply --shims-dir
```

**Génère** : `SHIMS_PLAN.json` (dry-run) ou `SHIMS_MAPPING.json` (apply)

### 4. `generate_priority_report_v2.py`
**Rapport intelligent** avec centralité, effort, sprints

**Génère** : `PRIORITIZATION_REPORT_V2.md`

### 5. `validate_strict.sh`
**Validation stricte** avec allow-list

## 🎯 Workflow de Travail

### Phase 1 : Diagnostic (Safe, 30min)

```bash
python3 comprehensive_diagnostic_v2.py
python3 extract_interface_contracts.py
```

**Résultat** :
- Liste des imports cassés
- Stubs détectés avec scores d'obsolescence
- Contrats d'interface par module

### Phase 2 : Revue & Planning (15min)

```bash
cat PRIORITIZATION_REPORT_V2.md
```

**Actions** :
- Identifier le TOP 20
- Vérifier les estimations d'effort
- Décider du premier sprint

### Phase 3 : Création Shims (15min)

```bash
# Test dry-run d'abord
python3 create_shims_safe.py

# Si OK, appliquer
python3 create_shims_safe.py --apply --shims-dir
```

### Phase 4 : Boucle de Restauration Unitaire

Pour chaque module du TOP 20, **5 étapes** :

#### 1. Choisir (1 min)
Module suivant de la liste prioritaire

#### 2. Chercher (15 min chrono)
```bash
find ~/iCloud -name "*emotional_core*"
grep -r "emotional_core" ~/backups/
```

#### 3. Décider (5 min)
- Trouvé → Copier + valider
- Introuvable → Recréer
- Obsolète → Supprimer imports

#### 4. Implémenter (1-4h)
```bash
# Consulter le contrat
cat interface_contracts/jeffrey_core_emotions_emotional_core.md

# Analyser l'usage
grep -r "emotional_core" src/ services/

# Implémenter avec le squelette fourni
# + Logique minimale viable
# + TODO pour parties complexes
```

#### 5. Valider (5 min)
```bash
# Test import
python3 -c "import jeffrey.core.emotions.emotional_core"

# Validation complète
bash validate_strict.sh

# Commit
git add src/jeffrey/core/emotions/emotional_core.py
git commit -m "feat(core): Recréation emotional_core (IMV)"
```

## 🚫 Règles Absolues

1. **Toujours en dry-run d'abord** - Jamais de `--apply` sans vérification
2. **Repo propre obligatoire** - Git clean avant modifications
3. **Contrats avant implémentation** - Consulter `interface_contracts/`
4. **Pas de stub** - Implémentation minimale mais fonctionnelle
5. **Validation continue** - Après chaque module

## 📊 Métriques de Succès

- ✅ 0 stub détecté
- ✅ 0 import cassé (runtime)
- ✅ 0 cycle dans shims
- ✅ Validation stricte passe
- ✅ Pre-commit hooks actifs

## 🔧 Installation Pre-Commit (Optionnel mais Recommandé)

```bash
pip install pre-commit
pre-commit install

# Test
git add -A
git commit -m "test: pre-commit hooks"
```

## 💡 Troubleshooting

### "Git not clean"
```bash
git status
git stash  # ou commit
```

### "Module not found après shim"
```bash
# Vérifier le chemin dans le shim
cat src/jeffrey/_shims/.../module.py

# Tester l'import direct
python3 -c "import sys; sys.path.insert(0, 'src'); import jeffrey..."
```

### "Stub score élevé mais encore utilisé"
Vérifiez l'âge Git :
```bash
git log -1 --format=%at path/to/file.py
```

## 🤝 Support

En cas de blocage :
1. Consulter `PRIORITIZATION_REPORT_V2.md`
2. Vérifier les contrats dans `interface_contracts/`
3. Relancer le diagnostic : `python3 comprehensive_diagnostic_v2.py`
4. Valider l'état : `bash validate_strict.sh`

---

**Bon courage pour la restauration ! 🚀**
