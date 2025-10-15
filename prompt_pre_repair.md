# Prompt pour Claude Code - Préparation Avant Réparation Chirurgicale

## Contexte
Le projet Jeffrey OS a des erreurs critiques qui bloquent l'exécution du script de réparation. Tu dois corriger UNIQUEMENT les 3 erreurs bloquantes, puis nettoyer l'état git pour permettre le lancement du script de réparation.

**Répertoire de travail** : `~/Desktop/Jeffrey_OS`

## Objectifs
1. Corriger 3 erreurs critiques qui empêchent l'import/runtime
2. Nettoyer l'état git proprement (SANS supprimer de fichiers)
3. Préparer un état propre pour `repair_surgical_complete.sh`

---

## Étape 1 : Corrections Critiques

### Erreur 1 : `agi_orchestrator.py` ligne 686 - MemoryManager non défini

**Fichier** : `src/jeffrey/core/orchestration/agi_orchestrator.py`

**Problème** : Utilisation de `MemoryManager` qui n'existe pas

**Action** :
1. Ouvre le fichier
2. Trouve la ligne 686 (environ) qui contient `MemoryManager`
3. Remplace par un alias protégé :
   ```python
   # Avant :
   something = MemoryManager

   # Après :
   try:
       from .unified_memory import UnifiedMemory as MemoryManager
   except Exception:
       MemoryManager = None  # fallback temporaire
   ```

### Erreur 2 : `jeffrey_emotional_core.py` - Variables non définies

**Fichiers** (2 copies identiques - **skip silencieux si n'existe pas**) :
- `src/vendors/icloud/core/jeffrey_emotional_core.py`
- `src/vendors/icloud/jeffrey_emotional_core.py`

**Problème** : Variables `notification` et `time` utilisées mais non importées

**Actions** (seulement si le fichier existe) :

Pour `time` :
1. Ajoute `import time` en haut du fichier (avec les autres imports)

Pour `notification` :
1. Cherche toutes les utilisations de `notification.` dans le fichier
2. Commente ces lignes avec un commentaire explicite
3. Exemple :
   ```python
   # Avant :
   notification.send(message)

   # Après :
   # notification.send(message)  # notification non disponible - désactivé temporairement
   ```

**Note** : Applique les mêmes corrections aux DEUX fichiers s'ils existent tous les deux

### Erreur 3 : `unified_memory.py` - Fonction save_fact définie 2 fois

**Fichier** : `src/vendors/icloud/memory/unified_memory.py`

**Problème** : Fonction `save_fact` définie ligne 448 ET ligne 487

**Action** :
1. Ouvre le fichier
2. Trouve les deux définitions de `def save_fact(`
3. Commente ENTIÈREMENT la première définition (ligne ~448-486)
4. Garde la deuxième définition (ligne ~487+)
5. Ajoute un commentaire explicatif :
   ```python
   # DUPLICATE REMOVED - save_fact était défini 2 fois
   # Première définition commentée, gardé la plus récente (ligne 487+)
   # def save_fact(...):
   #     ...
   ```

---

## Étape 2 : Vérification des Corrections

Après chaque correction :

```bash
# Vérifier que le fichier compile
python3 -m py_compile <fichier_modifié>
```

**Si ça échoue** : revert la dernière modification sur ce fichier et propose une alternative.

Si ça compile, passe au suivant.

---

## Étape 3 : Nettoyage Git (SANS supprimer)

### 3.1 Créer/basculer vers branche dédiée

```bash
cd ~/Desktop/Jeffrey_OS

# Créer ou basculer vers la branche de réparation
git checkout -b repair/surgical-from-phase2 2>/dev/null || git checkout repair/surgical-from-phase2
```

### 3.2 Ajouter tous les changements

```bash
# Ajouter TOUS les fichiers (y compris ceux reformatés par black/isort)
git add -A

# Vérifier ce qui sera commité
git status
```

### 3.3 Commit avec bypass des hooks

Les hooks pre-commit (black, isort, flake8, mypy) vont encore trouver des erreurs non-critiques. On les bypass car elles seront corrigées après la réparation.

```bash
git commit --no-verify -m "fix: corrections critiques pré-réparation chirurgicale

Corrections appliquées :
- agi_orchestrator.py : MemoryManager non défini → alias protégé UnifiedMemory
- jeffrey_emotional_core.py (si présent) : ajout import time, désactivé notification
- unified_memory.py : supprimé duplicate save_fact (gardé ligne 487)
- Reformatages black/isort appliqués automatiquement

Note: Commit avec --no-verify car erreurs non-critiques restantes
Seront corrigées après repair_surgical_complete.sh

Refs: #repair #pre-surgical"
```

### 3.4 Vérification finale

```bash
# L'état doit être propre
git status

# Doit afficher soit "nothing to commit, working tree clean"
# Soit seulement des "Untracked files" (OK, on ne les touche pas)
```

---

## Étape 4 : Rapport Final

Génère un rapport texte avec :

1. **Fichiers modifiés** : Liste des fichiers corrigés (avec mention si skip)
2. **État git** : Résultat de `git status`
3. **Branche courante** : Confirme qu'on est sur `repair/surgical-from-phase2`
4. **Prochaine commande** : Affiche clairement :
   ```bash
   cd ~/Desktop/Jeffrey_OS
   ./repair_surgical_complete.sh | tee repair_run_$(date +%Y%m%d_%H%M%S).log
   ```

---

## Contraintes Importantes

1. **NE SUPPRIME AUCUN FICHIER** - Même les backups .BEFORE_*, .STUB_BACKUP, etc.
2. **NE TOUCHE PAS** aux erreurs non-critiques (imports inutilisés, lignes longues, etc.)
3. **NE MODIFIE PAS** d'autres fichiers que les 3-4 mentionnés
4. **VÉRIFIE** que chaque fichier compile après modification
5. **SKIP SILENCIEUX** si un fichier vendor n'existe pas
6. **REVERT** si compilation échoue et propose alternative

---

## En Cas de Problème

Si un fichier ne compile pas après ta modification :
1. `git checkout -- <fichier>` pour revert
2. Montre l'erreur exacte
3. Propose une correction alternative plus conservatrice
4. Demande validation avant de continuer

Si git status n'est pas propre après le commit :
1. Affiche `git diff` pour les fichiers encore modifiés
2. Identifie pourquoi (nouveaux changements de hooks ?)
3. Propose `git add` + nouveau commit si nécessaire

---

## Résultat Attendu

À la fin de ton exécution :

✅ 3 erreurs critiques corrigées (ou skippées si fichiers absents)
✅ Tous les fichiers modifiés compilent
✅ Git status propre (ou seulement untracked files)
✅ Branche : `repair/surgical-from-phase2`
✅ Prêt pour lancer `repair_surgical_complete.sh`

---

## Commence Maintenant

Exécute les étapes 1 à 4 dans l'ordre. Montre-moi ta progression après chaque étape majeure.
