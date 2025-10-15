#!/usr/bin/env python3
"""
Script de correction complète des imports obsolètes (from core.* → from jeffrey.core.*)
Intégration des 3 patterns de GPT/Marc + création __init__.py + robustesse
"""

import os
import re
import subprocess
import sys
from pathlib import Path

# === GARDE-FOUS (GPT/Marc) ===
# Forcer UTF-8
os.environ['PYTHONUTF8'] = '1'

# Vérifier qu'on est à la racine du repo
if not (Path('.git').exists() and Path('src/jeffrey').exists()):
    print("❌ Lance ce script depuis la racine du repo Jeffrey OS")
    sys.exit(1)

# === CONFIGURATION ===
ROOTS = [Path('src'), Path('core'), Path('services')]
IGNORE_PATTERNS = ['test_', '_test.py', 'tests/', 'backups/', '__pycache__', 'venv/', '.venv/']

# === PATTERNS REGEX (3 patterns de GPT/Marc) ===
# Pattern 1 : from core.xxx import Y
PAT_FROM = re.compile(r'(^\s*from\s+)core(\.[\w\.]+)', re.MULTILINE)

# Pattern 2 : import core.xxx as y
PAT_IMPORT = re.compile(r'(^\s*import\s+)core(\.[\w\.]+)(\s+as\s+\w+)?', re.MULTILINE)

# Pattern 3 : from core import Y
PAT_FROM_DIRECT = re.compile(r'(^\s*from\s+)core(\s+import\s+[\w\*,\s]+)', re.MULTILINE)


def should_ignore(filepath: Path) -> bool:
    """Vérifie si un fichier doit être ignoré."""
    filepath_str = str(filepath)
    return any(pattern in filepath_str for pattern in IGNORE_PATTERNS)


def fix_imports_in_file(filepath: Path) -> bool:
    """Applique les 3 corrections d'imports dans un fichier."""
    try:
        content = filepath.read_text(encoding='utf-8', errors='ignore')
        original_content = content

        # Appliquer les 3 patterns dans l'ordre
        content = PAT_FROM.sub(r'\1jeffrey.core\2', content)
        content = PAT_IMPORT.sub(r'\1jeffrey.core\2\3', content)
        content = PAT_FROM_DIRECT.sub(r'\1jeffrey.core\2', content)

        if content != original_content:
            filepath.write_text(content, encoding='utf-8')

            # Ruff fix ciblé intégré (GPT/Marc + Grok)
            try:
                subprocess.run(
                    ['ruff', 'check', '--fix', '--select', 'F401,F822', str(filepath)], capture_output=True, timeout=5
                )
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass  # Ruff optionnel

            return True
        return False
    except Exception as e:
        print(f"❌ Erreur dans {filepath}: {e}", file=sys.stderr)
        return False


def ensure_init_files():
    """Crée les __init__.py manquants (GPT/Marc)."""
    critical_paths = [
        Path('src/jeffrey/__init__.py'),
        Path('src/jeffrey/core/__init__.py'),
    ]

    created = []
    for init_path in critical_paths:
        if not init_path.exists() and init_path.parent.exists():
            init_path.write_text('"""Package auto-généré."""\n')
            created.append(init_path)

    # Scan tous les dossiers de src/jeffrey/core pour __init__.py manquants
    core_path = Path('src/jeffrey/core')
    if core_path.exists():
        for dirpath in core_path.rglob('*'):
            if dirpath.is_dir():
                init_file = dirpath / '__init__.py'
                if not init_file.exists():
                    init_file.write_text('"""Package auto-généré."""\n')
                    created.append(init_file)

    return created


def main():
    """Exécution principale."""
    print("🔧 CORRECTION COMPLÈTE DES IMPORTS")
    print("=" * 60)

    # Vérifier l'environnement Python
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"📂 Répertoire: {Path.cwd()}")
    print("")

    # 1. Correction des imports
    fixed_count = 0
    errors = []

    for root in ROOTS:
        if not root.exists():
            print(f"⏭️  Dossier inexistant : {root}")
            continue

        print(f"📂 Scan de {root}...")
        for filepath in root.rglob('*.py'):
            if should_ignore(filepath):
                continue

            if fix_imports_in_file(filepath):
                print(f"  ✅ {filepath}")
                fixed_count += 1

    # 2. Création des __init__.py manquants
    print("\n📦 Vérification des __init__.py...")
    created_inits = ensure_init_files()
    for init_path in created_inits:
        print(f"  ✅ Créé : {init_path}")

    # 3. Résumé
    print("\n📊 RÉSUMÉ")
    print("=" * 60)
    print(f"✅ {fixed_count} fichiers corrigés")
    print(f"✅ {len(created_inits)} __init__.py créés")

    if errors:
        print(f"⚠️  {len(errors)} erreurs rencontrées")
        for filepath, error in errors[:5]:
            print(f"   • {filepath}: {error}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
