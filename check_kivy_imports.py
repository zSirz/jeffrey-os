#!/usr/bin/env python3
"""
Vérifie qu'aucun module Core n'importe Kivy
"""

import re
from pathlib import Path


def check_kivy_imports():
    """Scan pour trouver les imports Kivy mal placés"""

    print("🔍 Recherche des imports Kivy dans le Core...")
    print("=" * 60)

    # Patterns à chercher
    patterns = [r'from\s+kivy', r'import\s+kivy']

    # Dossiers à scanner
    core_dirs = ['src/jeffrey/core', 'src/jeffrey/bridge']

    # Dossiers autorisés pour Kivy
    allowed_dirs = ['avatars', 'ui', 'frontend', 'display']

    violations = []

    for core_dir in core_dirs:
        core_path = Path(core_dir)
        if not core_path.exists():
            continue

        for py_file in core_path.rglob('*.py'):
            # Skip si dans un dossier autorisé
            if any(allowed in str(py_file) for allowed in allowed_dirs):
                continue

            with open(py_file, encoding='utf-8') as f:
                try:
                    content = f.read()
                    for i, line in enumerate(content.splitlines(), 1):
                        for pattern in patterns:
                            if re.search(pattern, line, re.IGNORECASE):
                                # Ignorer les commentaires
                                if line.strip().startswith('#'):
                                    continue
                                violations.append({'file': str(py_file), 'line': i, 'content': line.strip()})
                except Exception as e:
                    print(f"⚠️ Erreur lecture {py_file}: {e}")

    # Rapport
    if violations:
        print(f"\n❌ TROUVÉ {len(violations)} IMPORTS KIVY DANS LE CORE:")
        print("-" * 60)

        for v in violations:
            print(f"\n📁 {v['file']}")
            print(f"   Ligne {v['line']}: {v['content']}")

        print("\n" + "=" * 60)
        print("⚠️ Ces imports doivent être déplacés vers les Avatars/UI!")

    else:
        print("\n✅ AUCUN IMPORT KIVY TROUVÉ DANS LE CORE")
        print("Le Core est propre et prêt pour le mode headless!")

    return len(violations) == 0


if __name__ == "__main__":
    import sys

    success = check_kivy_imports()
    sys.exit(0 if success else 1)
