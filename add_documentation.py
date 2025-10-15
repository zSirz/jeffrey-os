#!/usr/bin/env python3
"""
Script pour ajouter automatiquement la documentation française et les type hints
à tous les fichiers Python du projet Jeffrey OS.
"""

from pathlib import Path


def needs_future_annotations(content: str) -> bool:
    """Vérifie si le fichier nécessite l'import future annotations."""
    return "from __future__ import annotations" not in content


def add_future_annotations(content: str) -> str:
    """Ajoute l'import future annotations après le module docstring."""
    if needs_future_annotations(content):
        # Trouver la fin du module docstring
        lines = content.split('\n')
        insert_index = 0

        # Chercher après le docstring module
        in_docstring = False
        docstring_count = 0

        for i, line in enumerate(lines):
            if '"""' in line or "'''" in line:
                docstring_count += line.count('"""') + line.count("'''")
                if docstring_count >= 2:
                    insert_index = i + 1
                    break

        # Si pas de docstring, insérer après les imports existants ou au début
        if insert_index == 0:
            for i, line in enumerate(lines):
                if line.startswith('import ') or line.startswith('from '):
                    insert_index = i
                    break

        # Insérer l'import
        if insert_index > 0:
            lines.insert(insert_index, "\nfrom __future__ import annotations")
            return '\n'.join(lines)

    return content


def process_file(file_path: Path) -> bool:
    """
    Traite un fichier Python pour ajouter future annotations.

    Args:
        file_path: Chemin du fichier à traiter

    Returns:
        True si le fichier a été modifié, False sinon
    """
    try:
        # Ignorer __init__.py et fichiers courts
        if file_path.name == "__init__.py":
            return False

        with open(file_path, encoding='utf-8') as f:
            content = f.read()

        # Ignorer les fichiers trop courts
        if len(content.splitlines()) < 10:
            return False

        # Ajouter future annotations si nécessaire
        if needs_future_annotations(content):
            new_content = add_future_annotations(content)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            print(f"✅ Modifié: {file_path}")
            return True

        return False

    except Exception as e:
        print(f"❌ Erreur pour {file_path}: {e}")
        return False


def main():
    """Fonction principale pour traiter tous les fichiers."""
    src_path = Path("/Users/davidproz/Desktop/Jeffrey_OS/src/jeffrey")

    if not src_path.exists():
        print(f"❌ Le chemin {src_path} n'existe pas")
        return

    # Collecter tous les fichiers Python
    python_files = list(src_path.rglob("*.py"))

    print(f"📊 Trouvé {len(python_files)} fichiers Python")

    modified_count = 0
    for file_path in python_files:
        if process_file(file_path):
            modified_count += 1

    print(f"\n✨ Terminé! {modified_count} fichiers modifiés")


if __name__ == "__main__":
    main()
