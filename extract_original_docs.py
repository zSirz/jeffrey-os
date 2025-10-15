#!/usr/bin/env python3

"""
Script pour extraire et fusionner la documentation originale depuis les backups iCloud.
"""

import ast
import re
from pathlib import Path


def extract_module_docstring(content: str) -> str | None:
    """Extrait le module docstring d'un fichier Python."""
    try:
        tree = ast.parse(content)
        docstring = ast.get_docstring(tree)
        return docstring
    except:
        # Fallback: extraction regex
        match = re.search(r'^"""(.*?)"""', content, re.MULTILINE | re.DOTALL)
        if match:
            return match.group(1).strip()
        match = re.search(r"^'''(.*?)'''", content, re.MULTILINE | re.DOTALL)
        if match:
            return match.group(1).strip()
    return None


def extract_class_docstrings(content: str) -> dict[str, str]:
    """Extrait tous les docstrings de classes."""
    class_docs = {}
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                docstring = ast.get_docstring(node)
                if docstring:
                    class_docs[node.name] = docstring
    except:
        pass
    return class_docs


def extract_function_docstrings(content: str) -> dict[str, str]:
    """Extrait tous les docstrings de fonctions."""
    func_docs = {}
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                docstring = ast.get_docstring(node)
                if docstring:
                    func_docs[node.name] = docstring
    except:
        pass
    return func_docs


def find_backup_file(filename: str, backup_dirs: list[Path]) -> Path | None:
    """Trouve le fichier de backup le plus approprié."""
    candidates = []

    for backup_dir in backup_dirs:
        for path in backup_dir.rglob(filename):
            candidates.append(path)

    # Priorité aux fichiers JEFFREY_UNIFIED
    for candidate in candidates:
        if 'JEFFREY_UNIFIED' in str(candidate):
            return candidate

    # Sinon, retourne le premier trouvé
    return candidates[0] if candidates else None


def merge_documentation(current_file: Path, backup_file: Path) -> tuple[bool, str]:
    """Fusionne la documentation du backup dans le fichier actuel."""
    try:
        with open(current_file, encoding='utf-8') as f:
            current_content = f.read()

        with open(backup_file, encoding='utf-8') as f:
            backup_content = f.read()

        # Extraction des docstrings
        backup_module_doc = extract_module_docstring(backup_content)
        current_module_doc = extract_module_docstring(current_content)

        # Si le backup a une meilleure documentation
        if backup_module_doc and (not current_module_doc or len(backup_module_doc) > len(current_module_doc)):
            # Vérification de la qualité (documentation française)
            if any(
                word in backup_module_doc.lower() for word in ['ce module', 'cette classe', 'initialise', 'retourne']
            ):
                print(f"✅ Documentation française trouvée pour {current_file.name}")

                # Remplacement du module docstring
                if current_module_doc:
                    # Remplace l'ancien docstring
                    pattern = r'""".*?"""'
                    replacement = f'"""\n{backup_module_doc}\n"""'
                    new_content = re.sub(pattern, replacement, current_content, count=1, flags=re.DOTALL)
                else:
                    # Ajoute le docstring après les imports
                    lines = current_content.split('\n')
                    insert_pos = 0
                    for i, line in enumerate(lines):
                        if not line.startswith('#') and not line.startswith('import') and not line.startswith('from'):
                            insert_pos = i
                            break
                    lines.insert(insert_pos, f'"""\n{backup_module_doc}\n"""\n')
                    new_content = '\n'.join(lines)

                return True, new_content

        return False, current_content

    except Exception as e:
        print(f"❌ Erreur lors du traitement de {current_file}: {e}")
        return False, ""


def main():
    """Script principal pour extraire et fusionner la documentation."""

    # Dossiers de backup
    backup_dirs = [
        Path("/Users/davidproz/Library/Mobile Documents/com~apple~CloudDocs/Jeffrey/Jeffrey_Apps/JEFFREY_UNIFIED"),
        Path("/Users/davidproz/Library/Mobile Documents/com~apple~CloudDocs/Jeffrey/Jeffrey_Apps/Jeffrey_OS"),
        Path("/Users/davidproz/Library/Mobile Documents/com~apple~CloudDocs/Jeffrey/Jeffrey_Apps/Jeffrey_Phoenix"),
    ]

    # Dossier actuel
    src_dir = Path("/Users/davidproz/Desktop/Jeffrey_OS/src/jeffrey")

    # Modules prioritaires
    priority_modules = [
        "core/consciousness",
        "core/emotions",
        "core/memory",
        "core/learning",
        "core/orchestration",
        "core/personality",
    ]

    updated_files = []

    for module_path in priority_modules:
        module_dir = src_dir / module_path
        if not module_dir.exists():
            continue

        print(f"\n📁 Traitement du module {module_path}")

        for py_file in module_dir.glob("*.py"):
            if py_file.name == "__init__.py":
                continue

            # Recherche du fichier de backup
            backup_file = find_backup_file(py_file.name, backup_dirs)

            if backup_file:
                print(f"  📄 {py_file.name} -> Backup trouvé: {backup_file.parent.name}/{backup_file.name}")
                updated, new_content = merge_documentation(py_file, backup_file)

                if updated and new_content:
                    # Sauvegarde du fichier mis à jour
                    with open(py_file, 'w', encoding='utf-8') as f:
                        f.write(new_content)
                    updated_files.append(str(py_file))
            else:
                print(f"  ⚠️  {py_file.name} -> Aucun backup trouvé")

    # Rapport final
    print("\n📊 Résumé:")
    print(f"  ✅ {len(updated_files)} fichiers mis à jour avec documentation originale")

    if updated_files:
        print("\n📝 Fichiers mis à jour:")
        for file in updated_files[:10]:  # Affiche les 10 premiers
            print(f"  - {file}")
        if len(updated_files) > 10:
            print(f"  ... et {len(updated_files) - 10} autres")


if __name__ == "__main__":
    main()
