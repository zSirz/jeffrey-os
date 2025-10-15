#!/usr/bin/env python3

"""
Script complet pour transférer la documentation originale depuis les backups iCloud.
"""

import os
import re
from pathlib import Path

# Mapping des fichiers avec leurs emplacements dans les backups
FILE_MAPPINGS = {
    # Consciousness module
    "jeffrey_consciousness_v3.py": [
        "/Users/davidproz/Library/Mobile Documents/com~apple~CloudDocs/Jeffrey/Jeffrey_Apps/Jeffrey_OS/src/core/memory/jeffrey_consciousness_v3.py",
        "/Users/davidproz/Library/Mobile Documents/com~apple~CloudDocs/Jeffrey/Jeffrey_Apps/JEFFREY_UNIFIED/engine/consciousness/jeffrey_consciousness_v3.py",
    ],
    "jeffrey_living_consciousness.py": [
        "/Users/davidproz/Library/Mobile Documents/com~apple~CloudDocs/Jeffrey/Jeffrey_Apps/JEFFREY_UNIFIED/engine/consciousness/jeffrey_living_consciousness.py"
    ],
    "jeffrey_dream_system.py": [
        "/Users/davidproz/Library/Mobile Documents/com~apple~CloudDocs/Jeffrey/Jeffrey_Apps/JEFFREY_UNIFIED/engine/dreams/jeffrey_dream_system.py"
    ],
    "jeffrey_living_memory.py": [
        "/Users/davidproz/Library/Mobile Documents/com~apple~CloudDocs/Jeffrey/Jeffrey_Apps/JEFFREY_UNIFIED/engine/memory/jeffrey_living_memory.py"
    ],
    "jeffrey_secret_diary.py": [
        "/Users/davidproz/Library/Mobile Documents/com~apple~CloudDocs/Jeffrey/Jeffrey_Apps/JEFFREY_UNIFIED/interfaces/shared/jeffrey_secret_diary.py"
    ],
    "jeffrey_living_expressions.py": [
        "/Users/davidproz/Library/Mobile Documents/com~apple~CloudDocs/Jeffrey/Jeffrey_Apps/JEFFREY_UNIFIED/interfaces/shared/jeffrey_living_expressions.py"
    ],
    # Memory module
    "cortex_memoriel.py": [
        "/Users/davidproz/Library/Mobile Documents/com~apple~CloudDocs/Jeffrey/Jeffrey_Apps/Jeffrey_OS/src/core/memory/cortex_memoriel.py",
        "/Users/davidproz/Library/Mobile Documents/com~apple~CloudDocs/Jeffrey/Jeffrey_Apps/JEFFREY_UNIFIED/engine/memory/cortex_memoriel.py",
    ],
    "jeffrey_human_memory.py": [
        "/Users/davidproz/Library/Mobile Documents/com~apple~CloudDocs/Jeffrey/Jeffrey_Apps/JEFFREY_UNIFIED/engine/memory/jeffrey_human_memory.py"
    ],
    "memory_manager.py": [
        "/Users/davidproz/Library/Mobile Documents/com~apple~CloudDocs/Jeffrey/Jeffrey_Apps/JEFFREY_UNIFIED/engine/memory/memory_manager.py"
    ],
    # Emotions module
    "emotion_engine.py": [
        "/Users/davidproz/Library/Mobile Documents/com~apple~CloudDocs/Jeffrey/Jeffrey_Apps/JEFFREY_UNIFIED/engine/emotions/emotion_engine.py"
    ],
    "jeffrey_intimate_mode.py": [
        "/Users/davidproz/Library/Mobile Documents/com~apple~CloudDocs/Jeffrey/Jeffrey_Apps/JEFFREY_UNIFIED/engine/emotions/jeffrey_intimate_mode.py"
    ],
    "jeffrey_curiosity_engine.py": [
        "/Users/davidproz/Library/Mobile Documents/com~apple~CloudDocs/Jeffrey/Jeffrey_Apps/JEFFREY_UNIFIED/engine/perception/jeffrey_curiosity_engine.py"
    ],
    # Learning module
    "theory_of_mind.py": [
        "/Users/davidproz/Library/Mobile Documents/com~apple~CloudDocs/Jeffrey/Jeffrey_Apps/Jeffrey_OS/src/core/learning/theory_of_mind.py",
        "/Users/davidproz/Library/Mobile Documents/com~apple~CloudDocs/Jeffrey/Jeffrey_Apps/JEFFREY_UNIFIED/engine/learning/theory_of_mind.py",
    ],
    "jeffrey_deep_learning.py": [
        "/Users/davidproz/Library/Mobile Documents/com~apple~CloudDocs/Jeffrey/Jeffrey_Apps/JEFFREY_UNIFIED/engine/learning/jeffrey_deep_learning.py"
    ],
    # Orchestration module
    "ia_orchestrator_ultimate.py": [
        "/Users/davidproz/Library/Mobile Documents/com~apple~CloudDocs/Jeffrey/Jeffrey_Apps/Jeffrey_OS/src/core/ia_orchestrator_ultimate.py",
        "/Users/davidproz/Library/Mobile Documents/com~apple~CloudDocs/Jeffrey/Jeffrey_Apps/JEFFREY_UNIFIED/engine/orchestrator/ia_orchestrator_ultimate.py",
    ],
}


def extract_docstring(content: str) -> str | None:
    """Extrait le module docstring d'un fichier Python."""
    # Recherche du docstring triple-quotes
    patterns = [
        (r'^"""(.*?)"""', re.MULTILINE | re.DOTALL),
        (r"^'''(.*?)'''", re.MULTILINE | re.DOTALL),
    ]

    for pattern, flags in patterns:
        match = re.search(pattern, content, flags)
        if match:
            return match.group(0)  # Retourne avec les triple-quotes

    return None


def has_quality_french_doc(docstring: str) -> bool:
    """Vérifie si le docstring contient une documentation française de qualité."""
    if not docstring:
        return False

    # Mots-clés indiquant une documentation française de qualité
    french_keywords = [
        'ce module',
        'cette classe',
        'système',
        'mémoire',
        'émotions',
        'conscience',
        'initialise',
        'retourne',
        'gère',
        'implémente',
        'architecture',
        'composants',
        'fonctionnalités',
        'évolutive',
        'cognitive',
        'artificielle',
    ]

    doc_lower = docstring.lower()
    keyword_count = sum(1 for kw in french_keywords if kw in doc_lower)

    # Critères de qualité
    has_keywords = keyword_count >= 3
    has_length = len(docstring) > 200
    has_structure = any(marker in docstring for marker in [':', '-', '•', '*'])

    return has_keywords and has_length and has_structure


def transfer_documentation(target_file: Path, backup_paths: list[str]) -> bool:
    """Transfère la documentation depuis le backup vers le fichier cible."""

    # Lire le fichier actuel
    if not target_file.exists():
        print(f"  ⚠️  Fichier cible n'existe pas: {target_file}")
        return False

    with open(target_file, encoding='utf-8') as f:
        current_content = f.read()

    current_doc = extract_docstring(current_content)
    current_has_quality = has_quality_french_doc(current_doc)

    # Chercher la meilleure documentation dans les backups
    best_doc = None
    best_source = None

    for backup_path in backup_paths:
        if not os.path.exists(backup_path):
            continue

        try:
            with open(backup_path, encoding='utf-8') as f:
                backup_content = f.read()

            backup_doc = extract_docstring(backup_content)

            if has_quality_french_doc(backup_doc):
                # Préférer la documentation la plus longue et détaillée
                if not best_doc or len(backup_doc) > len(best_doc):
                    best_doc = backup_doc
                    best_source = backup_path
        except Exception as e:
            print(f"    Erreur lecture {backup_path}: {e}")
            continue

    # Transférer si on a trouvé une meilleure documentation
    if best_doc and (not current_has_quality or len(best_doc) > len(current_doc or "")):
        print(f"  ✅ Documentation trouvée depuis: {Path(best_source).parent.name}/{Path(best_source).name}")

        # Remplacer le docstring
        if current_doc:
            new_content = current_content.replace(current_doc, best_doc, 1)
        else:
            # Insérer après le shebang et encoding
            lines = current_content.split('\n')
            insert_pos = 0

            for i, line in enumerate(lines):
                if line.startswith('#'):
                    insert_pos = i + 1
                else:
                    break

            lines.insert(insert_pos, '\n' + best_doc + '\n')
            new_content = '\n'.join(lines)

        # Sauvegarder
        with open(target_file, 'w', encoding='utf-8') as f:
            f.write(new_content)

        return True

    elif current_has_quality:
        print("  ℹ️  Documentation française déjà présente")
    else:
        print("  ⚠️  Aucune documentation de qualité trouvée")

    return False


def main():
    """Script principal pour transférer la documentation."""

    base_dir = Path("/Users/davidproz/Desktop/Jeffrey_OS/src/jeffrey")

    # Statistiques
    total_files = 0
    updated_files = 0
    already_good = 0
    not_found = 0

    print("🚀 Transfert de la documentation originale depuis les backups iCloud\n")
    print("=" * 70)

    # Modules à traiter
    modules = {
        "consciousness": base_dir / "core/consciousness",
        "emotions": base_dir / "core/emotions/core",
        "memory": base_dir / "core/memory",
        "learning": base_dir / "core/learning",
        "orchestration": base_dir / "core/orchestration",
        "personality": base_dir / "core/personality",
    }

    for module_name, module_path in modules.items():
        print(f"\n📁 Module: {module_name}")
        print("-" * 50)

        if not module_path.exists():
            print(f"  Module path n'existe pas: {module_path}")
            continue

        # Traiter les fichiers Python du module
        for py_file in module_path.glob("*.py"):
            if py_file.name == "__init__.py":
                continue

            total_files += 1
            print(f"\n📄 {py_file.name}")

            # Chercher dans les mappings
            if py_file.name in FILE_MAPPINGS:
                backup_paths = FILE_MAPPINGS[py_file.name]
                if transfer_documentation(py_file, backup_paths):
                    updated_files += 1
                else:
                    already_good += 1
            else:
                # Recherche générique dans les backups
                backup_paths = [
                    f"/Users/davidproz/Library/Mobile Documents/com~apple~CloudDocs/Jeffrey/Jeffrey_Apps/JEFFREY_UNIFIED/**/{py_file.name}",
                    f"/Users/davidproz/Library/Mobile Documents/com~apple~CloudDocs/Jeffrey/Jeffrey_Apps/Jeffrey_OS/**/{py_file.name}",
                ]

                found_paths = []
                for pattern in backup_paths:
                    from glob import glob

                    found_paths.extend(glob(pattern, recursive=True))

                if found_paths:
                    if transfer_documentation(py_file, found_paths[:3]):  # Limite à 3 candidats
                        updated_files += 1
                    else:
                        already_good += 1
                else:
                    print("  ⚠️  Aucun backup trouvé")
                    not_found += 1

    # Rapport final
    print("\n" + "=" * 70)
    print("📊 RAPPORT FINAL")
    print("=" * 70)
    print(f"Total fichiers traités:    {total_files}")
    print(f"✅ Fichiers mis à jour:    {updated_files}")
    print(f"ℹ️  Déjà documentés:       {already_good}")
    print(f"⚠️  Backups non trouvés:   {not_found}")
    print(f"\nTaux de succès: {(updated_files + already_good) / total_files * 100:.1f}%")

    print("\n✨ Transfert terminé!")


if __name__ == "__main__":
    main()
