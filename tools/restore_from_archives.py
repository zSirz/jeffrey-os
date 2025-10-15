#!/usr/bin/env python3
"""
Outil de restauration intelligente depuis les archives
"""

import shutil
import sys
from os import getenv
from pathlib import Path


def find_in_archives(filename: str, archive_dirs: list[str]) -> Path | None:
    """Cherche un fichier dans les archives"""
    for archive in archive_dirs:
        archive_path = Path(archive)
        if not archive_path.exists():
            continue

        # Recherche récursive
        matches = list(archive_path.rglob(f"*{filename}*"))
        if matches:
            # Prendre le plus récent
            most_recent = max(matches, key=lambda p: p.stat().st_mtime)
            return most_recent

    return None


def restore_file(source: Path, target: Path, backup_dir: Path) -> bool:
    """Restaure un fichier avec backup de l'existant"""
    try:
        # Créer le dossier cible
        target.parent.mkdir(parents=True, exist_ok=True)

        # Backup si le fichier existe déjà
        if target.exists():
            backup_path = backup_dir / target.name
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, backup_path)
            print(f"   📦 Backup : {backup_path}")

        # Copier le fichier
        shutil.copy2(source, target)
        print(f"   ✅ Restauré : {target}")

        # Vérifier la syntaxe Python
        import py_compile

        try:
            py_compile.compile(str(target), doraise=True)
            print("   ✅ Syntaxe valide")
            return True
        except py_compile.PyCompileError as e:
            print(f"   ❌ Erreur de syntaxe : {e}")
            return False

    except Exception as e:
        print(f"   ❌ Erreur de restauration : {e}")
        return False


def main():
    if len(sys.argv) < 4:
        print("Usage: restore_from_archives.py <filename> <target_path> <archive_dir1> [archive_dir2...]")
        sys.exit(1)

    filename = sys.argv[1]
    target = Path(sys.argv[2])
    archives = sys.argv[3:]
    backup_dir = Path(getenv("BACKUPS_DIR", "backups_repair")) / "restored_files"

    print(f"🔍 Recherche de '{filename}' dans les archives...")

    source = find_in_archives(filename, archives)

    if source is None:
        print(f"❌ Fichier '{filename}' introuvable dans les archives")
        return 1

    print(f"✅ Trouvé : {source}")
    print(f"   Taille : {source.stat().st_size} bytes")
    print(f"   Date : {source.stat().st_mtime}")

    print(f"\n📋 Restauration vers : {target}")

    if restore_file(source, target, backup_dir):
        print("✅ Restauration réussie")
        return 0
    else:
        print("❌ Restauration échouée")
        return 1


if __name__ == "__main__":
    sys.exit(main())
