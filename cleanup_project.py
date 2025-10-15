#!/usr/bin/env python3
"""
Script de nettoyage final du projet
Supprime les fichiers temporaires et réorganise
"""

import os
import shutil
from pathlib import Path


def cleanup():
    """Nettoie le projet"""

    print("🧹 NETTOYAGE DU PROJET JEFFREY OS")
    print("=" * 50)

    # Supprimer les __pycache__
    print("\n📁 Suppression des __pycache__...")
    count = 0
    for pycache in Path('.').rglob('__pycache__'):
        try:
            shutil.rmtree(pycache)
            count += 1
        except Exception as e:
            print(f"   ⚠️ Impossible de supprimer {pycache}: {e}")
    print(f"   ✅ {count} dossiers supprimés")

    # Supprimer les .pyc
    print("\n📁 Suppression des .pyc...")
    count = 0
    for pyc in Path('.').rglob('*.pyc'):
        try:
            pyc.unlink()
            count += 1
        except Exception as e:
            print(f"   ⚠️ Impossible de supprimer {pyc}: {e}")
    print(f"   ✅ {count} fichiers supprimés")

    # Vérifier les imports Kivy
    print("\n🔍 Vérification des imports Kivy...")
    if Path('check_kivy_imports.py').exists():
        os.system('python check_kivy_imports.py')
    else:
        print("   ⚠️ check_kivy_imports.py non trouvé")

    # Créer les dossiers manquants
    dirs_to_create = ['data', 'logs', 'tests', 'docs']

    print("\n📁 Création des dossiers standards...")
    for dir_name in dirs_to_create:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"   ✅ {dir_name}/")

    # Nettoyer l'ancien dossier base_old_dir s'il existe
    old_base = Path('src/jeffrey/bridge/base_old_dir')
    if old_base.exists():
        print("\n📁 Suppression de l'ancien dossier base_old_dir...")
        try:
            shutil.rmtree(old_base)
            print("   ✅ Supprimé")
        except Exception as e:
            print(f"   ⚠️ Erreur: {e}")

    print("\n✅ Nettoyage terminé!")


if __name__ == "__main__":
    cleanup()
