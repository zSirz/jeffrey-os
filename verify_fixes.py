#!/usr/bin/env python3
"""Script de vérification des corrections d'indentation pour Jeffrey OS."""

import ast
import sys
from pathlib import Path


def check_file_syntax(filepath):
    """Vérifie la syntaxe d'un fichier Python."""
    path_obj = Path(filepath)
    try:
        with open(filepath, encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        print(f"✅ {path_obj.name} - Syntaxe OK")
        return True
    except SyntaxError as e:
        print(f"❌ {path_obj.name} - Erreur ligne {e.lineno}: {e.msg}")
        return False
    except IndentationError as e:
        print(f"❌ {path_obj.name} - Indentation ligne {e.lineno}: {e.msg}")
        return False
    except Exception as e:
        print(f"❌ {path_obj.name} - Erreur inattendue: {e}")
        return False


# Fichiers à vérifier
files_to_check = [
    'src/jeffrey/core/orchestration/ia_orchestrator_ultimate.py',
    'src/jeffrey/core/consciousness/jeffrey_consciousness_v3.py',
]

print("🔧 VÉRIFICATION DES CORRECTIONS D'INDENTATION - JEFFREY OS")
print("=" * 60)
print()

all_ok = True
for file_path in files_to_check:
    if Path(file_path).exists():
        if not check_file_syntax(file_path):
            all_ok = False
    else:
        print(f"⚠️  {file_path} - Fichier non trouvé")
        all_ok = False

print()
print("=" * 60)
if all_ok:
    print("✅ SUCCÈS ! Tous les fichiers sont syntaxiquement corrects!")
    print()
    print("📋 RAPPORT DES CORRECTIONS:")
    print("  • ia_orchestrator_ultimate.py : Ligne 391-397 - Docstring de MockProfessor indentée")
    print("  • jeffrey_consciousness_v3.py : Ligne 29 - Import __future__ déplacé au début")
    print()
    print("🚀 Tu peux maintenant lancer : python start_jeffrey.py")
else:
    print("❌ ÉCHEC ! Des erreurs persistent. Corrige-les avant de continuer.")
    sys.exit(1)
