#!/usr/bin/env python3
"""
Crée des shims (ponts) pour permettre les imports depuis les bons emplacements
sans déplacer les fichiers actuels
"""

from pathlib import Path


def create_shim(target_dir: Path, target_file: str, source_import: str, class_name: str):
    """Crée un fichier shim qui ré-exporte depuis l'emplacement actuel"""

    # Créer le dossier si nécessaire
    target_dir.mkdir(parents=True, exist_ok=True)

    # Créer le __init__.py
    init_file = target_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text("# Jeffrey OS Module\n")
        print(f"   📄 Created {init_file.relative_to(Path.cwd())}")

    # Créer le shim
    shim_path = target_dir / target_file
    shim_content = f"""# Shim: ré-exporte depuis l'emplacement actuel (temporaire)
# TODO P1: Déplacer le vrai fichier ici et supprimer ce shim

try:
    from {source_import} import *  # noqa: F401,F403
    from {source_import} import {class_name}  # noqa: F401
except ImportError as e:
    print(f"Warning: Shim import failed: {{e}}")
    # Fallback pour éviter les erreurs complètes
    class {class_name}:
        def __init__(self):
            raise ImportError(f"{{{class_name}}} not available - check {source_import}")

# Pour debug
__shim__ = True
__original_location__ = "{source_import}"
"""

    shim_path.write_text(shim_content)
    print(f"   ✅ Created shim: {shim_path.relative_to(Path.cwd())}")
    return shim_path


def main():
    print("🌉 Creating shims for P0 modules")
    print("=" * 50)

    base_dir = Path.cwd()

    # Vérifier qu'on est dans le bon répertoire
    if not (base_dir / "src/jeffrey/core").exists():
        print("❌ Error: src/jeffrey/core not found. Run from project root.")
        return False

    # Créer tous les __init__.py nécessaires
    init_paths = [
        base_dir / "src/__init__.py",
        base_dir / "src/jeffrey/__init__.py",
        base_dir / "src/jeffrey/core/__init__.py",
        base_dir / "src/jeffrey/core/consciousness/__init__.py",
    ]

    for init_path in init_paths:
        if not init_path.exists():
            init_path.parent.mkdir(parents=True, exist_ok=True)
            init_path.write_text("# Jeffrey OS Module\n")
            print(f"   📄 Created {init_path.relative_to(base_dir)}")

    print("\n📦 Creating shims:")

    # Shim 1: DreamEngine (consciousness → dreaming)
    create_shim(
        target_dir=base_dir / "src/jeffrey/core/dreaming",
        target_file="dream_engine.py",
        source_import="..consciousness.dream_engine",
        class_name="DreamEngine",
    )

    # Shim 2: CognitiveSynthesis (consciousness → memory)
    create_shim(
        target_dir=base_dir / "src/jeffrey/core/memory",
        target_file="cognitive_synthesis.py",
        source_import="..consciousness.cognitive_synthesis",
        class_name="CognitiveSynthesis",
    )

    print("\n✅ Shims created successfully!")
    print("\n📝 What shims do:")
    print("   - Allow imports from CORRECT locations (dreaming/, memory/)")
    print("   - While keeping files in CURRENT locations (consciousness/)")
    print("   - Enable gradual migration without breaking existing code")

    return True


if __name__ == "__main__":
    success = main()
    import sys

    sys.exit(0 if success else 1)
