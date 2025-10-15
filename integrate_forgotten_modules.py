#!/usr/bin/env python3
"""
Script pour intégrer automatiquement tous les modules oubliés dans Jeffrey OS
"""

import shutil
from pathlib import Path


def integrate_modules():
    """Intègre tous les modules oubliés dans la structure Jeffrey OS"""

    base_dir = Path(__file__).parent
    modules_dir = base_dir / "Modules oubliés"
    src_dir = base_dir / "src" / "jeffrey"

    if not modules_dir.exists():
        print("❌ Dossier 'Modules oubliés' non trouvé!")
        return False

    print("=" * 60)
    print("🚀 INTÉGRATION DES MODULES OUBLIÉS")
    print("=" * 60)

    # Mapping des modules vers leurs destinations
    module_mapping = {
        # Dreaming System
        "dream_evaluator.py": "core/dreaming/",
        "dream_state.py": "core/dreaming/",
        "dream_suggester.py": "core/dreaming/",
        "jeffrey_dreammode_integration.py": "core/dreaming/",
        "neural_mutator.py": "core/dreaming/",
        "variant_generator.py": "core/dreaming/",
        "scenario_simulator.py": "core/dreaming/",
        "ethical_guard.py": "core/dreaming/",
        # Feedback System
        "feedback_analyzer.py": "core/feedback/",
        "human_interface.py": "core/feedback/",
        "proposal_manager.py": "core/feedback/",
        # Learning Advanced
        "causal_predictor.py": "core/learning/",
        "explainer.py": "core/learning/",
        "feature_extractor.py": "core/learning/",
        "meta_learner.py": "core/learning/",
        # Monitoring
        "alert_chainer.py": "infrastructure/monitoring/",
        "baseline_tracker.py": "infrastructure/monitoring/",
        "delta_analyzer.py": "infrastructure/monitoring/",
        # Security
        "adaptive_rotator.py": "core/security/",
    }

    # Stats
    successful = 0
    failed = 0
    skipped = 0

    print("\n📦 Déplacement des modules Python...")
    print("-" * 40)

    # Déplacer chaque module
    for module, dest_path in module_mapping.items():
        source = modules_dir / module
        if source.exists():
            dest_dir = src_dir / dest_path
            dest_dir.mkdir(parents=True, exist_ok=True)

            dest_file = dest_dir / module

            # Check if file already exists
            if dest_file.exists():
                print(f"⚠️  {module} → déjà présent, ignoré")
                skipped += 1
            else:
                try:
                    shutil.copy2(source, dest_file)
                    print(f"✅ {module} → {dest_path}")
                    successful += 1
                except Exception as e:
                    print(f"❌ {module} → Erreur: {e}")
                    failed += 1
        else:
            print(f"⚠️  {module} → non trouvé dans Modules oubliés")
            skipped += 1

    # Déplacer l'UI Avatar
    print("\n🎨 Intégration de l'UI Avatar Kivy...")
    print("-" * 40)

    ui_source = modules_dir / "ui"
    if ui_source.exists():
        ui_dest = src_dir / "interfaces" / "ui" / "avatar"
        ui_dest.mkdir(parents=True, exist_ok=True)

        # Copier tous les fichiers UI
        ui_files = 0
        for item in ui_source.iterdir():
            try:
                if item.is_file() and item.suffix == ".py":
                    dest_file = ui_dest / item.name
                    if not dest_file.exists():
                        shutil.copy2(item, dest_file)
                        ui_files += 1
                        print(f"  ✅ {item.name}")
                elif item.is_dir():
                    dest_subdir = ui_dest / item.name
                    if not dest_subdir.exists():
                        shutil.copytree(item, dest_subdir)
                        print(f"  ✅ {item.name}/ (dossier)")
            except Exception as e:
                print(f"  ❌ Erreur avec {item.name}: {e}")

        print(f"✅ UI Avatar → {ui_files} fichiers copiés vers interfaces/ui/avatar/")
        successful += ui_files

    # Déplacer les widgets
    print("\n🔧 Intégration des Widgets Kivy...")
    print("-" * 40)

    widgets_source = modules_dir / "widgets"
    if widgets_source.exists():
        widgets_dest = src_dir / "interfaces" / "ui" / "widgets" / "kivy"
        widgets_dest.mkdir(parents=True, exist_ok=True)

        widget_files = 0
        for item in widgets_source.iterdir():
            if item.is_file() and item.suffix == ".py":
                dest_file = widgets_dest / item.name
                if not dest_file.exists():
                    shutil.copy2(item, dest_file)
                    widget_files += 1
                    print(f"  ✅ {item.name}")

        print(f"✅ Widgets → {widget_files} fichiers copiés vers interfaces/ui/widgets/kivy/")
        successful += widget_files

    # Créer les __init__.py
    print("\n📝 Création des fichiers __init__.py...")
    print("-" * 40)

    init_files = [
        src_dir / "core" / "dreaming" / "__init__.py",
        src_dir / "core" / "feedback" / "__init__.py",
        src_dir / "interfaces" / "ui" / "avatar" / "__init__.py",
        src_dir / "interfaces" / "ui" / "widgets" / "kivy" / "__init__.py",
    ]

    for init_file in init_files:
        if not init_file.exists():
            init_file.parent.mkdir(parents=True, exist_ok=True)
            init_file.write_text("")
            print(f"  ✅ {init_file.relative_to(base_dir)}")

    # Résumé
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DE L'INTÉGRATION")
    print("=" * 60)
    print(f"✅ Réussis: {successful}")
    print(f"⚠️  Ignorés: {skipped}")
    print(f"❌ Échecs: {failed}")

    print("\n🎯 PROCHAINES ÉTAPES:")
    print("-" * 40)
    print("1. Installer les dépendances:")
    print("   pip install kivy kivymd pillow")
    print("   pip install scikit-learn numpy pandas")
    print("   pip install nltk spacy textblob")
    print("\n2. Tester l'avatar UI:")
    print("   python test_avatar_ui.py")
    print("\n3. Lancer Jeffrey avec les nouveaux modules:")
    print("   python jeffrey_brain.py")

    return successful > 0


if __name__ == "__main__":
    success = integrate_modules()
    exit(0 if success else 1)
