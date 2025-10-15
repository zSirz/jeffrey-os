#!/usr/bin/env python3
"""
Scanner pour trouver les modules non migrés
"""

import json
from collections import defaultdict
from pathlib import Path


def get_migrated_modules():
    """Liste les modules déjà migrés dans Jeffrey_OS"""
    jeffrey_os = Path.home() / "Desktop" / "Jeffrey_OS"
    migrated = set()

    for py_file in jeffrey_os.rglob("*.py"):
        if "__pycache__" not in str(py_file):
            migrated.add(py_file.name)

    for rs_file in jeffrey_os.rglob("*.rs"):
        migrated.add(rs_file.name)

    return migrated


def scan_icloud_jeffrey():
    """Scanner tous les dossiers Jeffrey dans iCloud"""
    # Chemins iCloud à scanner
    icloud_base = Path.home() / "Library" / "Mobile Documents" / "com~apple~CloudDocs"

    jeffrey_folders = [
        "Jeffrey",
        "Jeffrey_OS",
        "Jeffrey_DEV",
        "Jeffrey_Phoenix",
        "JEFFREY_UNIFIED",
        "Jeffrey_SYN",
        "Jeffrey_Apps",
        "CashZen",
    ]

    all_modules = defaultdict(list)

    for folder in jeffrey_folders:
        folder_path = icloud_base / folder
        if folder_path.exists():
            print(f"📁 Scanning {folder}...")

            # Scanner les fichiers Python
            for py_file in folder_path.rglob("*.py"):
                if "__pycache__" not in str(py_file):
                    relative_path = py_file.relative_to(folder_path)
                    all_modules[py_file.name].append(
                        {"folder": folder, "path": str(relative_path), "size": py_file.stat().st_size}
                    )

            # Scanner les fichiers Rust
            for rs_file in folder_path.rglob("*.rs"):
                if "target" not in str(rs_file):
                    relative_path = rs_file.relative_to(folder_path)
                    all_modules[rs_file.name].append(
                        {"folder": folder, "path": str(relative_path), "size": rs_file.stat().st_size}
                    )

    return all_modules


def analyze_missing():
    """Analyse principale"""
    print("🔍 ANALYSE DES MODULES NON MIGRÉS")
    print("=" * 50)

    # 1. Modules déjà migrés
    migrated = get_migrated_modules()
    print(f"✅ Modules migrés : {len(migrated)}")

    # 2. Tous les modules dans iCloud
    all_modules = scan_icloud_jeffrey()
    print(f"📊 Total modules trouvés dans iCloud : {len(all_modules)}")

    # 3. Identifier les non migrés
    not_migrated = {}
    potentially_important = []

    for module_name, locations in all_modules.items():
        if module_name not in migrated:
            not_migrated[module_name] = locations

            # Identifier les potentiellement importants
            max_size = max(loc['size'] for loc in locations)
            if max_size > 5000:  # Plus de 5KB
                potentially_important.append({"name": module_name, "size": max_size, "locations": locations})

    # 4. Afficher les résultats
    print(f"\n❌ MODULES NON MIGRÉS : {len(not_migrated)}")
    print("-" * 50)

    # Trier par importance (taille)
    potentially_important.sort(key=lambda x: x['size'], reverse=True)

    print("\n🔥 TOP 30 MODULES IMPORTANTS NON MIGRÉS (par taille):")
    for i, module in enumerate(potentially_important[:30], 1):
        print(f"\n{i}. {module['name']} ({module['size'] // 1024}KB)")
        for loc in module['locations']:
            print(f"   📁 {loc['folder']}/{loc['path']}")

    # 5. Catégoriser par type
    categories = defaultdict(list)
    for name in not_migrated.keys():
        if "test" in name.lower():
            categories["tests"].append(name)
        elif "ui" in name.lower() or "widget" in name.lower():
            categories["ui"].append(name)
        elif "voice" in name.lower():
            categories["voice"].append(name)
        elif "memory" in name.lower():
            categories["memory"].append(name)
        elif "emotion" in name.lower():
            categories["emotions"].append(name)
        elif "consciousness" in name.lower():
            categories["consciousness"].append(name)
        elif "learn" in name.lower():
            categories["learning"].append(name)
        else:
            categories["other"].append(name)

    print("\n📊 MODULES NON MIGRÉS PAR CATÉGORIE:")
    for category, modules in categories.items():
        print(f"\n{category.upper()} ({len(modules)} modules):")
        for m in sorted(modules)[:10]:  # Top 10 par catégorie
            print(f"   • {m}")

    # 6. Sauvegarder le rapport
    report = {
        "migrated_count": len(migrated),
        "total_in_icloud": len(all_modules),
        "not_migrated_count": len(not_migrated),
        "not_migrated": not_migrated,
        "important_missing": potentially_important[:50],
    }

    report_path = Path.home() / "Desktop" / "Jeffrey_OS" / "missing_modules_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n💾 Rapport complet sauvé : {report_path}")

    # Statistiques finales
    print("\n" + "=" * 50)
    print("📈 RÉSUMÉ FINAL:")
    print(f"   • Modules migrés: {len(migrated)}")
    print(f"   • Modules dans iCloud: {len(all_modules)}")
    print(f"   • Modules non migrés: {len(not_migrated)}")
    print(f"   • Taux de migration: {len(migrated) * 100 / len(all_modules):.1f}%")


if __name__ == "__main__":
    analyze_missing()
