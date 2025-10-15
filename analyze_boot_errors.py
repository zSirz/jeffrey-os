#!/usr/bin/env python3
"""
Analyse intelligente du boot.log avec :
- Grille des 4 Archétypes d'Erreurs (Gemini)
- Auto-suggestion de fixes (Grok)
- Mapping PKG_MAP pour noms corrects (GPT/Marc)
- Rapport actionnable (GPT/Marc)
"""

import os
import re
import sys
from collections import Counter
from pathlib import Path

# === GARDE-FOUS (GPT/Marc) ===
os.environ['PYTHONUTF8'] = '1'

if not (Path('.git').exists() and Path('src/jeffrey').exists()):
    print("❌ Lance ce script depuis la racine du repo Jeffrey OS")
    sys.exit(1)

# === MAPPING PACKAGES PIP (GPT/Marc) ===
PKG_MAP = {
    "sklearn": "scikit-learn",
    "PIL": "Pillow",
    "yaml": "PyYAML",
    "cv2": "opencv-python",
    "Crypto": "pycryptodome",
    "bs4": "beautifulsoup4",
    "dateutil": "python-dateutil",
}

# === GRILLE DES 4 ARCHÉTYPES D'ERREURS (Gemini) ===
ARCHETYPE_PATTERNS = {
    'TYPE_1_PATH': {
        'pattern': r"ModuleNotFoundError: No module named 'core\.[\w\.]+'",
        'description': "Erreur de chemin d'import (relatif vs absolu)",
        'action': "Correction automatisée via fix_core_imports.py",
        'priority': 1,
    },
    'TYPE_2_EXTERNAL': {
        'pattern': r"ModuleNotFoundError: No module named '(?!jeffrey\.)[\w\.]+'",
        'description': "Dépendance externe manquante",
        'action': "Installation via pip install",
        'priority': 2,
    },
    'TYPE_3_INTERNAL': {
        'pattern': r"ModuleNotFoundError: No module named 'jeffrey\.[\w\.]+'",
        'description': "Module interne Jeffrey vraiment manquant",
        'action': "Restauration manuelle ou shim",
        'priority': 3,
    },
    'TYPE_4_INIT': {
        'pattern': r"(AttributeError|TypeError|ValueError).*",
        'description': "Erreur d'initialisation (import OK, mais bug interne)",
        'action': "Débogage approfondi du module",
        'priority': 4,
    },
}


def analyze_boot_log(log_path: Path) -> dict:
    """Analyse le boot.log et catégorise les erreurs."""
    content = log_path.read_text()

    # Extraire toutes les erreurs
    all_errors = re.findall(r'(ModuleNotFoundError|ImportError|AttributeError|TypeError|ValueError): (.+)', content)

    # Catégoriser par archétype
    categorized = {'TYPE_1_PATH': [], 'TYPE_2_EXTERNAL': [], 'TYPE_3_INTERNAL': [], 'TYPE_4_INIT': []}

    for error_type, error_msg in all_errors:
        full_error = f"{error_type}: {error_msg}"

        # Détecter le type d'erreur
        if 'core.' in error_msg and 'jeffrey.core' not in error_msg:
            categorized['TYPE_1_PATH'].append(error_msg)
        elif 'ModuleNotFoundError' in error_type and 'jeffrey.' in error_msg:
            categorized['TYPE_3_INTERNAL'].append(error_msg)
        elif 'ModuleNotFoundError' in error_type:
            categorized['TYPE_2_EXTERNAL'].append(error_msg)
        else:
            categorized['TYPE_4_INIT'].append(full_error)

    # Extraire les modules manquants
    missing_modules = re.findall(r"No module named '([^']+)'", content)
    module_counter = Counter(missing_modules)

    return {'categorized': categorized, 'missing_modules': module_counter, 'total_errors': len(all_errors)}


def suggest_fixes(analysis: dict) -> list[dict]:
    """Génère des suggestions de fixes automatiques (Grok + GPT/Marc)."""
    suggestions = []

    # Type 1 : Imports obsolètes
    if analysis['categorized']['TYPE_1_PATH']:
        suggestions.append(
            {
                'priority': 1,
                'type': 'AUTO_FIX',
                'command': 'python3 fix_core_imports_complete.py',
                'description': f"Corriger {len(analysis['categorized']['TYPE_1_PATH'])} imports obsolètes",
            }
        )

    # Type 2 : Dépendances externes (avec mapping PKG_MAP)
    external_modules = analysis['categorized']['TYPE_2_EXTERNAL']
    if external_modules:
        # Extraire les noms de packages uniques
        packages = set()
        for error in external_modules:
            match = re.search(r"'([\w]+)", error)
            if match:
                pkg = match.group(1)
                # Utiliser PKG_MAP pour le nom correct
                correct_pkg = PKG_MAP.get(pkg, pkg)
                packages.add(correct_pkg)

        for pkg in list(packages)[:5]:  # Top 5
            suggestions.append(
                {
                    'priority': 2,
                    'type': 'INSTALL',
                    'command': f'pip install {pkg}',
                    'description': f"Installer le package externe {pkg}",
                }
            )

    # Type 3 : Modules internes manquants (top 5)
    internal_modules = analysis['categorized']['TYPE_3_INTERNAL']
    if internal_modules:
        top_5_internal = Counter(internal_modules).most_common(5)
        for module, count in top_5_internal:
            match = re.search(r"'(jeffrey\.[\w\.]+)'", module)
            if match:
                module_name = match.group(1)
                suggestions.append(
                    {
                        'priority': 3,
                        'type': 'RESTORE',
                        'command': f'# Créer shim ou restaurer {module_name}',
                        'description': f"Module {module_name} manquant ({count}x)",
                    }
                )

    # Privacy check (Grok)
    if any('memory' in str(m).lower() or 'emotional' in str(m).lower() for m in internal_modules):
        suggestions.append(
            {
                'priority': 4,
                'type': 'PRIVACY',
                'command': '# Vérifier encryption dans unified_memory.store()',
                'description': "⚠️  Modules émotionnels détectés - vérifier privacy/encryption",
            }
        )

    return sorted(suggestions, key=lambda x: x['priority'])


def generate_report(analysis: dict, suggestions: list[dict], output_path: Path):
    """Génère le rapport BOOT_ANALYSIS.md."""
    report = []

    report.append("# 🚀 RAPPORT D'ANALYSE DE DÉMARRAGE JEFFREY OS\n")
    report.append(f"## Date : {Path('boot.log').stat().st_mtime}\n")

    # Résumé exécutif
    report.append("## 📊 RÉSUMÉ EXÉCUTIF\n")
    report.append(f"- **Total erreurs détectées :** {analysis['total_errors']}\n")
    report.append(f"- **Modules manquants uniques :** {len(analysis['missing_modules'])}\n")

    # Grille des 4 Archétypes (Gemini)
    report.append("\n## 🔍 ANALYSE PAR ARCHÉTYPE (Gemini)\n")

    for archetype_key, archetype_info in ARCHETYPE_PATTERNS.items():
        count = len(analysis['categorized'][archetype_key])
        if count > 0:
            report.append(f"\n### {archetype_key.replace('_', ' ')} ({count} erreurs)\n")
            report.append(f"**Description :** {archetype_info['description']}\n")
            report.append(f"**Action :** {archetype_info['action']}\n")
            report.append(f"**Priorité :** {archetype_info['priority']}\n")

            # Afficher quelques exemples
            examples = analysis['categorized'][archetype_key][:3]
            if examples:
                report.append("\n**Exemples :**\n")
                for ex in examples:
                    report.append(f"- `{ex}`\n")

    # Top 10 Modules Manquants
    report.append("\n## 📈 TOP 10 MODULES MANQUANTS\n")
    for module, count in analysis['missing_modules'].most_common(10):
        report.append(f"{count:3}x  `{module}`\n")

    # Suggestions automatiques (Grok + GPT/Marc)
    report.append("\n## 🎯 ACTIONS RECOMMANDÉES (Auto-générées)\n")
    report.append("\n**Exécuter dans l'ordre :**\n")
    for i, suggestion in enumerate(suggestions, 1):
        report.append(f"\n### {i}. {suggestion['description']}\n")
        report.append(f"```bash\n{suggestion['command']}\n```\n")

    # Catégorisation interne vs externe
    jeffrey_modules = [m for m in analysis['missing_modules'] if m.startswith('jeffrey.')]
    external_modules = [m for m in analysis['missing_modules'] if not m.startswith('jeffrey.')]

    report.append("\n## 📦 CATÉGORISATION\n")
    report.append(f"- **Modules internes (jeffrey.*) :** {len(jeffrey_modules)}\n")
    report.append(f"- **Modules externes :** {len(external_modules)}\n")

    if external_modules:
        report.append("\n**Top 5 modules externes à installer :**\n")
        for m in external_modules[:5]:
            # Appliquer PKG_MAP
            correct_pkg = PKG_MAP.get(m, m)
            if correct_pkg != m:
                report.append(f"- `{m}` → `pip install {correct_pkg}`\n")
            else:
                report.append(f"- `{m}`\n")

    # Doctrine Gemini
    report.append("\n## 📖 DOCTRINE D'EXÉCUTION (Gemini : La Règle des Trois)\n")
    report.append("\n### 1. Prioriser (Focus sur le N°1)\n")
    report.append("- Ne traiter qu'UNE seule action à la fois (la première de la liste ci-dessus)\n")
    report.append("- Ignorer tout le reste jusqu'à résolution\n")
    report.append("\n### 2. Isoler (Une Branche, Un Fix)\n")
    report.append("- Créer une branche Git dédiée : `git checkout -b fix/nom-du-fix`\n")
    report.append("- Faire le fix, tester, committer\n")
    report.append("\n### 3. Valider (Le Test Miroir)\n")
    report.append("- Relancer `bash test_boot_complete.sh`\n")
    report.append("- Vérifier que l'erreur corrigée a disparu\n")
    report.append("- Ré-analyser avec `python3 analyze_boot_errors.py`\n")
    report.append("- Recommencer avec la nouvelle priorité N°1\n")

    # Écrire le rapport
    output_path.write_text(''.join(report))
    print(f"\n✅ Rapport généré : {output_path}")


def main():
    """Exécution principale."""
    log_path = Path('boot.log')

    if not log_path.exists():
        print("❌ Fichier boot.log introuvable")
        return 1

    print("🔍 Analyse du boot.log...")
    analysis = analyze_boot_log(log_path)

    print(f"📊 {analysis['total_errors']} erreurs détectées")
    print(f"📦 {len(analysis['missing_modules'])} modules manquants uniques")

    # Générer suggestions
    suggestions = suggest_fixes(analysis)
    print(f"🎯 {len(suggestions)} actions recommandées\n")

    # Afficher le top 3 des suggestions
    print("🚀 TOP 3 ACTIONS À FAIRE MAINTENANT :\n")
    for i, suggestion in enumerate(suggestions[:3], 1):
        print(f"{i}. {suggestion['description']}")
        print(f"   → {suggestion['command']}\n")

    # Générer rapport complet
    generate_report(analysis, suggestions, Path('BOOT_ANALYSIS.md'))

    return 0


if __name__ == '__main__':
    sys.exit(main())
