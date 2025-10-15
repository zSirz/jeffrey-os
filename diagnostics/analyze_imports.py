#!/usr/bin/env python3
# Fichier: diagnostics/analyze_imports.py

import ast
import json
from collections import defaultdict
from pathlib import Path

# --- injected helper: safe relative path ---
from pathlib import Path as _Path

__CWD = _Path.cwd().resolve()


def _safe_rel(p) -> str:
    try:
        p = p if isinstance(p, _Path) else _Path(p)
        p = p.resolve()
        return str(p.relative_to(__CWD))
    except Exception:
        return str(p)


# --- end helper ---


def diagnose_imports():
    """Analyse complète des imports, dépendances et modules orphelins"""
    root = Path("src/jeffrey")

    analysis = {
        'legacy_imports': [],
        'missing_modules': [],
        'orphan_modules': [],  # Modules jamais importés
        'missing_init_files': [],
        'import_graph': defaultdict(set),  # Pour détecter les cycles
    }

    # Collecter tous les modules existants avec nom fully-qualified
    all_modules = set()
    for py_file in root.rglob("*.py"):
        if '__pycache__' not in str(py_file):
            module_path = py_file.relative_to(root).with_suffix('')
            # Nom fully-qualified pour éviter les erreurs
            module_name = "jeffrey." + str(module_path).replace('/', '.')
            all_modules.add(module_name)

    # Analyser les imports
    imported_modules = set()

    print("📦 Analyse des imports et dépendances...")

    for py_file in root.rglob("*.py"):
        if '__pycache__' in str(py_file):
            continue

        try:
            content = py_file.read_text(encoding='utf-8')
            module_path = py_file.relative_to(root).with_suffix('')
            module_name = "jeffrey." + str(module_path).replace('/', '.')

            # Imports legacy
            for i, line in enumerate(content.splitlines(), 1):
                if not line.strip().startswith('#'):
                    if 'Orchestrateur_IA' in line or 'src.jeffrey' in line:
                        analysis['legacy_imports'].append(
                            {'file': _safe_rel(py_file), 'line': i, 'import': line.strip()}
                        )

            # Analyse AST pour graphe d'imports
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                if alias.name.startswith('jeffrey'):
                                    imported_modules.add(alias.name)
                                    analysis['import_graph'][module_name].add(alias.name)
                        elif node.module and node.module.startswith('jeffrey'):
                            imported_modules.add(node.module)
                            analysis['import_graph'][module_name].add(node.module)
            except:
                pass

        except Exception as e:
            print(f"  ⚠️ Erreur analyse imports {py_file.name}: {e}")

    # Identifier les modules orphelins
    orphans = all_modules - imported_modules
    for orphan in orphans:
        if not orphan.endswith('__init__'):  # Ignorer les __init__.py
            analysis['orphan_modules'].append({'module': orphan, 'path': f"src/{orphan.replace('.', '/')}.py"})

    # Vérifier __init__.py manquants
    for directory in root.rglob("*"):
        if directory.is_dir() and '__pycache__' not in str(directory):
            init_file = directory / "__init__.py"
            if not init_file.exists():
                analysis['missing_init_files'].append(_safe_rel(directory))

    # Vérifier les modules manquants critiques
    critical_modules = [
        'jeffrey.core.memory.memory_manager',
        'jeffrey.core.orchestration.ia_orchestrator_ultimate',
        'jeffrey.api.audit_logger_enhanced',
        'jeffrey.core.sandbox_manager_enhanced',
    ]

    for module in critical_modules:
        module_path = root / module.replace('jeffrey.', '').replace('.', '/')
        if not module_path.with_suffix('.py').exists():
            analysis['missing_modules'].append(
                {'module': module, 'critical': True, 'expected_path': str(module_path.with_suffix('.py'))}
            )

    # Générer le rapport
    report = {
        'total_issues': (
            len(analysis['legacy_imports'])
            + len(analysis['missing_modules'])
            + len(analysis['orphan_modules'])
            + len(analysis['missing_init_files'])
        ),
        'orphan_count': len(analysis['orphan_modules']),
        'missing_critical': len([m for m in analysis['missing_modules'] if m.get('critical')]),
        'legacy_count': len(analysis['legacy_imports']),
        'details': analysis,
    }

    # Sauvegarder
    report_path = Path("diagnostics/imports_report.json")
    report_path.write_text(json.dumps(report, indent=2, default=str))

    # Afficher résumé
    print("\n📊 RÉSUMÉ:")
    print(f"  Total problèmes: {report['total_issues']}")
    print(f"  - Modules orphelins: {report['orphan_count']}")
    print(f"  - Modules critiques manquants: {report['missing_critical']}")
    print(f"  - Imports legacy: {report['legacy_count']}")
    print(f"  - Dossiers sans __init__.py: {len(analysis['missing_init_files'])}")

    if report['orphan_count'] > 0:
        print("\n📍 TOP 5 MODULES ORPHELINS:")
        for orphan in analysis['orphan_modules'][:5]:
            print(f"  - {orphan['module']}")

    return report


if __name__ == "__main__":
    diagnose_imports()
