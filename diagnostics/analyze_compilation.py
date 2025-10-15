#!/usr/bin/env python3
# Fichier: diagnostics/analyze_compilation.py

import json
import pathlib
import subprocess
import sys
from collections import defaultdict

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


def diagnose_compilation():
    """Analyse complète des erreurs de compilation avec priorité"""
    root = pathlib.Path("src/jeffrey")
    errors = defaultdict(list)
    priority_map = {'syntax': 1, 'indentation': 2, 'import': 3, 'other': 4}

    print("🔍 Analyse des erreurs de compilation...")

    for py_file in root.rglob("*.py"):
        if '__pycache__' in str(py_file):
            continue

        result = subprocess.run([sys.executable, "-m", "py_compile", str(py_file)], capture_output=True, text=True)

        if result.returncode != 0:
            error_msg = result.stderr.strip()

            # Catégorisation avec priorité
            if 'SyntaxError' in error_msg:
                error_type = 'syntax'
            elif 'IndentationError' in error_msg:
                error_type = 'indentation'
            elif 'ImportError' in error_msg or 'ModuleNotFoundError' in error_msg:
                error_type = 'import'
            else:
                error_type = 'other'

            # Extraction du numéro de ligne
            line_num = None
            import re

            match = re.search(r'line (\d+)', error_msg)
            if match:
                line_num = int(match.group(1))

            errors[error_type].append(
                {
                    'file': _safe_rel(py_file),
                    'error': error_msg.split('\n')[-1] if error_msg else 'Unknown error',
                    'line': line_num,
                    'priority': priority_map[error_type],
                }
            )

    # Tri par priorité
    all_errors = []
    for error_type in sorted(errors.keys(), key=lambda x: priority_map[x]):
        all_errors.extend(errors[error_type])

    # Générer le rapport
    report = {
        'total_files': len(all_errors),
        'by_type': {k: len(v) for k, v in errors.items()},
        'critical_errors': all_errors[:10],  # Top 10 plus critiques
        'all_errors': all_errors,
    }

    # Sauvegarder
    report_path = pathlib.Path("diagnostics/compilation_report.json")
    report_path.write_text(json.dumps(report, indent=2))

    # Afficher résumé
    print("\n📊 RÉSUMÉ:")
    print(f"  Total: {report['total_files']} fichiers avec erreurs")
    for error_type, count in report['by_type'].items():
        print(f"  - {error_type}: {count}")

    if report['critical_errors']:
        print("\n🔥 TOP 5 ERREURS CRITIQUES:")
        for err in report['critical_errors'][:5]:
            print(f"  {err['file']}:{err.get('line', '?')} - {err['error'][:80]}")

    return report


if __name__ == "__main__":
    diagnose_compilation()
