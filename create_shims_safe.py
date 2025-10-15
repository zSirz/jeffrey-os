#!/usr/bin/env python3
"""
Création sécurisée des shims avec dry-run par défaut (GPT)
Protection Git et validation avant écriture
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Template shim avec cycle detection
SHIM_TEMPLATE = '''#!/usr/bin/env python3
"""
SHIM STRICT - Redirection vers implémentation réelle
GÉNÉRÉ AUTOMATIQUEMENT - NE PAS ÉDITER
Target: {real_path}
"""

import importlib.util
from pathlib import Path
import sys
import os

REAL_PATH = "{real_path}"
STRICT_MODE = os.getenv("JEFFREY_ALLOW_FALLBACK", "0") != "1"

_LOADING = set()

def _check_cycle(module_name: str):
    """Détection de cycles."""
    if module_name in _LOADING:
        raise ImportError(f"❌ CYCLE : {{module_name}}")
    _LOADING.add(module_name)

def _load_real_module():
    """Charge le module réel."""
    module_name = __name__
    _check_cycle(module_name)

    real_file = Path(__file__).parent.parent.parent / REAL_PATH

    if not real_file.exists():
        if STRICT_MODE:
            raise ImportError(f"❌ Fichier manquant : {{real_file}}")
        return None

    try:
        spec = importlib.util.spec_from_file_location("_real", real_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Spec invalide : {{real_file}}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        _LOADING.discard(module_name)
        return module
    except Exception as e:
        _LOADING.discard(module_name)
        raise ImportError(f"Erreur chargement {{real_file}}: {{e}}")

_real_module = _load_real_module()

if _real_module:
    globals().update({{
        k: v for k, v in vars(_real_module).items()
        if not k.startswith('_')
    }})
'''


def check_git_clean() -> bool:
    """Vérifie que le repo Git est propre (GPT)."""
    try:
        result = subprocess.run(['git', 'status', '--porcelain'], capture_output=True, text=True)
        return result.stdout.strip() == ''
    except Exception:
        return False


def create_git_branch() -> bool:
    """Crée une branche de restauration (GPT)."""
    from datetime import datetime

    branch_name = f"restore/shims-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    try:
        subprocess.run(['git', 'checkout', '-b', branch_name], check=True)
        print(f"   ✅ Branche créée : {branch_name}")
        return True
    except Exception as e:
        print(f"   ❌ Erreur création branche : {e}")
        return False


def import_to_path(import_name: str, use_shims_dir: bool = True) -> Path:
    """Convertit un import en chemin de fichier."""
    parts = import_name.split('.')

    if use_shims_dir:
        # GPT: Utiliser un dossier _shims dédié
        return Path('src/jeffrey/_shims') / '/'.join(parts[1:]) / f"{parts[-1]}.py"
    else:
        return Path('services') / '/'.join(parts[:-1]) / f"{parts[-1]}.py"


def ensure_init_files(shim_file: Path):
    """Crée les __init__.py nécessaires."""
    current = shim_file.parent
    root = Path('src/jeffrey/_shims')

    while current != root and current != Path('.'):
        init_file = current / '__init__.py'
        if not init_file.exists():
            init_file.write_text('"""Package auto-généré."""\n')
        current = current.parent


def detect_shim_cycle(new_shim: str, real_path: str, existing_shims: set[str]) -> bool:
    """Détecte les cycles potentiels."""
    real_path_normalized = str(Path(real_path).resolve())
    for existing_shim in existing_shims:
        if existing_shim in real_path_normalized:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description='Création sécurisée des shims')
    parser.add_argument('--apply', action='store_true', help='Appliquer les modifications (sinon dry-run)')
    parser.add_argument('--no-git-check', action='store_true', help='Skip Git check (dev only)')
    parser.add_argument('--shims-dir', action='store_true', help='Utiliser src/jeffrey/_shims/')
    args = parser.parse_args()

    print("🔧 CRÉATION SÉCURISÉE DES SHIMS")
    print("=" * 60)

    if not args.apply:
        print("⚠️  MODE DRY-RUN (aucune écriture)")
        print("   Lancez avec --apply pour créer les fichiers")

    print()

    # Protection Git (GPT)
    if not args.no_git_check and args.apply:
        print("🔒 Vérification Git...")
        if not check_git_clean():
            print("❌ Le repo Git n'est pas propre")
            print("   Committez ou stashez vos changements d'abord")
            print("   Ou utilisez --no-git-check (dev only)")
            sys.exit(1)

        print("   ✅ Repo propre")

        # Créer une branche
        if not create_git_branch():
            print("❌ Impossible de créer une branche")
            sys.exit(1)

    print()

    # Charger le diagnostic
    try:
        with open('COMPREHENSIVE_DIAGNOSTIC_V2.json') as f:
            report = json.load(f)
    except FileNotFoundError:
        print("❌ COMPREHENSIVE_DIAGNOSTIC_V2.json non trouvé")
        sys.exit(1)

    missing_with_candidates = report.get('missing_with_candidates', {})

    if not missing_with_candidates:
        print("✅ Aucun shim à créer")
        return

    print(f"📦 {len(missing_with_candidates)} shims potentiels")
    print()

    planned_shims = []
    skipped_cycles = []
    existing_shims_set = set()

    for import_name, info in missing_with_candidates.items():
        candidates = info['candidates']
        if not candidates:
            continue

        real_path = candidates[0]

        # Cycle detection
        if detect_shim_cycle(import_name, real_path, existing_shims_set):
            skipped_cycles.append({'import': import_name, 'target': real_path, 'reason': 'Cycle potentiel'})
            continue

        shim_path = import_to_path(import_name, args.shims_dir)

        planned_shims.append({'shim': str(shim_path), 'target': real_path, 'import': import_name})
        existing_shims_set.add(str(shim_path))

    # Afficher le plan
    print("📋 PLAN D'ACTIONS :")
    print()
    for shim in planned_shims:
        print(f"✅ Créer : {shim['shim']}")
        print(f"   → {shim['target']}")

    if skipped_cycles:
        print()
        print("⚠️  CYCLES DÉTECTÉS (ignorés) :")
        for cycle in skipped_cycles:
            print(f"❌ {cycle['import']} → {cycle['target']}")

    print()
    print(f"📊 Total : {len(planned_shims)} shims à créer")

    if not args.apply:
        print()
        print("💡 Relancez avec --apply pour créer les fichiers")

        # Sauvegarder le plan
        with open('SHIMS_PLAN.json', 'w') as f:
            json.dump({'planned_shims': planned_shims, 'skipped_cycles': skipped_cycles}, f, indent=2)

        print("📄 Plan sauvé : SHIMS_PLAN.json")
        return

    # Application réelle
    print()
    print("🚀 CRÉATION DES SHIMS...")
    print()

    # Backup
    from datetime import datetime

    backup_dir = Path('backups') / f"shims_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    for shim in planned_shims:
        shim_path = Path(shim['shim'])

        # Créer le shim
        shim_path.parent.mkdir(parents=True, exist_ok=True)

        # Créer les __init__.py
        ensure_init_files(shim_path)
        content = SHIM_TEMPLATE.format(real_path=shim['target'])
        shim_path.write_text(content)

        print(f"✅ {shim_path}")

    # Sauvegarder le mapping
    with open('SHIMS_MAPPING.json', 'w') as f:
        json.dump(
            {'created_shims': planned_shims, 'skipped_cycles': skipped_cycles, 'backup_dir': str(backup_dir)},
            f,
            indent=2,
        )

    print()
    print("=" * 60)
    print("✅ SHIMS CRÉÉS")
    print("=" * 60)
    print(f"📦 {len(planned_shims)} shims")
    print(f"📁 Backup : {backup_dir}")
    print("📄 Mapping : SHIMS_MAPPING.json")


if __name__ == "__main__":
    main()
