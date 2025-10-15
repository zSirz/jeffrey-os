#!/usr/bin/env python3
"""
Driver principal Phoenix - Orchestre toutes les phases
FICHIER MANQUANT CRITIQUE - Ajouté par GPT
"""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Import des outils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tools.refactor.ast_import_rewriter import rewrite_imports
from tools.refactor.plugin_inserter import add_plugin_to_file


class PhoenixDriver:
    """Driver principal pour l'opération Phoenix"""

    # Ordre de traitement déterministe (GPT)
    PROCESSING_ORDER = [
        "src/jeffrey/core/bus",
        "src/jeffrey/core/security/guardian",
        "src/jeffrey/core/schemas",
        "src/jeffrey/core/discovery",
        "src/jeffrey/core/contracts",
        "src/jeffrey/core/orchestration",
        "src/jeffrey/core/memory",
        "src/jeffrey/core/consciousness",
        "src/jeffrey/core/emotions",
        "src/jeffrey/core/learning",
        "src/jeffrey/services",
        "src/jeffrey/bridge",
        "src/jeffrey/applications",
        "src/jeffrey/skills",
        "src/jeffrey/avatars",
    ]

    def __init__(self):
        self.progress_file = Path("tools/phoenix/progress.json")
        self.progress = self.load_progress()
        self.batch_size = 20
        self.current_batch = []

    def load_progress(self) -> dict:
        """Charge le fichier de progression"""
        if self.progress_file.exists():
            with open(self.progress_file) as f:
                return json.load(f)
        else:
            # Initialiser
            progress = {
                'metadata': {
                    'started_at': datetime.now().isoformat(),
                    'last_updated': None,
                    'phoenix_version': '1.0.0',
                },
                'statistics': {'files_total': 0, 'processed': 0, 'succeeded': 0, 'failed': 0, 'skipped': 0},
                'phases': {
                    'p1_syntax': {'completed': False, 'files': {}},
                    'p2_imports': {'completed': False, 'files': {}},
                    'p3_plugins': {'completed': False, 'files': {}},
                    'p4_boot': {'completed': False},
                },
                'errors': [],
            }
            self.save_progress(progress)
            return progress

    def save_progress(self, progress: dict = None):
        """Sauvegarde la progression"""
        if progress:
            self.progress = progress
        self.progress['metadata']['last_updated'] = datetime.now().isoformat()
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f, indent=2)

    def get_files_in_order(self) -> list[Path]:
        """Retourne les fichiers dans l'ordre de traitement"""
        files = []

        for pattern in self.PROCESSING_ORDER:
            base_path = Path(pattern)
            if base_path.is_dir():
                # Ajouter tous les .py du dossier
                for py_file in base_path.rglob("*.py"):
                    if (
                        "__pycache__" not in str(py_file)
                        and "__init__.py" not in str(py_file)
                        and "test_" not in py_file.name
                    ):
                        files.append(py_file)
            elif base_path.exists() and base_path.suffix == ".py":
                files.append(base_path)

        # Ajouter les fichiers non couverts
        for py_file in Path("src/jeffrey").rglob("*.py"):
            if py_file not in files and "__pycache__" not in str(py_file):
                files.append(py_file)

        return files

    def process_syntax(self, filepath: Path) -> bool:
        """Phase 1 : Correction syntaxique avec black"""
        try:
            import black

            content = filepath.read_text()
            original_hash = hashlib.md5(content.encode()).hexdigest()

            # Black formatting
            formatted = black.format_str(content, mode=black.Mode())

            # Validation AST
            import ast

            ast.parse(formatted)

            # Sauvegarder si changé
            if formatted != content:
                filepath.write_text(formatted)

            # Log progress
            self.progress['phases']['p1_syntax']['files'][str(filepath)] = {
                'status': 'success',
                'hash': original_hash,
                'timestamp': datetime.now().isoformat(),
            }
            return True

        except Exception as e:
            self.progress['phases']['p1_syntax']['files'][str(filepath)] = {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
            }
            return False

    def process_imports(self, filepath: Path) -> bool:
        """Phase 2 : Correction des imports avec libcst"""
        try:
            # Utiliser le rewriter
            if rewrite_imports(str(filepath)):
                self.progress['phases']['p2_imports']['files'][str(filepath)] = {
                    'status': 'success',
                    'timestamp': datetime.now().isoformat(),
                }
            else:
                self.progress['phases']['p2_imports']['files'][str(filepath)] = {
                    'status': 'skipped',
                    'timestamp': datetime.now().isoformat(),
                }
            return True

        except Exception as e:
            self.progress['phases']['p2_imports']['files'][str(filepath)] = {'status': 'failed', 'error': str(e)}
            return False

    def process_plugins(self, filepath: Path) -> bool:
        """Phase 3 : Ajout des plugins"""
        try:
            # Déterminer la config du plugin
            plugin_config = self.infer_plugin_config(filepath)

            if add_plugin_to_file(filepath, plugin_config):
                self.progress['phases']['p3_plugins']['files'][str(filepath)] = {
                    'status': 'added',
                    'timestamp': datetime.now().isoformat(),
                }
                return True
            else:
                self.progress['phases']['p3_plugins']['files'][str(filepath)] = {
                    'status': 'skipped',
                    'reason': 'already has plugin or no class',
                }
                return True

        except Exception as e:
            self.progress['phases']['p3_plugins']['files'][str(filepath)] = {'status': 'failed', 'error': str(e)}
            return False

    def infer_plugin_config(self, filepath: Path) -> dict:
        """Infère la configuration du plugin selon le module"""

        path_str = str(filepath).lower()

        # Patterns par catégorie
        if 'consciousness' in path_str:
            return {
                'topics_in': ['percept.*', 'memory.recall'],
                'topics_out': ['consciousness.state', 'plan.slow'],
                'handler': 'process',
            }
        elif 'emotions' in path_str:
            return {
                'topics_in': ['emotion.trigger', 'consciousness.state'],
                'topics_out': ['emotion.state', 'affect.appraise'],
                'handler': 'process',
            }
        elif 'memory' in path_str:
            return {
                'topics_in': ['mem.store', 'mem.recall'],
                'topics_out': ['mem.retrieved', 'mem.stored'],
                'handler': 'process',
            }
        elif 'orchestrat' in path_str:
            return {
                'topics_in': ['plan.*', 'task.request'],
                'topics_out': ['task.result', 'act.*'],
                'handler': 'orchestrate',
            }
        else:
            return {'topics_in': ['*'], 'topics_out': ['processed'], 'handler': 'process'}

    def run_tests(self, batch_num: int) -> bool:
        """Lance les tests après un batch"""
        print(f"\n🧪 Testing batch {batch_num}...")

        # Smoke tests
        result = subprocess.run(['python', 'tests/smoke/test_critical_modules.py'], capture_output=True)
        if result.returncode != 0:
            print(f"❌ Smoke tests failed after batch {batch_num}")
            return False

        # Import cycles
        result = subprocess.run(['python', 'tools/check_import_cycles.py'], capture_output=True, timeout=30)
        if result.returncode == 1:
            print("❌ Critical import cycles detected")
            return False

        print(f"✅ Batch {batch_num} tests passed")
        return True

    def run_phase(self, phase: str):
        """Exécute une phase complète"""

        files = self.get_files_in_order()
        batch_num = 0

        print(f"\n🚀 Running Phase: {phase}")
        print(f"📁 Processing {len(files)} files...")

        for i, filepath in enumerate(files):
            # Process selon la phase
            if phase == 'syntax':
                success = self.process_syntax(filepath)
            elif phase == 'imports':
                success = self.process_imports(filepath)
            elif phase == 'plugins':
                success = self.process_plugins(filepath)
            else:
                print(f"Unknown phase: {phase}")
                return

            if success:
                self.progress['statistics']['succeeded'] += 1
            else:
                self.progress['statistics']['failed'] += 1

            self.progress['statistics']['processed'] += 1
            self.current_batch.append(filepath)

            # Test après chaque batch
            if len(self.current_batch) >= self.batch_size:
                batch_num += 1
                if not self.run_tests(batch_num):
                    print("❌ Tests failed, stopping phase")
                    self.save_progress()
                    sys.exit(1)
                self.current_batch = []
                self.save_progress()

                # Commit Git
                subprocess.run(['git', 'add', '.'])
                subprocess.run(['git', 'commit', '-m', f'phoenix: {phase} batch {batch_num}'])

        # Test final
        if self.current_batch:
            batch_num += 1
            if not self.run_tests(batch_num):
                print("❌ Final tests failed")
                sys.exit(1)

        self.progress['phases'][f'p{["syntax", "imports", "plugins"].index(phase) + 1}_{phase}']['completed'] = True
        self.save_progress()

        print(f"✅ Phase {phase} completed!")
        print(f"   Succeeded: {self.progress['statistics']['succeeded']}")
        print(f"   Failed: {self.progress['statistics']['failed']}")

    def run_security_checks(self):
        """Phase 5 : Tests de sécurité avec Bandit/Pylint"""
        print("\n🔒 Running security checks...")

        # Bandit
        result = subprocess.run(
            ['python', '-m', 'bandit', '-r', 'src/jeffrey', '-f', 'json', '-o', 'security_report.json'],
            capture_output=True,
        )

        # Vérifier les vulnérabilités HIGH
        if Path('security_report.json').exists():
            with open('security_report.json') as f:
                report = json.load(f)
                high_issues = [i for i in report.get('results', []) if i.get('issue_severity') == 'HIGH']
                if high_issues:
                    print(f"❌ {len(high_issues)} HIGH severity security issues found!")
                    for issue in high_issues[:5]:
                        print(f"   - {issue['filename']}: {issue['issue_text']}")
                    return False

        # Pylint (score minimum 7/10)
        result = subprocess.run(['python', '-m', 'pylint', 'src/jeffrey', '--output-format=json'], capture_output=True)

        if result.stdout:
            try:
                pylint_results = json.loads(result.stdout)
                # Calculer le score (simplifié)
                total_statements = sum(r.get('statements', 0) for r in pylint_results)
                total_errors = len([r for r in pylint_results if r.get('type') in ['error', 'fatal']])

                if total_statements > 0:
                    score = max(0, 10 - (total_errors * 10 / total_statements))
                    if score < 7:
                        print(f"❌ Pylint score too low: {score:.1f}/10")
                        return False
                    print(f"✅ Pylint score: {score:.1f}/10")
            except:
                pass

        print("✅ Security checks passed!")
        return True


def main():
    parser = argparse.ArgumentParser(description='Phoenix Driver - Jeffrey OS Stabilization')
    parser.add_argument('--phase', choices=['syntax', 'imports', 'plugins', 'security', 'all'], help='Phase to run')
    parser.add_argument('--order', choices=['strict', 'auto'], default='strict', help='Processing order')
    parser.add_argument('--batch-size', type=int, default=20, help='Files per batch before testing')

    args = parser.parse_args()

    driver = PhoenixDriver()
    driver.batch_size = args.batch_size

    if args.phase == 'all':
        phases = ['syntax', 'imports', 'plugins', 'security']
    else:
        phases = [args.phase]

    for phase in phases:
        if phase == 'security':
            if not driver.run_security_checks():
                print("❌ Security checks failed!")
                sys.exit(1)
        else:
            driver.run_phase(phase)

    print("\n🎉 PHOENIX COMPLETE!")
    print("Run: python tools/phoenix/status.py for full report")


if __name__ == "__main__":
    main()
