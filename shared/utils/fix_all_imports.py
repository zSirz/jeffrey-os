#!/usr/bin/env python3
"""
Correction complète des imports avec détection avancée
Intègre pylint pour détection et isort pour organisation
"""

import json
import logging
import re
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImportFixer:
    """Corrige tous les imports dans jeffrey_platform"""

    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.platform_path = self.base_path / "jeffrey_platform"
        self.fixes_applied = 0
        self.files_processed = 0
        self.issues_found = []

    def analyze_with_pylint(self, file_path: Path) -> list:
        """Utilise pylint pour détecter les problèmes d'import"""
        try:
            result = subprocess.run(
                [
                    "pylint",
                    "--output-format=json",
                    "--disable=all",
                    "--enable=import-error,no-name-in-module",
                    str(file_path),
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.stdout:
                issues = json.loads(result.stdout)
                import_issues = [i for i in issues if 'import' in i.get('message', '').lower()]
                return import_issues
        except (subprocess.TimeoutExpired, subprocess.SubprocessError, json.JSONDecodeError):
            pass
        except FileNotFoundError:
            logger.debug("pylint non disponible")
        return []

    def fix_imports_in_file(self, file_path: Path) -> bool:
        """Corrige les imports dans un fichier Python"""
        if not file_path.suffix == '.py' or '__pycache__' in str(file_path):
            return False

        try:
            content = file_path.read_text()
            original = content

            # Patterns de remplacement complets
            replacements = [
                # Imports relatifs vers absolus
                (r'from \.\.modules import', 'from jeffrey_platform.os.orchestrator.modules import'),
                (r'from \.modules import', 'from jeffrey_platform.os.orchestrator.modules import'),
                (r'from modules import', 'from jeffrey_platform.os.orchestrator.modules import'),
                # Orchestration
                (r'from orchestration import', 'from jeffrey_platform.os.orchestrator import'),
                (r'import orchestration\b', 'import jeffrey_platform.os.orchestrator'),
                (r'from orchestrator import', 'from jeffrey_platform.os.orchestrator.modules.orchestrator import'),
                # Context
                (r'from \.\.context import', 'from jeffrey_platform.os.orchestrator.context import'),
                (r'from \.context import', 'from jeffrey_platform.os.orchestrator.context import'),
                (r'from context import', 'from jeffrey_platform.os.orchestrator.context import'),
                # Security
                (r'from \.\.security import', 'from jeffrey_platform.os.orchestrator.security import'),
                (r'from \.security import', 'from jeffrey_platform.os.orchestrator.security import'),
                (r'from security import', 'from jeffrey_platform.os.orchestrator.security import'),
                # Bridge
                (r'from \.\.bridge import', 'from jeffrey_platform.os.orchestrator.bridge import'),
                (r'from \.bridge import', 'from jeffrey_platform.os.orchestrator.bridge import'),
                (r'from bridge import', 'from jeffrey_platform.os.orchestrator.bridge import'),
                # Core
                (r'from \.\.core import', 'from jeffrey_platform.os.orchestrator.core import'),
                (r'from \.core import', 'from jeffrey_platform.os.orchestrator.core import'),
                (r'from core import', 'from jeffrey_platform.os.orchestrator.core import'),
            ]

            for pattern, replacement in replacements:
                content = re.sub(pattern, replacement, content)

            # Détecter les imports circulaires potentiels
            if 'from jeffrey_platform' in content:
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if 'from jeffrey_platform' in line and 'import' in line:
                        # Vérifier si c'est un import du même module
                        if str(file_path).replace('/', '.').replace('.py', '') in line:
                            logger.warning(f"Import circulaire potentiel détecté: {file_path}:{i + 1}")
                            self.issues_found.append(f"Circular import: {file_path}:{i + 1}")

            if content != original:
                file_path.write_text(content)
                self.fixes_applied += 1
                return True

        except Exception as e:
            logger.error(f"Erreur dans {file_path}: {e}")
            self.issues_found.append(f"Error processing {file_path}: {e}")

        return False

    def fix_all(self):
        """Corrige tous les fichiers Python dans jeffrey_platform"""
        if not self.platform_path.exists():
            logger.error(f"{self.platform_path} n'existe pas")
            return

        print(f"🔍 Analyse des imports dans {self.platform_path}...")

        # Parcourir tous les fichiers Python
        py_files = list(self.platform_path.rglob("*.py"))
        total_files = len(py_files)

        for i, py_file in enumerate(py_files, 1):
            # Progress
            if i % 10 == 0:
                print(f"  Progress: {i}/{total_files} fichiers...")

            self.files_processed += 1

            # Analyser avec pylint si disponible
            pylint_issues = self.analyze_with_pylint(py_file)
            if pylint_issues:
                logger.info(f"  {py_file.name}: {len(pylint_issues)} issues pylint")

            # Corriger
            if self.fix_imports_in_file(py_file):
                print(f"  ✅ Corrigé: {py_file.relative_to(self.base_path)}")

        # Tenter isort pour organiser
        try:
            print("\n🎨 Organisation des imports avec isort...")
            result = subprocess.run(
                ["isort", str(self.platform_path), "--profile", "black", "--quiet"], capture_output=True, timeout=30
            )
            if result.returncode == 0:
                print("  ✅ Imports organisés avec isort")
        except FileNotFoundError:
            print("  ⚠️ isort non disponible (pip install isort)")
        except subprocess.TimeoutExpired:
            print("  ⚠️ isort timeout")

        # Rapport
        print("\n📊 RAPPORT:")
        print(f"  Fichiers traités: {self.files_processed}")
        print(f"  Corrections appliquées: {self.fixes_applied}")
        print(f"  Issues trouvées: {len(self.issues_found)}")

        if self.issues_found:
            print("\n⚠️ Issues à vérifier manuellement:")
            for issue in self.issues_found[:5]:  # Limiter à 5
                print(f"  - {issue}")


if __name__ == "__main__":
    fixer = ImportFixer()
    fixer.fix_all()
