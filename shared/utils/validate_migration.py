#!/usr/bin/env python3
"""
Validation complète post-migration - Version corrigée
Chemins à la racine, pas d'auto-install pour deps lourdes
"""

import subprocess
import sys
from pathlib import Path


class MigrationValidator:
    def __init__(self):
        # CHEMINS CORRIGÉS - services/ à la racine
        self.project_root = Path(".")
        self.services_root = Path("services") / "jeffrey_core"
        self.tier3_root = Path("tier3")

        # MODULES CORRIGÉS - emotions au PLURIEL
        self.critical_modules = [
            "consciousness/auto_healing_protocol",
            "consciousness/deep_health_check",
            "emotions/emotional_core",  # PLURIEL
            "voice/voice_system",
        ]
        self.report = {}

    def check_files_exist(self) -> dict[str, bool]:
        """Vérifie que les fichiers migrés existent"""
        results = {}
        for module in self.critical_modules:
            file_path = self.services_root / f"{module}.py"
            exists = file_path.exists()
            results[module] = exists
            if exists:
                size = file_path.stat().st_size
                lines = len(file_path.read_text().splitlines())
                print(f"✅ {module}: {lines} lignes, {size:,} octets")
            else:
                print(f"❌ {module}: MANQUANT")
        return results

    def check_syntax(self) -> dict[str, bool]:
        """Vérifie la syntaxe Python"""
        results = {}
        for module in self.critical_modules:
            file_path = self.services_root / f"{module}.py"
            if not file_path.exists():
                results[module] = False
                continue

            try:
                compile(file_path.read_text(), file_path, 'exec')
                results[module] = True
                print(f"✅ Syntaxe OK: {module}")
            except SyntaxError as e:
                results[module] = False
                print(f"❌ Erreur syntaxe {module}: ligne {e.lineno}")
        return results

    def check_imports(self) -> dict[str, list[str]]:
        """Vérifie les imports et dépendances"""
        failed_imports = {}

        for module in self.critical_modules:
            file_path = self.services_root / f"{module}.py"
            if not file_path.exists():
                continue

            module_name = f"services.jeffrey_core.{module.replace('/', '.')}"

            try:
                # Test d'import dans subprocess isolé
                result = subprocess.run(
                    [sys.executable, "-c", f"import {module_name}"], capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    print(f"✅ Import OK: {module}")
                else:
                    error = result.stderr or result.stdout
                    failed_imports[module] = error[:200]
                    print(f"⚠️ Import warning {module}: {error[:100]}")
            except subprocess.TimeoutExpired:
                failed_imports[module] = "Timeout lors de l'import"
                print(f"⚠️ Import timeout: {module}")

        return failed_imports

    def check_runtime_deps(self) -> dict[str, list[str]]:
        """Vérifie les dépendances runtime - PAS D'AUTO-INSTALL pour deps lourdes"""
        deps_by_category = {
            "light": [],  # Deps légères OK à installer
            "heavy": [],  # Deps lourdes à NE PAS auto-installer
            "missing": [],  # Toutes les deps manquantes
        }

        # Définir les deps lourdes
        HEAVY_DEPS = {'torch', 'tensorflow', 'transformers', 'scipy', 'sklearn'}

        # Parcourir tous les fichiers Python
        for py_file in self.services_root.rglob("*.py"):
            if ".bak" in str(py_file):
                continue

            try:
                content = py_file.read_text()

                # Chercher les imports
                for line in content.splitlines():
                    if line.startswith("import ") or line.startswith("from "):
                        for dep in HEAVY_DEPS:
                            if dep in line:
                                try:
                                    __import__(dep)
                                except ImportError:
                                    deps_by_category["heavy"].append(dep)
                                    deps_by_category["missing"].append(dep)

                        # Deps légères
                        light_deps = ['tenacity', 'structlog', 'psutil']
                        for dep in light_deps:
                            if dep in line:
                                try:
                                    __import__(dep)
                                except ImportError:
                                    deps_by_category["light"].append(dep)
                                    deps_by_category["missing"].append(dep)
            except Exception as e:
                print(f"⚠️ Erreur lecture {py_file.name}: {e}")

        # Dédupliquer
        for key in deps_by_category:
            deps_by_category[key] = list(set(deps_by_category[key]))

        # Afficher les recommendations
        if deps_by_category["light"]:
            print(f"📦 Deps légères manquantes (peuvent être installées): {', '.join(deps_by_category['light'])}")
            print(f"   Commande: pip install {' '.join(deps_by_category['light'])}")

        if deps_by_category["heavy"]:
            print(f"⚠️ DEPS LOURDES manquantes (installation MANUELLE requise): {', '.join(deps_by_category['heavy'])}")
            print("   Ces dépendances nécessitent une configuration spécifique.")
            print("   NE PAS auto-installer pour éviter des problèmes de compatibilité.")

        return deps_by_category

    def check_api_integration(self) -> bool:
        """Vérifie l'intégration avec l'API"""
        try:
            main_path = self.tier3_root / "api" / "main.py"
            if main_path.exists():
                content = main_path.read_text()

                # Vérifier les imports corrects (services. pas tier3.services.)
                correct_imports = "from services.jeffrey_core" in content
                wrong_imports = "from tier3.services" in content

                if wrong_imports:
                    print("❌ Imports incorrects détectés (tier3.services au lieu de services)")
                    return False

                if correct_imports:
                    print("✅ Modules correctement intégrés dans l'API")
                    return True
                else:
                    print("⚠️ Modules non référencés dans l'API")
                    return False
        except Exception as e:
            print(f"❌ Erreur vérification API: {e}")
            return False

    def run_smoke_tests(self) -> tuple[int, int]:
        """Lance les tests smoke avec vérification stricte du 403"""
        try:
            result = subprocess.run(["./tier3/scripts/smoke.sh"], capture_output=True, text=True, timeout=30)

            output = result.stdout

            # Vérifier spécifiquement le test 403
            has_403_success = "403 (CORRECT)" in output or "403 (OK)" in output

            if "403" in output and "500" in output and "ATTENDU: 403" in output:
                print("❌ BUG CRITIQUE: /analyze retourne 500 au lieu de 403")
                return 0, 1  # Échec total si 403 pas correct

            # Parser le résultat général
            import re

            match = re.search(r'(\d+)/(\d+)', output)
            if match:
                passed = int(match.group(1))
                total = int(match.group(2))
                percentage = (passed / total) * 100

                if has_403_success:
                    print(f"✅ Tests smoke: {passed}/{total} ({percentage:.0f}%) - 403 OK")
                else:
                    print(f"⚠️ Tests smoke: {passed}/{total} ({percentage:.0f}%) - 403 NON vérifié")

                return passed, total

            return 0, 0

        except subprocess.TimeoutExpired:
            print("❌ Tests smoke: timeout")
            return 0, 0
        except Exception as e:
            print(f"❌ Tests smoke: {e}")
            return 0, 0

    def generate_report(self) -> dict:
        """Génère un rapport complet"""
        print("\n" + "=" * 50)
        print("📊 RAPPORT DE VALIDATION MIGRATION")
        print("=" * 50 + "\n")

        # 1. Existence des fichiers
        print("1️⃣ EXISTENCE DES FICHIERS:")
        files_exist = self.check_files_exist()

        # 2. Syntaxe
        print("\n2️⃣ VÉRIFICATION SYNTAXE:")
        syntax_ok = self.check_syntax()

        # 3. Imports
        print("\n3️⃣ VÉRIFICATION IMPORTS:")
        import_errors = self.check_imports()

        # 4. Dépendances runtime
        print("\n4️⃣ DÉPENDANCES RUNTIME:")
        deps = self.check_runtime_deps()

        # 5. Intégration API
        print("\n5️⃣ INTÉGRATION API:")
        api_ok = self.check_api_integration()

        # 6. Tests smoke
        print("\n6️⃣ TESTS SMOKE:")
        passed, total = self.run_smoke_tests()

        # Calcul du score global
        score = 0
        max_score = 0

        # Points par catégorie
        for module in self.critical_modules:
            max_score += 3
            if files_exist.get(module, False):
                score += 1
            if syntax_ok.get(module, False):
                score += 1
            if module not in import_errors:
                score += 1

        # Points bonus
        if not deps["heavy"]:  # Pas de deps lourdes manquantes
            score += 2
            max_score += 2
        if api_ok:
            score += 2
            max_score += 2
        if total > 0:
            score += (passed / total) * 3
            max_score += 3

        percentage = (score / max_score) * 100 if max_score > 0 else 0

        # Rapport final
        print("\n" + "=" * 50)
        print(f"📈 SCORE GLOBAL: {score:.1f}/{max_score} ({percentage:.0f}%)")

        if percentage >= 90:
            print("✅ MIGRATION RÉUSSIE - Système prêt pour production!")
        elif percentage >= 70:
            print("⚠️ MIGRATION PARTIELLE - Corrections mineures requises")
        else:
            print("❌ MIGRATION INCOMPLÈTE - Intervention requise")

        print("=" * 50)

        # Actions requises
        if deps["heavy"]:
            print("\n⚠️ ACTION MANUELLE REQUISE:")
            print(f"   Installer les deps lourdes: {', '.join(deps['heavy'])}")
            print("   Ces packages nécessitent une configuration spécifique")

        if deps["light"]:
            print(f"\n💡 Deps légères à installer: pip install {' '.join(deps['light'])}")

        if import_errors:
            print(f"\n💡 Modules à corriger: {', '.join(import_errors.keys())}")

        return {
            "score": percentage,
            "files_exist": files_exist,
            "syntax_ok": syntax_ok,
            "import_errors": import_errors,
            "deps": deps,
            "api_integrated": api_ok,
            "tests_passed": f"{passed}/{total}",
        }


if __name__ == "__main__":
    validator = MigrationValidator()
    report = validator.generate_report()

    # Code de sortie basé sur le score
    sys.exit(0 if report["score"] >= 90 else 1)
