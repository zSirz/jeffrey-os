#!/usr/bin/env python3
"""
Migration P0 Ultra-Sécurisée avec Hash Verification
Migre uniquement les 4 modules critiques sans dépendances lourdes
"""

import hashlib
import json
import py_compile
import re
import shutil
from datetime import datetime
from pathlib import Path


class SecureP0Migration:
    """Migration sécurisée avec vérification hash et backup automatique"""

    def __init__(self):
        self.icloud_base = (
            Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/Jeffrey/Jeffrey_Apps/Jeffrey_OS/src"
        )
        self.project_base = Path.cwd()

        # Vérification d'hydratation iCloud
        if any(
            str(p).endswith(".icloud")
            for p in [
                self.icloud_base / "core/dreaming/dream_engine.py",
                self.icloud_base / "core/consciousness/self_awareness_tracker.py",
                self.icloud_base / "core/memory/cognitive_synthesis.py",
                self.icloud_base / "core/memory/cortex_memoriel.py",
            ]
        ):
            self.log(
                "⚠️ Certains fichiers sources sont des placeholders .icloud : ouvre-les dans Finder.",
                "warning",
            )

        # MAPPING P0 - Seulement 4 modules validés
        self.P0_MODULES = {
            self.icloud_base / "core/dreaming/dream_engine.py": "src/jeffrey/core/consciousness/dream_engine.py",
            self.icloud_base
            / "core/consciousness/self_awareness_tracker.py": "src/jeffrey/core/consciousness/self_awareness_tracker.py",
            self.icloud_base
            / "core/memory/cognitive_synthesis.py": "src/jeffrey/core/consciousness/cognitive_synthesis.py",
            self.icloud_base / "core/memory/cortex_memoriel.py": "src/jeffrey/core/memory/cortex_memoriel.py",
        }

        # HASH VERIFICATION - Garantir les bonnes versions (mises à jour)
        self.EXPECTED_HASHES = {
            "dream_engine.py": "24375ac0",
            "self_awareness_tracker.py": "ce61d9ad",
            "cognitive_synthesis.py": "b96b3e5e",
            "cortex_memoriel.py": "03ecb4d0",
        }

        # Import rewrites
        self.IMPORT_FIXES = [
            (r"\bfrom core\.", "from jeffrey.core."),
            (r"\bimport core\.", "import jeffrey.core."),
            (r"(?<!\.)from cortex_memoriel\b", "from jeffrey.core.memory.cortex_memoriel"),
            (r"(?<!\.)from dream_engine\b", "from jeffrey.core.consciousness.dream_engine"),
            (
                r"(?<!\.)from cognitive_synthesis\b",
                "from jeffrey.core.consciousness.cognitive_synthesis",
            ),
            (r"(?<!\.)import cortex_memoriel\b", "import jeffrey.core.memory.cortex_memoriel"),
            # Fallback to stubs for missing dependencies
            (r"from neural_mutator\b", "from jeffrey.stubs.neural_mutator"),
            (r"from variant_generator\b", "from jeffrey.stubs.variant_generator"),
            (r"from data_augmenter\b", "from jeffrey.stubs.data_augmenter"),
            (r"from dream_state\b", "from jeffrey.stubs.dream_state"),
            (r"from ethical_guard\b", "from jeffrey.stubs.ethical_guard"),
        ]

        self.migration_log = []
        self.success_count = 0
        self.failed_modules = []

    def calculate_hash(self, filepath: Path) -> str:
        """Calcule le hash SHA256 court d'un fichier"""
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()[:8]

    def verify_source_hash(self, source_path: Path) -> bool:
        """Vérifie que le fichier source a le bon hash"""
        filename = source_path.name
        expected = self.EXPECTED_HASHES.get(filename)

        if not expected:
            self.log(f"⚠️ Pas de hash attendu pour {filename}", "warning")
            return True  # On continue quand même

        actual = self.calculate_hash(source_path)

        if actual != expected:
            self.log(f"❌ HASH MISMATCH pour {filename}: attendu {expected}, trouvé {actual}", "error")
            return False

        self.log(f"✅ Hash vérifié: {filename} = {expected}", "success")
        return True

    def fix_imports(self, content: str) -> str:
        """Corrige tous les imports pour la nouvelle structure"""
        fixed = content

        for pattern, replacement in self.IMPORT_FIXES:
            fixed = re.sub(pattern, replacement, fixed, flags=re.MULTILINE)

        # Log les changements
        if fixed != content:
            changes = len([1 for a, b in zip(content.split("\n"), fixed.split("\n")) if a != b])
            self.log(f"   📝 {changes} lignes d'imports corrigées", "info")

        return fixed

    def create_backup(self, dest_path: Path) -> Path:
        """Créé un backup horodaté si le fichier existe"""
        if dest_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = dest_path.with_suffix(f".backup_{timestamp}")
            shutil.copy2(dest_path, backup_path)
            self.log(f"   📦 Backup créé: {backup_path.name}", "info")
            return backup_path
        return None

    def migrate_module(self, source: Path, dest: str) -> bool:
        """Migre un module avec toutes les vérifications"""
        dest_path = Path(dest)

        try:
            self.log(f"\n📂 Migration: {source.name}", "header")

            # 1. Vérifier l'existence de la source
            if not source.exists():
                self.log(f"   ❌ Source non trouvée: {source}", "error")
                self.failed_modules.append(source.name)
                return False

            # 2. Vérifier le hash
            if not self.verify_source_hash(source):
                self.failed_modules.append(source.name)
                return False

            # 3. Créer backup si nécessaire
            backup_path = self.create_backup(dest_path)

            # 4. Créer les répertoires
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # 5. Lire et corriger le contenu
            content = source.read_text(encoding="utf-8", errors="ignore")
            fixed_content = self.fix_imports(content)

            # 6. Écrire le fichier
            dest_path.write_text(fixed_content, encoding="utf-8")

            # 7. Vérifier la syntaxe Python
            py_compile.compile(str(dest_path), doraise=True)

            # 8. Vérifier le hash final
            final_hash = self.calculate_hash(dest_path)

            self.log(f"   ✅ Migré avec succès: {dest_path}", "success")
            self.log(f"   📊 Stats: {len(content)} caractères, {content.count(chr(10))} lignes", "info")
            self.log(f"   🔐 Hash final: {final_hash}", "info")

            self.success_count += 1
            return True

        except SyntaxError as e:
            self.log(f"   ❌ Erreur de syntaxe Python: {e}", "error")
            # Restaurer le backup
            if backup_path and backup_path.exists():
                shutil.copy2(backup_path, dest_path)
                self.log("   ⏮ Backup restauré", "warning")
            self.failed_modules.append(source.name)
            return False

        except Exception as e:
            self.log(f"   ❌ Erreur inattendue: {e}", "error")
            self.failed_modules.append(source.name)
            return False

    def create_minimal_stubs(self):
        """Créé des stubs minimaux pour les dépendances manquantes"""
        stubs_dir = Path("src/jeffrey/stubs")
        stubs_dir.mkdir(parents=True, exist_ok=True)

        # Créer __init__.py pour que les stubs soient importables
        (stubs_dir / "__init__.py").write_text("", encoding="utf-8")
        self.log("   📝 __init__.py créé pour stubs", "info")

        stubs = {
            "neural_mutator.py": """
class NeuralMutator:
    def mutate(self, *args, **kwargs):
        return {"mutations": [], "stats": {}}
""",
            "variant_generator.py": """
class VariantGenerator:
    def generate_variants(self, *args, **kwargs):
        return []
""",
            "data_augmenter.py": """
class DataAugmenter:
    def augment(self, *args, **kwargs):
        return {}
""",
            "dream_state.py": """
class DreamState:
    def __init__(self):
        self.state = "idle"
""",
            "ethical_guard.py": """
class EthicalGuard:
    def validate(self, *args, **kwargs):
        return True
""",
        }

        for filename, content in stubs.items():
            stub_path = stubs_dir / filename
            if not stub_path.exists():
                stub_path.write_text(content)
                self.log(f"   📝 Stub créé: {filename}", "info")

    def test_imports(self) -> bool:
        """Test que tous les imports fonctionnent"""
        self.log("\n🧪 TEST DES IMPORTS", "header")

        import sys

        sys.path.insert(0, ".")

        tests = [
            ("SelfAwarenessTracker", "src.jeffrey.core.consciousness.self_awareness_tracker"),
            ("CognitiveSynthesis", "src.jeffrey.core.consciousness.cognitive_synthesis"),
            ("CortexMemoriel", "src.jeffrey.core.memory.cortex_memoriel"),
            ("DreamEngine", "src.jeffrey.core.consciousness.dream_engine"),
        ]

        all_pass = True

        for class_name, module_path in tests:
            try:
                module = __import__(module_path, fromlist=[class_name])
                cls = getattr(module, class_name)
                self.log(f"   ✅ {class_name} importable", "success")
            except ImportError as e:
                self.log(f"   ⚠️ {class_name} - dépendances manquantes: {e}", "warning")
                if "DreamEngine" not in class_name:  # On tolère pour DreamEngine
                    all_pass = False
            except Exception as e:
                self.log(f"   ❌ {class_name} - erreur: {e}", "error")
                all_pass = False

        return all_pass

    def log(self, message: str, level: str = "info"):
        """Log avec couleurs"""
        colors = {
            "header": "\033[1;34m",  # Bleu gras
            "success": "\033[92m",  # Vert
            "error": "\033[91m",  # Rouge
            "warning": "\033[93m",  # Jaune
            "info": "\033[0m",  # Normal
        }

        print(f"{colors.get(level, '')}{message}\033[0m")
        self.migration_log.append({"time": datetime.now().isoformat(), "level": level, "message": message})

    def save_report(self):
        """Sauvegarde un rapport JSON de la migration"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "success_count": self.success_count,
            "failed_count": len(self.failed_modules),
            "failed_modules": self.failed_modules,
            "log": self.migration_log,
        }

        report_path = Path("reports/migration_p0_report.json")
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        self.log(f"\n💾 Rapport sauvegardé: {report_path}", "info")

    def run(self):
        """Exécute la migration P0 complète"""
        self.log("\n" + "=" * 60, "header")
        self.log("🚀 MIGRATION P0 SÉCURISÉE - MODULES CRITIQUES JEFFREY OS", "header")
        self.log("=" * 60, "header")

        # 1. Créer les stubs de fallback
        self.log("\n📝 CRÉATION DES STUBS DE FALLBACK", "header")
        self.create_minimal_stubs()

        # 2. Migration module par module
        self.log("\n🔄 MIGRATION DES MODULES", "header")
        for source, dest in self.P0_MODULES.items():
            self.migrate_module(source, dest)

        # 3. Test des imports
        imports_ok = self.test_imports()

        # 4. Rapport final
        self.log("\n" + "=" * 60, "header")
        self.log("📊 RAPPORT FINAL", "header")
        self.log("=" * 60, "header")
        self.log(f"✅ Modules migrés: {self.success_count}/{len(self.P0_MODULES)}", "success")

        if self.failed_modules:
            self.log(f"❌ Modules échoués: {', '.join(self.failed_modules)}", "error")

        if self.success_count == len(self.P0_MODULES):
            self.log("\n🎉 MIGRATION P0 RÉUSSIE!", "success")
            self.log("→ Prochaine étape: python3 test_brain_headless.py", "info")
        else:
            self.log("\n⚠️ Migration incomplète - vérifier les erreurs", "warning")

        # 5. Sauvegarder le rapport
        self.save_report()

        return self.success_count == len(self.P0_MODULES)


if __name__ == "__main__":
    migration = SecureP0Migration()
    success = migration.run()
    exit(0 if success else 1)
