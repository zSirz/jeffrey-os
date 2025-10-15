#!/usr/bin/env python3
"""
Jeffrey OS - P0 Smoke Test Ultimate
Validation complète des modules P0 consolidés
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path


class P0SmokeTest:
    """Test de validation P0 avec flags mapping"""

    def __init__(self):
        self.base_dir = Path.cwd()
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "flags": {},
            "status": "IN_PROGRESS",
        }

        # Flags mapping as per GPT
        self.flag_mapping = {
            "self_awareness_tracker": "JEFFREY_ENABLE_AWARENESS",
            "cognitive_synthesis": "JEFFREY_ENABLE_SYNTHESIS",
            "cortex_memoriel": "JEFFREY_ENABLE_CORTEX",
            "dream_engine": "JEFFREY_ENABLE_DREAM",
        }

    def log(self, message, level="info"):
        """Simple logging"""
        colors = {
            "success": "\033[92m",
            "warning": "\033[93m",
            "error": "\033[91m",
            "info": "\033[94m",
            "endc": "\033[0m",
        }

        if sys.stdout.isatty():
            color = colors.get(level, colors["info"])
            print(f"{color}{message}{colors['endc']}")
        else:
            print(message)

    def check_integrity_lock(self):
        """Vérifier p0_integrity.lock"""
        self.log("\n" + "=" * 50)
        self.log("🔒 Checking P0 Integrity Lock")

        lock_file = self.base_dir / "p0_integrity.lock"

        if not lock_file.exists():
            self.log("   ❌ p0_integrity.lock not found", "error")
            self.results["tests"]["integrity"] = "MISSING"
            return False

        with open(lock_file) as f:
            lock_data = json.load(f)

        modules_count = len(lock_data.get("modules", {}))
        self.log(f"   ✅ Found {modules_count} modules in lock", "success")

        # Vérifier que les hashes sont 64 chars
        for module, info in lock_data.get("modules", {}).items():
            hash_len = len(info.get("hash", ""))
            if hash_len != 64:
                self.log(f"   ⚠️ {module}: hash length {hash_len} (expected 64)", "warning")
            else:
                self.log(f"   ✅ {module}: hash OK ({info['hash'][:8]}...)", "success")

        self.results["tests"]["integrity"] = "PASS"
        return True

    async def test_module_imports(self):
        """Test import de chaque module P0"""
        self.log("\n" + "=" * 50)
        self.log("🧪 Testing Module Imports")

        modules = [
            ("dream_engine", "src.jeffrey.core.dreaming.dream_engine"),
            ("self_awareness_tracker", "src.jeffrey.core.consciousness.self_awareness_tracker"),
            ("cognitive_synthesis", "src.jeffrey.core.memory.cognitive_synthesis"),
            ("cortex_memoriel", "src.jeffrey.core.memory.cortex_memoriel"),
        ]

        for module_name, import_path in modules:
            try:
                # Test import avec les chemins corrects
                exec(f"from {import_path} import *")
                self.log(f"   ✅ {module_name}: import OK", "success")
                self.results["tests"][module_name] = "PASS"
            except ImportError as e:
                self.log(f"   ❌ {module_name}: import failed - {e}", "error")
                self.results["tests"][module_name] = "FAIL"
            except Exception as e:
                self.log(f"   ❌ {module_name}: unexpected error - {e}", "error")
                self.results["tests"][module_name] = "ERROR"

    def test_environment_flags(self):
        """Test des flags d'environnement"""
        self.log("\n" + "=" * 50)
        self.log("🎩 Testing Environment Flags")

        for module, flag in self.flag_mapping.items():
            value = os.environ.get(flag, "0")
            enabled = value == "1"

            if enabled:
                self.log(f"   ✅ {flag}: ENABLED", "success")
            else:
                self.log(f"   ⚠️ {flag}: disabled (set to '1' to enable)", "warning")

            self.results["flags"][flag] = enabled

    async def test_symbiosis_basic(self):
        """Test basique de symbiosis"""
        self.log("\n" + "=" * 50)
        self.log("🌐 Testing Symbiosis (Basic)")

        try:
            from jeffrey.core.symbiosis.symbiosis_engine import SymbiosisEngine

            symbiosis = SymbiosisEngine()

            # Test simple compatibility check
            score = await asyncio.wait_for(symbiosis.check_compat("dream_engine", "cortex_memoriel"), timeout=3.0)

            if score > 0.7:
                self.log(f"   ✅ Symbiosis compatibility: {score:.2f}", "success")
                self.results["tests"]["symbiosis"] = "PASS"
            else:
                self.log(f"   ⚠️ Low symbiosis score: {score:.2f}", "warning")
                self.results["tests"]["symbiosis"] = "LOW"

        except ImportError:
            self.log("   🔍 Symbiosis module not available", "info")
            self.results["tests"]["symbiosis"] = "SKIP"
        except TimeoutError:
            self.log("   ⏱️ Symbiosis check timeout", "warning")
            self.results["tests"]["symbiosis"] = "TIMEOUT"
        except Exception as e:
            self.log(f"   ❌ Symbiosis test failed: {e}", "error")
            self.results["tests"]["symbiosis"] = "ERROR"

    def check_stub_count(self):
        """Vérifier le nombre de stubs"""
        self.log("\n" + "=" * 50)
        self.log("📦 Checking Stub Files")

        stub_dir = self.base_dir / "src/jeffrey/stubs"

        if not stub_dir.exists():
            self.log("   🔍 Stubs directory not found", "info")
            self.results["tests"]["stubs"] = "NO_DIR"
            return

        stub_files = list(stub_dir.glob("*_stub.py"))
        stub_count = len(stub_files)

        self.log(f"   Found {stub_count} stub files", "info")

        # Show top 5 stubs
        for stub in stub_files[:5]:
            size = stub.stat().st_size
            self.log(f"      - {stub.stem}: {size} bytes", "info")

        if stub_count > 5:
            self.log(f"      ... and {stub_count - 5} more", "info")

        self.results["tests"]["stub_count"] = stub_count

    def generate_summary(self):
        """Générer le résumé final"""
        self.log("\n" + "=" * 50)
        self.log("📊 SMOKE TEST SUMMARY")
        self.log("=" * 50)

        # Count results
        passed = sum(1 for v in self.results["tests"].values() if v == "PASS")
        failed = sum(1 for v in self.results["tests"].values() if v in ["FAIL", "ERROR"])
        skipped = sum(1 for v in self.results["tests"].values() if v == "SKIP")

        # Overall status
        if failed > 0:
            self.results["status"] = "FAILED"
            self.log(f"\n❌ FAILED: {failed} tests failed", "error")
        elif passed == len(self.results["tests"]):
            self.results["status"] = "SUCCESS"
            self.log(f"\n✅ SUCCESS: All {passed} tests passed!", "success")
        else:
            self.results["status"] = "PARTIAL"
            self.log(f"\n⚠️ PARTIAL: {passed} passed, {skipped} skipped", "warning")

        # Details
        self.log("\nDetails:")
        self.log(f"   Passed: {passed}")
        self.log(f"   Failed: {failed}")
        self.log(f"   Skipped: {skipped}")

        # Critical modules status
        self.log("\nP0 Modules:")
        for module in [
            "dream_engine",
            "self_awareness_tracker",
            "cognitive_synthesis",
            "cortex_memoriel",
        ]:
            status = self.results["tests"].get(module, "UNKNOWN")
            if status == "PASS":
                self.log(f"   ✅ {module}", "success")
            else:
                self.log(f"   ❌ {module}: {status}", "error")

        # Save results
        report_file = self.base_dir / "reports" / f"smoke_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_file.parent.mkdir(exist_ok=True)

        with open(report_file, "w") as f:
            json.dump(self.results, f, indent=2)

        self.log(f"\n📄 Report saved to: {report_file}")

        return self.results["status"] == "SUCCESS"

    async def run(self):
        """Exécuter tous les tests"""
        self.log("🚀 Jeffrey OS - P0 Smoke Test Ultimate")
        self.log("=" * 50)

        try:
            # 1. Check integrity lock
            self.check_integrity_lock()

            # 2. Test imports
            await self.test_module_imports()

            # 3. Test environment flags
            self.test_environment_flags()

            # 4. Test symbiosis
            await self.test_symbiosis_basic()

            # 5. Check stubs
            self.check_stub_count()

            # Generate summary
            success = self.generate_summary()

            # Exit code
            sys.exit(0 if success else 1)

        except Exception as e:
            self.log(f"\n❌ Smoke test crashed: {e}", "error")
            sys.exit(2)


def main():
    """Main entry point"""
    tester = P0SmokeTest()
    asyncio.run(tester.run())


if __name__ == "__main__":
    main()
