#!/usr/bin/env python3
"""
Orchestrateur de tests progressifs pour Jeffrey OS
Lance les phases de test dans l'ordre optimal avec validation
"""

import asyncio
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Add scripts directory to path for nats_manager import if needed
sys.path.insert(0, str(Path(__file__).parent))


class SoakTestOrchestrator:
    """Orchestrateur principal des tests de validation"""

    def __init__(self, non_interactive: bool = False):
        self.non_interactive = non_interactive or os.getenv("CI") == "true"
        self.results = {}
        self.start_time = datetime.now()
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / f"orchestrator_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"

    async def run_setup(self) -> bool:
        """Setup initial du projet"""
        print("\n🔧 PROJECT SETUP")
        print("=" * 50)

        # Vérifier si déjà installé (import sans src.*)
        try:
            from jeffrey.core.loops.loop_manager import LoopManager

            print("✅ Project already installed")
            return True
        except ImportError:
            pass

        # Installer en mode éditable
        print("📦 Installing project in editable mode...")

        try:
            result = await self.run_command([sys.executable, "-m", "pip", "install", "-e", ".", "--quiet"], timeout=60)

            if result[0]:
                print("✅ Project installed successfully")
                return True
            else:
                print(f"❌ Installation failed: {result[2]}")
                return False

        except Exception as e:
            print(f"❌ Setup error: {e}")
            return False

    async def run_validation(self) -> tuple[bool, dict]:
        """Lance la validation Phase 3.0"""
        print("\n🔍 PHASE 3.0 VALIDATION")
        print("=" * 50)

        validation_script = Path("scripts/validate_phase30.py")
        if not validation_script.exists():
            print("⚠️  Validation script not found, skipping")
            return True, {"skipped": True}

        try:
            success, stdout, stderr = await self.run_command([sys.executable, str(validation_script)], timeout=60)

            # Parser les résultats
            results = {
                "passed": success and ("PASSED" in stdout or "RÉUSSIE" in stdout),
                "output": stdout,
                "errors": stderr,
            }

            if results["passed"]:
                print("✅ Validation PASSED")
            else:
                print("❌ Validation FAILED")
                if not self.non_interactive:
                    print("\nOutput:", stdout[:500])
                    print("\nErrors:", stderr[:500])

            self.results["validation"] = results
            return results["passed"], results

        except Exception as e:
            print(f"❌ Validation error: {e}")
            return False, {"error": str(e)}

    async def run_progressive_tests(self) -> bool:
        """Lance les tests dans l'ordre optimal"""
        print("\n🚀 PROGRESSIVE LOAD TESTS")
        print("=" * 50)

        # Définir l'ordre des phases
        test_phases = [
            {
                "name": "Quick Baseline",
                "phase": "quick",
                "description": "Établit la baseline de performance",
                "critical": True,
            },
            {
                "name": "Stress Test",
                "phase": "stress",
                "description": "Test de charge nominale avec ML",
                "critical": True,
            },
            {
                "name": "Chaos Engineering",
                "phase": "chaos",
                "description": "Résilience aux pannes",
                "critical": False,
            },
        ]

        all_passed = True

        for i, test in enumerate(test_phases, 1):
            print(f"\n📊 Test {i}/{len(test_phases)}: {test['name']}")
            print(f"   Description: {test['description']}")
            print("-" * 40)

            # Commande pour lancer le test
            cmd = [sys.executable, "scripts/generate_load.py", "--phase", test["phase"]]

            if self.non_interactive:
                cmd.append("--non-interactive")

            # Lancer et attendre
            try:
                success, stdout, stderr = await self.run_command(
                    cmd,
                    timeout=600,  # 10 minutes max par phase
                )

                # Parser les résultats
                phase_passed = success and "PASSED" in stdout

                self.results[test["phase"]] = {
                    "passed": phase_passed,
                    "critical": test["critical"],
                    "output_summary": self.extract_summary(stdout),
                }

                if phase_passed:
                    print(f"   ✅ {test['name']} PASSED")
                else:
                    print(f"   ❌ {test['name']} FAILED")

                    if test["critical"]:
                        all_passed = False

                        if not self.non_interactive:
                            response = input("   Critical test failed. Continue? (y/n): ")
                            if response.lower() != "y":
                                break
                        else:
                            print("   Critical test failed in CI mode, stopping.")
                            break

            except TimeoutError:
                print(f"   ⚠️  {test['name']} timeout")
                self.results[test["phase"]] = {"passed": False, "error": "timeout"}
                if test["critical"]:
                    all_passed = False
                    break

            except Exception as e:
                print(f"   ❌ {test['name']} error: {e}")
                self.results[test["phase"]] = {"passed": False, "error": str(e)}
                if test["critical"]:
                    all_passed = False
                    break

        return all_passed

    async def run_soak_test(self) -> bool:
        """Lance le soak test final si tous les tests passent"""
        print("\n🏁 FINAL SOAK TEST")
        print("=" * 50)

        if not self.non_interactive:
            print("\n⚠️  This will run for 2 hours. Continue? (y/n): ", end="")
            if input().lower() != "y":
                print("Skipping soak test.")
                return True

        print("Starting 2-hour soak test...")
        print("You can monitor progress at http://localhost:8000/metrics")

        cmd = [
            sys.executable,
            "scripts/generate_load.py",
            "--phase",
            "soak",
            "--ml",
            "--monitor",
            "--prometheus-port",
            "8000",
        ]

        if self.non_interactive:
            cmd.append("--non-interactive")

        try:
            # Pour le soak test, on veut voir la progression
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

            # Stream output
            for line in proc.stdout:
                print(line, end="")

                # Sauver les métriques importantes
                if "VALIDATION RESULTS" in line:
                    self.results["soak"] = {"running": True}

            proc.wait()

            self.results["soak"]["passed"] = proc.returncode == 0
            return proc.returncode == 0

        except Exception as e:
            print(f"❌ Soak test error: {e}")
            self.results["soak"] = {"passed": False, "error": str(e)}
            return False

    async def run_command(self, cmd: list[str], timeout: int = 60) -> tuple[bool, str, str]:
        """Execute une commande avec timeout"""
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path.cwd(),  # S'assurer qu'on est dans le bon répertoire
            )

            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)

            return (
                proc.returncode == 0,
                stdout.decode("utf-8", errors="ignore"),
                stderr.decode("utf-8", errors="ignore"),
            )

        except TimeoutError:
            raise
        except Exception as e:
            return False, "", str(e)

    def extract_summary(self, output: str) -> dict:
        """Extrait les métriques clés de la sortie"""
        summary = {}

        for line in output.split("\n"):
            if "P99 Latency:" in line:
                try:
                    summary["p99"] = float(line.split(":")[1].replace("ms", "").strip())
                except:
                    pass
            elif "Symbiosis Score:" in line:
                try:
                    summary["symbiosis"] = float(line.split(":")[1].strip())
                except:
                    pass
            elif "Messages sent:" in line:
                try:
                    summary["messages"] = int(line.split(":")[1].replace(",", "").strip())
                except:
                    pass

        return summary

    def generate_report(self):
        """Génère le rapport final"""
        duration = (datetime.now() - self.start_time).total_seconds() / 60

        print("\n" + "=" * 70)
        print("📋 JEFFREY OS - SOAK TEST ORCHESTRATION REPORT")
        print("=" * 70)

        print(f"\nTotal Duration: {duration:.1f} minutes")
        print(f"Mode: {'CI/CD' if self.non_interactive else 'Interactive'}")

        print("\n📊 TEST RESULTS:")

        all_passed = True

        for phase, result in self.results.items():
            if isinstance(result, dict):
                passed = result.get("passed", False)
                critical = result.get("critical", False)

                status = "✅ PASS" if passed else "❌ FAIL"
                crit = " [CRITICAL]" if critical and not passed else ""

                print(f"\n  {phase.upper()}{crit}:")
                print(f"    Status: {status}")

                if "output_summary" in result:
                    summary = result["output_summary"]
                    if "p99" in summary:
                        print(f"    P99: {summary['p99']:.1f}ms")
                    if "symbiosis" in summary:
                        print(f"    Symbiosis: {summary['symbiosis']:.3f}")
                    if "messages" in summary:
                        print(f"    Messages: {summary['messages']:,}")

                if not passed and critical:
                    all_passed = False

        # Sauver le rapport JSON
        with open(self.log_file, "w") as f:
            json.dump(
                {
                    "timestamp": self.start_time.isoformat(),
                    "duration_minutes": duration,
                    "all_passed": all_passed,
                    "results": self.results,
                },
                f,
                indent=2,
            )

        print(f"\n📄 Detailed report saved to: {self.log_file}")

        print("\n" + "=" * 70)
        if all_passed:
            print("🎉 ALL TESTS PASSED - SYSTEM IS PRODUCTION READY!")
            print("\n✅ Next steps:")
            print("  1. Deploy to staging environment")
            print("  2. Run 24-hour soak test")
            print("  3. Enable production monitoring")
        else:
            print("⚠️  SOME TESTS FAILED - REVIEW AND FIX REQUIRED")
            print("\n❌ Failed components need attention before production")
        print("=" * 70)

        return all_passed


async def main():
    """Point d'entrée principal"""
    import argparse

    parser = argparse.ArgumentParser(description="Jeffrey OS Soak Test Orchestrator")
    parser.add_argument("--non-interactive", action="store_true", help="Run in non-interactive mode for CI/CD")
    parser.add_argument("--skip-setup", action="store_true", help="Skip project setup")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation phase")
    parser.add_argument("--soak-only", action="store_true", help="Run only the 2-hour soak test")

    args = parser.parse_args()

    print("🧪 JEFFREY OS - PRODUCTION VALIDATION ORCHESTRATOR")
    print("=" * 70)
    print(f"Mode: {'CI/CD' if args.non_interactive else 'Interactive'}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    orchestrator = SoakTestOrchestrator(non_interactive=args.non_interactive)

    try:
        # 1. Setup
        if not args.skip_setup:
            if not await orchestrator.run_setup():
                print("\n❌ Setup failed. Cannot continue.")
                return 1

        # 2. Validation
        if not args.skip_validation and not args.soak_only:
            passed, _ = await orchestrator.run_validation()
            if not passed and not args.non_interactive:
                if input("\nValidation failed. Continue anyway? (y/n): ").lower() != "y":
                    orchestrator.generate_report()
                    return 1

        # 3. Progressive tests
        if not args.soak_only:
            if not await orchestrator.run_progressive_tests():
                print("\n⚠️  Progressive tests had failures.")
                if not args.non_interactive:
                    if input("Run soak test anyway? (y/n): ").lower() != "y":
                        orchestrator.generate_report()
                        return 1

        # 4. Soak test (optionnel)
        if args.soak_only or (not args.non_interactive and input("\nRun 2-hour soak test? (y/n): ").lower() == "y"):
            await orchestrator.run_soak_test()

    except KeyboardInterrupt:
        print("\n\n🛑 Orchestration interrupted by user")
    except Exception as e:
        print(f"\n❌ Orchestration error: {e}")
        import traceback

        traceback.print_exc()

    # 5. Rapport final
    all_passed = orchestrator.generate_report()

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
