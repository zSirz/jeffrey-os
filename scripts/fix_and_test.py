#!/usr/bin/env python3
"""
Script qui corrige tous les problèmes et lance les tests automatiquement
"""

import os
import subprocess
import sys
import time
from pathlib import Path


def fix_neuralbus_issues():
    """Corrige tous les problèmes liés à NeuralBus"""
    print("🔧 Fixing NeuralBus issues...")

    fixes_applied = []

    # 1. Fix loop_manager.py - NullBus import
    loop_manager = Path("src/jeffrey/core/loops/loop_manager.py")
    if loop_manager.exists():
        content = loop_manager.read_text()
        original = content

        # Fix import NullBus
        if "from .base import NullBus" in content:
            # Crée NullBus inline car test_helpers peut ne pas exister
            if "class NullBus:" not in content:
                nullbus_code = '''
# Temporary NullBus implementation for fallback
class NullBus:
    """Fallback bus for when NeuralBus is not available"""
    async def start(self):
        pass

    async def stop(self):
        pass

    async def publish(self, *args, **kwargs):
        pass

    async def subscribe(self, *args, **kwargs):
        pass

    def get_metrics(self):
        return {
            'published': 0,
            'consumed': 0,
            'dropped': 0,
            'p99_latency_ms': 0,
            'p95_latency_ms': 0,
            'p50_latency_ms': 0,
            'pending_messages': 0,
            'dlq_count': 0
        }
'''
                # Remplace l'import par la classe
                content = content.replace("from .base import NullBus", "")
                # Ajoute la classe après les imports
                lines = content.split("\n")
                import_end = 0
                for i, line in enumerate(lines):
                    if (
                        line
                        and not line.strip().startswith("#")
                        and not line.startswith("import")
                        and not line.startswith("from")
                    ):
                        import_end = i
                        break
                lines.insert(import_end, nullbus_code)
                content = "\n".join(lines)
                fixes_applied.append("Created inline NullBus class in loop_manager.py")

        # Fix initialize() calls if any
        if ".initialize()" in content:
            content = content.replace(".initialize()", ".start()")
            fixes_applied.append("Changed initialize() to start() in loop_manager.py")

        if content != original:
            loop_manager.write_text(content)
            print(f"  ✅ Fixed {loop_manager}")

    # 2. Fix validate_phase30.py - syntax error in import
    validate_script = Path("scripts/validate_phase30.py")
    if validate_script.exists():
        content = validate_script.read_text()
        original = content

        # Fix double "as NeuralBusV2 as NeuralBus"
        if "as NeuralBusV2 as NeuralBus" in content:
            content = content.replace(
                "from jeffrey.core.neuralbus.bus_v2 import NeuralBusV2 as NeuralBusV2 as NeuralBus",
                "from jeffrey.core.neuralbus.bus_v2 import NeuralBusV2 as NeuralBus",
            )
            fixes_applied.append("Fixed double alias in validate_phase30.py")

        if content != original:
            validate_script.write_text(content)
            print(f"  ✅ Fixed {validate_script}")

    # 3. Fix any remaining initialize() calls
    for py_file in Path("src").rglob("*.py"):
        try:
            content = py_file.read_text()
            original = content

            # Only fix neural/bus related initialize calls
            if ".initialize()" in content and ("neural" in content.lower() or "bus" in content.lower()):
                content = content.replace(".initialize()", ".start()")
                fixes_applied.append(f"Fixed initialize() in {py_file.name}")

            if content != original:
                py_file.write_text(content)
                print(f"  ✅ Fixed {py_file.name}")
        except Exception:
            pass

    # 4. Same for scripts
    for script in Path("scripts").glob("*.py"):
        if script.name == "fix_and_test.py":
            continue  # Skip self
        try:
            content = script.read_text()
            original = content

            if ".initialize()" in content:
                content = content.replace(".initialize()", ".start()")
                fixes_applied.append(f"Fixed initialize() in {script.name}")

            if content != original:
                script.write_text(content)
                print(f"  ✅ Fixed {script.name}")
        except Exception:
            pass

    print(f"\n📊 Applied {len(fixes_applied)} fixes:")
    for fix in fixes_applied:
        print(f"  - {fix}")

    return len(fixes_applied) > 0


def test_imports():
    """Teste que les imports fonctionnent"""
    print("\n🧪 Testing imports...")

    tests = [
        ("LoopManager", "from jeffrey.core.loops.loop_manager import LoopManager"),
        ("NeuralBusV2", "from jeffrey.core.neuralbus.bus_v2 import NeuralBusV2"),
        ("BaseLoop", "from jeffrey.core.loops.base import BaseLoop"),
        (
            "LoadConfig",
            "import sys; sys.path.insert(0, 'scripts'); from generate_load import LoadConfig",
        ),
    ]

    all_ok = True
    for name, import_stmt in tests:
        try:
            exec(import_stmt)
            print(f"  ✅ {name}")
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            all_ok = False

    return all_ok


def check_nats():
    """Vérifie que NATS est lancé"""
    print("\n🔍 Checking NATS status...")

    # Utilise le nats_manager.py
    result = subprocess.run([sys.executable, "scripts/nats_manager.py", "status"], capture_output=True, text=True)

    if "NATS running" in result.stdout or "External NATS" in result.stdout:
        print("  ✅ NATS is available")
        return True
    else:
        # Essaie de démarrer NATS
        print("  ⚠️ NATS not running, attempting to start...")
        result = subprocess.run(
            [sys.executable, "scripts/nats_manager.py", "start"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if "started" in result.stdout.lower() or "external" in result.stdout.lower():
            print("  ✅ NATS started or external NATS detected")
            return True
        else:
            print("  ❌ Could not start NATS")
            return False


def run_smoke_test():
    """Lance le smoke test"""
    print("\n🚀 Running smoke test...")

    # Set namespace
    os.environ["NB_NS"] = f"test_{int(time.time())}"

    # Lance le test très court
    cmd = [
        sys.executable,
        "scripts/generate_load.py",
        "--phase",
        "quick",
        "--hours",
        "0.008",  # ~30 secondes
        "--rate",
        "50",  # Réduit le rate pour un test rapide
        "--non-interactive",
    ]

    print(f"  Command: {' '.join(cmd)}")
    print(f"  Namespace: {os.environ['NB_NS']}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        output = result.stdout + result.stderr

        # Affiche les dernières lignes pour debug
        lines = output.split("\n")

        # Cherche les indicateurs de succès
        passed = False
        for line in lines:
            if "PHASE QUICK PASSED" in line or "SYSTEM VALIDATED" in line:
                passed = True
            if "P99" in line and "ms" in line:
                print(f"  {line.strip()}")
            elif "Messages sent:" in line:
                print(f"  {line.strip()}")
            elif "VALIDATION RESULTS" in line:
                # Affiche les dernières lignes après validation
                idx = lines.index(line)
                for i in range(idx, min(idx + 15, len(lines))):
                    if lines[i].strip():
                        print(f"  {lines[i].strip()}")

        if passed:
            print("\n✅ Smoke test PASSED!")
            return True
        else:
            print("\n❌ Smoke test did not pass validation")

            # Cherche des erreurs spécifiques
            for line in lines:
                if "error" in line.lower() or "exception" in line.lower():
                    print(f"  Error: {line.strip()}")
                    break

            print("\nLast 10 lines of output:")
            print("\n".join(lines[-10:]))
            return False

    except subprocess.TimeoutExpired:
        print("❌ Smoke test timeout")
        return False
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False


def main():
    """Main function"""
    print("=" * 60)
    print("🔧 JEFFREY OS - AUTOMATIC FIX AND TEST")
    print("=" * 60)

    # 1. Applique les fixes d'abord
    if fix_neuralbus_issues():
        print("\n✅ Fixes applied successfully")
    else:
        print("\n✅ No fixes needed or already applied")

    # 2. Teste les imports
    if not test_imports():
        print("\n⚠️ Some imports failed, but continuing...")

    # 3. Vérifie NATS
    if not check_nats():
        print("\n❌ NATS is required. Please ensure NATS is running.")
        print("You can start it with: nats-server -js")
        return 1

    # 4. Lance le smoke test
    print("\n" + "=" * 60)
    print("🏃 Running validation...")
    print("=" * 60)

    if run_smoke_test():
        print("\n" + "=" * 60)
        print("🎉 SUCCESS! System is working!")
        print("=" * 60)
        print("\n✅ Next steps:")
        print("  1. Quick stress test (3 min):")
        print("     make load-stress")
        print("  2. Full test sequence:")
        print("     make load-all")
        print("  3. Production soak test (2h):")
        print("     make load-soak")
        return 0
    else:
        print("\n⚠️ Smoke test did not fully pass, but system may still be functional.")
        print("Try running a longer test:")
        print("  python scripts/generate_load.py --phase quick --hours 0.05")
        return 1


if __name__ == "__main__":
    sys.exit(main())
