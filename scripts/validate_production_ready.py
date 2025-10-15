#!/usr/bin/env python3
"""
Production-ready validation script for Jeffrey OS v2.4.2.
Runs all critical checks before tagging as production.

Usage:
  python scripts/validate_production_ready.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def check(name, condition, message):
    """Helper for checks."""
    status = "✅" if condition else "❌"
    print(f"{status} {name}: {message}")
    return condition


def main():
    """Run all production-ready checks."""
    print("🔍 JEFFREY OS v2.4.2 - PRODUCTION-READY VALIDATION")
    print("=" * 70)

    all_passed = True

    # Check 1: Linear head metadata exists and valid (priority over prototypes)
    linear_meta_path = Path("data/linear_head.meta.json")
    proto_meta_path = Path("data/prototypes.meta.json")

    if linear_meta_path.exists():
        with open(linear_meta_path) as f:
            meta = json.load(f)

        f1 = meta.get("training", {}).get("val_f1", 0)
        all_passed &= check("Linear Head F1", f1 >= 0.45, f"F1={f1:.3f} (target: ≥0.45, linear head validation)")
        all_passed &= check("Linear Head Metadata", True, "linear_head.meta.json found")
    elif proto_meta_path.exists():
        with open(proto_meta_path) as f:
            meta = json.load(f)

        f1 = meta.get("validation", {}).get("f1_macro", 0)
        ece = meta.get("validation", {}).get("ece", 1)

        all_passed &= check("F1 LOSO (Proto)", f1 >= 0.30, f"F1={f1:.3f} (fallback to prototypes)")
        all_passed &= check("ECE (Proto)", ece < 0.25, f"ECE={ece:.3f} (prototype calibration)")
    else:
        all_passed &= check("Metadata", False, "Neither linear_head.meta.json nor prototypes.meta.json found")

    # Check 2: Model files exist (linear head priority)
    linear_head_path = Path("data/linear_head.joblib")
    proto_path = Path("data/prototypes.npz")

    if linear_head_path.exists():
        all_passed &= check("Linear Head Model", True, "linear_head.joblib found")
    else:
        all_passed &= check("Linear Head Model", False, "linear_head.joblib NOT FOUND")

    if proto_path.exists():
        check("Prototypes (Fallback)", True, "prototypes.npz found (fallback available)")
    else:
        check("Prototypes (Fallback)", False, "prototypes.npz NOT FOUND (no fallback)")

    # Check 3: Real data used (not synthetic)
    real_data_path = Path("data/conversations_real")
    goemotions_path = Path("data/conversations_goemotions")
    has_real = real_data_path.exists() or goemotions_path.exists()
    all_passed &= check(
        "Real Data", has_real, f"Real dataset {'found' if has_real else 'NOT FOUND (still synthetic!)'}"
    )

    # Check 4: Smoke test (run and parse)
    print("\n🧪 Running smoke test...")
    try:
        result = subprocess.run(
            [sys.executable, "scripts/smoke_test_fr_en.py"],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "PYTHONPATH": "."},
        )

        # Parse output for accuracy and fallback
        output = result.stdout

        # Extract accuracy from "Total cases : 56, Correct : XX, Accuracy : XX.X%"
        accuracy_match = [l for l in output.split("\n") if "Accuracy" in l and ":" in l]

        if accuracy_match:
            # Extract percentage from line like "Accuracy         : 41.1%"
            try:
                acc_line = accuracy_match[0]
                acc_percent = float(acc_line.split(":")[-1].replace("%", "").strip())
                accuracy_ok = acc_percent >= 60.0  # Quick wins target
                fallback_ok = True  # If ML is working (no dimension errors), fallback is OK

                all_passed &= check("Smoke Test Accuracy", accuracy_ok, f"Accuracy {acc_percent:.1f}% (target: ≥60%)")
                all_passed &= check(
                    "Smoke Test ML Working", "dimension mismatch" not in output.lower(), "No encoder dimension errors"
                )
            except Exception as e:
                all_passed &= check("Smoke Test", False, f"Could not parse accuracy: {e}")
        else:
            all_passed &= check("Smoke Test", False, "Could not find accuracy in smoke test output")

    except Exception as e:
        all_passed &= check("Smoke Test", False, f"Failed to run: {e}")

    # Check 5: Pre-commit hooks (optional but recommended)
    print("\n🔧 Checking code quality...")
    try:
        result = subprocess.run(["pre-commit", "run", "--all-files"], capture_output=True, timeout=120)
        precommit_ok = result.returncode == 0
        check(
            "Pre-commit Hooks",
            precommit_ok,
            "All hooks passed" if precommit_ok else "Some hooks failed (fix with: ruff --fix, black, isort)",
        )
        # Not blocking, just warning
    except FileNotFoundError:
        print("⚠️  Pre-commit not installed (optional)")
    except Exception:
        print("⚠️  Pre-commit check skipped")

    # Final verdict
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL CRITICAL CHECKS PASSED - PRODUCTION-READY ✅")
        print("\nNext steps:")
        print("  1. git add .")
        print("  2. git commit -m 'feat: v2.4.2 - Real data training (GoEmotions)'")
        print("  3. git tag v2.4.2-prod")
        print("  4. Deploy to production")
        return 0
    else:
        print("❌ SOME CHECKS FAILED - NOT READY FOR PRODUCTION")
        print("\nFix issues above before tagging v2.4.2-prod")
        return 1


if __name__ == "__main__":
    sys.exit(main())
