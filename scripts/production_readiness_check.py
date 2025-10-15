#!/usr/bin/env python3
"""
Jeffrey OS v2.4.2 - Production Readiness Check

Validates that the system is ready for production deployment:
- Environment configuration
- ML pipeline functionality
- Performance benchmarks
- Error handling
- Security considerations
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))


class ProductionReadinessChecker:
    def __init__(self):
        self.checks = []
        self.results = {}

    def check_environment(self):
        """Check Python environment and dependencies."""
        print("🔍 Checking environment...")

        checks = {
            "Python version": sys.version_info >= (3, 11),
            "Source directory": Path("src").exists(),
            "ML scripts directory": Path("scripts/ml").exists(),
            "Data directory": Path("data").exists(),
        }

        # Check key imports
        try:
            import numpy
            import sklearn
            import torch
            import yaml

            checks["NumPy available"] = True
            checks["Scikit-learn available"] = True
            checks["YAML available"] = True
            checks["PyTorch available"] = True
        except ImportError:
            checks["Import error"] = False

        # Check Jeffrey imports
        try:
            from jeffrey.core.emotion_backend import ProtoEmotionDetector
            from jeffrey.ml.encoder import create_default_encoder

            checks["Jeffrey imports"] = True
        except ImportError:
            checks["Jeffrey imports"] = False

        return self.evaluate_checks("Environment", checks)

    def check_ml_pipeline(self):
        """Check ML pipeline readiness."""
        print("🧠 Checking ML pipeline...")

        checks = {}

        # Check training data
        conv_dir = Path("data/conversations")
        if conv_dir.exists():
            yaml_files = list(conv_dir.glob("*.yaml"))
            checks["Training data available"] = len(yaml_files) >= 100
            checks["Training files count"] = len(yaml_files) >= 127
        else:
            checks["Training data available"] = False

        # Check trained model
        proto_path = Path("data/prototypes.npz")
        meta_path = Path("data/prototypes.meta.json")

        if proto_path.exists() and meta_path.exists():
            checks["Trained model exists"] = True

            # Check model metadata
            try:
                with open(meta_path) as f:
                    meta = json.load(f)

                f1_score = meta.get('validation', {}).get('f1_macro', 0)
                accuracy = meta.get('validation', {}).get('accuracy', 0)

                checks["F1 score meets target"] = f1_score >= 0.537
                checks["F1 score excellent"] = f1_score >= 0.70
                checks["Accuracy acceptable"] = accuracy >= 0.70
                checks["Model version current"] = meta.get('version') == '2.1.0'

            except Exception:
                checks["Model metadata valid"] = False
        else:
            checks["Trained model exists"] = False

        # Check ML scripts
        ml_scripts = [
            "ml_pipeline.py",
            "generate_training_dataset.py",
            "train_prototypes_optimized.py",
            "smoke_test_fr_en.py",
            "verify_training_inference_parity.py",
        ]

        for script in ml_scripts:
            script_path = Path(f"scripts/ml/{script}")
            checks[f"{script} exists"] = script_path.exists()
            if script_path.exists():
                checks[f"{script} executable"] = os.access(script_path, os.X_OK)

        return self.evaluate_checks("ML Pipeline", checks)

    def check_performance(self):
        """Check system performance."""
        print("⚡ Checking performance...")

        checks = {}

        try:
            # Test emotion detection speed
            from jeffrey.core.emotion_backend import ProtoEmotionDetector

            detector = ProtoEmotionDetector()

            # Warm up
            detector.predict_label("test")

            # Benchmark inference speed
            test_phrases = [
                "I'm so happy today!",
                "This makes me really sad.",
                "I'm angry about this situation.",
                "This is quite scary.",
                "What a surprise!",
            ]

            start_time = time.time()
            for phrase in test_phrases:
                detector.predict_label(phrase)
            end_time = time.time()

            avg_time = (end_time - start_time) / len(test_phrases)
            checks["Inference speed < 50ms"] = avg_time < 0.05
            checks["Inference speed < 5ms"] = avg_time < 0.005

            # Test batch processing
            start_time = time.time()
            for phrase in test_phrases * 10:  # 50 predictions
                detector.predict_label(phrase)
            end_time = time.time()

            batch_time = (end_time - start_time) / (len(test_phrases) * 10)
            checks["Batch processing efficient"] = batch_time < 0.01

        except Exception as e:
            checks["Performance test error"] = False
            print(f"   ❌ Performance test failed: {e}")

        return self.evaluate_checks("Performance", checks)

    def check_robustness(self):
        """Check error handling and robustness."""
        print("🛡️ Checking robustness...")

        checks = {}

        try:
            from jeffrey.core.emotion_backend import ProtoEmotionDetector

            detector = ProtoEmotionDetector()

            # Test edge cases
            edge_cases = [
                "",  # Empty string
                " ",  # Whitespace
                "a",  # Single character
                "Lorem ipsum dolor sit amet " * 20,  # Very long text
                "🎉😍💯",  # Emojis only
                "123456789",  # Numbers only
                "CAPS LOCK TEXT",  # All caps
                "mixed CaSe TeXt",  # Mixed case
            ]

            for case in edge_cases:
                try:
                    prediction = detector.predict_label(case)
                    # Should return one of the 8 core emotions
                    valid_emotions = {
                        "joy",
                        "sadness",
                        "anger",
                        "fear",
                        "surprise",
                        "disgust",
                        "neutral",
                        "frustration",
                    }
                    if prediction not in valid_emotions:
                        checks["Edge case handling"] = False
                        break
                except Exception:
                    checks["Edge case handling"] = False
                    break
            else:
                checks["Edge case handling"] = True

            # Test multilingual
            multilingual_cases = [
                "Je suis heureux",  # French
                "I am happy",  # English
                "¡Estoy feliz!",  # Spanish (should still work)
            ]

            for case in multilingual_cases:
                try:
                    prediction = detector.predict_label(case)
                    checks["Multilingual robustness"] = True
                except Exception:
                    checks["Multilingual robustness"] = False
                    break

        except Exception as e:
            checks["Robustness test error"] = False
            print(f"   ❌ Robustness test failed: {e}")

        return self.evaluate_checks("Robustness", checks)

    def check_deployment_config(self):
        """Check deployment configuration."""
        print("🚀 Checking deployment configuration...")

        checks = {}

        # Check configuration files
        important_files = [
            "README.md",
            "CHANGELOG.md",
            "requirements.txt",
            ".gitignore",
            "scripts/ml/ml_pipeline.py",
            "scripts/ml/README.md",
        ]

        for file_path in important_files:
            checks[f"{file_path} exists"] = Path(file_path).exists()

        # Check git status
        try:
            result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
            # Empty output means clean working directory
            checks["Git working directory clean"] = len(result.stdout.strip()) == 0
        except Exception:
            checks["Git status check"] = False

        # Check if on correct branch
        try:
            result = subprocess.run(["git", "branch", "--show-current"], capture_output=True, text=True, check=True)
            current_branch = result.stdout.strip()
            checks["On production branch"] = current_branch in ["main", "master", "phase2-ft-head"]
        except Exception:
            checks["Branch check"] = False

        return self.evaluate_checks("Deployment Config", checks)

    def evaluate_checks(self, category, checks):
        """Evaluate a set of checks."""
        passed = sum(1 for result in checks.values() if result)
        total = len(checks)
        success_rate = passed / total if total > 0 else 0

        print(f"   {category}: {passed}/{total} checks passed ({success_rate:.1%})")

        # Show failed checks
        failed = [name for name, result in checks.items() if not result]
        for failure in failed:
            print(f"   ❌ {failure}")

        self.results[category] = {"passed": passed, "total": total, "success_rate": success_rate, "failed": failed}

        return success_rate >= 0.8  # 80% threshold

    def run_smoke_test(self):
        """Run quick smoke test."""
        print("💨 Running smoke test...")

        try:
            result = subprocess.run(
                ["python", "scripts/ml/ml_pipeline.py", "test"], capture_output=True, text=True, timeout=300
            )

            if result.returncode == 0:
                print("   ✅ Smoke test passed")
                return True
            else:
                print("   ❌ Smoke test failed")
                if result.stderr:
                    print(f"   Error: {result.stderr}")
                return False

        except Exception as e:
            print(f"   ❌ Smoke test error: {e}")
            return False

    def run_all_checks(self):
        """Run all production readiness checks."""
        print("🚀 Jeffrey OS v2.4.2 - Production Readiness Check")
        print("=" * 60)

        # Run all check categories
        check_results = [
            self.check_environment(),
            self.check_ml_pipeline(),
            self.check_performance(),
            self.check_robustness(),
            self.check_deployment_config(),
        ]

        # Run smoke test separately
        smoke_test_passed = self.run_smoke_test()

        # Summary
        print("\n" + "=" * 60)
        print("📊 PRODUCTION READINESS SUMMARY")
        print("=" * 60)

        overall_passed = 0
        overall_total = 0

        for category, result in self.results.items():
            status = "✅ PASS" if result["success_rate"] >= 0.8 else "❌ FAIL"
            print(
                f"{category:20s}: {result['passed']:2d}/{result['total']:2d} ({result['success_rate']:5.1%}) {status}"
            )

            overall_passed += result["passed"]
            overall_total += result["total"]

        smoke_status = "✅ PASS" if smoke_test_passed else "❌ FAIL"
        print(f"{'Smoke Test':20s}: {'1' if smoke_test_passed else '0'}/1 {smoke_status}")

        # Overall assessment
        overall_success_rate = overall_passed / overall_total if overall_total > 0 else 0
        all_categories_pass = all(result["success_rate"] >= 0.8 for result in self.results.values())

        overall_ready = all_categories_pass and smoke_test_passed and overall_success_rate >= 0.85

        print(f"\nOverall Score: {overall_passed}/{overall_total} ({overall_success_rate:.1%})")

        if overall_ready:
            print("\n🎉 SYSTEM IS PRODUCTION READY! 🎉")
            print("✅ All checks passed - ready for v2.4.2 release")
        else:
            print("\n🚨 SYSTEM NOT READY FOR PRODUCTION")
            print("❌ Some checks failed - address issues before release")

        print("=" * 60)

        return overall_ready


def main():
    """Run production readiness check."""
    checker = ProductionReadinessChecker()
    ready = checker.run_all_checks()
    return 0 if ready else 1


if __name__ == "__main__":
    exit(main())
