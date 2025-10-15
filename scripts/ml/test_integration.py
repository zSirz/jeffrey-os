#!/usr/bin/env python3
"""
Integration Test for Jeffrey OS v2.4.2 ML Pipeline

Tests the complete workflow:
1. Data generation
2. Model training
3. Inference
4. Performance validation

This ensures the complete pipeline works end-to-end in production.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from jeffrey.core.emotion_backend import ProtoEmotionDetector


class MLIntegrationTest:
    def __init__(self):
        self.temp_dir = None
        self.original_dir = os.getcwd()

    def setup_test_environment(self):
        """Create temporary environment for testing."""
        print("🔧 Setting up test environment...")

        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        print(f"   Temp dir: {self.temp_dir}")

        # Copy source code
        src_path = Path(self.original_dir) / "src"
        scripts_path = Path(self.original_dir) / "scripts" / "ml"

        temp_src = Path(self.temp_dir) / "src"
        temp_scripts = Path(self.temp_dir) / "scripts" / "ml"

        shutil.copytree(src_path, temp_src)
        temp_scripts.parent.mkdir(exist_ok=True)
        shutil.copytree(scripts_path, temp_scripts)

        # Change to temp directory
        os.chdir(self.temp_dir)

        print("✅ Test environment ready")

    def cleanup_test_environment(self):
        """Clean up test environment."""
        print("🧹 Cleaning up test environment...")
        os.chdir(self.original_dir)
        if self.temp_dir:
            shutil.rmtree(self.temp_dir)
        print("✅ Cleanup complete")

    def run_command(self, cmd, description, check_success=True):
        """Run a command and return success status."""
        print(f"🔄 {description}...")
        try:
            result = subprocess.run(cmd, shell=True, check=check_success, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                print(f"✅ {description} successful")
                return True
            else:
                print(f"❌ {description} failed")
                if result.stderr:
                    print(f"   Error: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print(f"⏰ {description} timed out")
            return False
        except Exception as e:
            print(f"💥 {description} error: {e}")
            return False

    def test_data_generation(self):
        """Test training data generation."""
        cmd = "PYTHONPATH=src python scripts/ml/ml_pipeline.py generate-data"
        success = self.run_command(cmd, "Data generation")

        if success:
            # Verify data was created
            conv_dir = Path("data/conversations")
            if conv_dir.exists():
                yaml_files = list(conv_dir.glob("*.yaml"))
                if len(yaml_files) >= 100:  # Should have ~127 files
                    print(f"✅ Generated {len(yaml_files)} YAML files")
                    return True
                else:
                    print(f"❌ Only {len(yaml_files)} YAML files generated")
            else:
                print("❌ No conversation directory created")

        return False

    def test_model_training(self):
        """Test model training."""
        cmd = "PYTHONPATH=src python scripts/ml/ml_pipeline.py train"
        success = self.run_command(cmd, "Model training")

        if success:
            # Verify model artifacts were created
            proto_path = Path("data/prototypes.npz")
            meta_path = Path("data/prototypes.meta.json")

            if proto_path.exists() and meta_path.exists():
                # Check metadata
                with open(meta_path) as f:
                    meta = json.load(f)

                f1_score = meta.get('validation', {}).get('f1_macro', 0)
                if f1_score >= 0.5:  # Reasonable threshold
                    print(f"✅ Model trained with F1: {f1_score:.3f}")
                    return True
                else:
                    print(f"❌ Low F1 score: {f1_score:.3f}")
            else:
                print("❌ Model artifacts not created")

        return False

    def test_inference(self):
        """Test model inference."""
        try:
            print("🔄 Testing inference...")

            # Initialize detector
            detector = ProtoEmotionDetector()

            # Test predictions
            test_cases = [
                ("I'm so happy today!", "joy"),
                ("This is really sad", "sadness"),
                ("I'm angry about this", "anger"),
                ("This scares me", "fear"),
            ]

            correct = 0
            total = len(test_cases)

            for text, expected in test_cases:
                prediction = detector.predict_label(text)
                if prediction == expected:
                    correct += 1
                print(f"   '{text}' → {prediction} {'✅' if prediction == expected else '❌'}")

            accuracy = correct / total
            if accuracy >= 0.5:  # 50% threshold for basic functionality
                print(f"✅ Inference working: {accuracy:.1%} accuracy")
                return True
            else:
                print(f"❌ Low inference accuracy: {accuracy:.1%}")
                return False

        except Exception as e:
            print(f"❌ Inference failed: {e}")
            return False

    def test_smoke_tests(self):
        """Test smoke test validation."""
        cmd = "PYTHONPATH=src python scripts/ml/ml_pipeline.py test"
        return self.run_command(cmd, "Smoke tests")

    def run_full_integration_test(self):
        """Run complete integration test."""
        print("🚀 Starting ML Pipeline Integration Test")
        print("=" * 60)

        try:
            self.setup_test_environment()

            # Test steps in order
            steps = [
                ("Data Generation", self.test_data_generation),
                ("Model Training", self.test_model_training),
                ("Inference Testing", self.test_inference),
                ("Smoke Tests", self.test_smoke_tests),
            ]

            results = {}
            for step_name, step_func in steps:
                print(f"\n📋 Step: {step_name}")
                print("-" * 40)

                success = step_func()
                results[step_name] = success

                if not success:
                    print(f"❌ Integration test failed at: {step_name}")
                    break

            # Summary
            print("\n" + "=" * 60)
            print("📊 INTEGRATION TEST RESULTS")
            print("=" * 60)

            all_passed = all(results.values())
            for step, success in results.items():
                status = "✅ PASS" if success else "❌ FAIL"
                print(f"{step:20s}: {status}")

            overall = "🎉 ALL TESTS PASSED" if all_passed else "🚨 SOME TESTS FAILED"
            print(f"\nOverall Result: {overall}")
            print("=" * 60)

            return all_passed

        except Exception as e:
            print(f"💥 Integration test crashed: {e}")
            return False

        finally:
            self.cleanup_test_environment()


def main():
    """Run integration tests."""
    test = MLIntegrationTest()
    success = test.run_full_integration_test()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
