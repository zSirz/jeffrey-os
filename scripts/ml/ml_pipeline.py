#!/usr/bin/env python3
"""
Jeffrey OS v2.4.2 - ML Pipeline Manager
Production-ready interface for emotion detection ML operations.

Available commands:
- generate-data: Create training YAML dataset
- train: Train prototype classifier
- test: Run smoke tests
- verify: Check training-inference parity
- monitor: Start emotion detection monitoring
- status: Show current model status
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))


def run_command(cmd, description, check=True):
    """Run a subprocess command with proper error handling."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Stderr: {e.stderr}")
        return False


def check_environment():
    """Check if the environment is properly set up."""
    print("🔍 Checking environment...")

    # Check Python path
    python_cmd = sys.executable
    print(f"Using Python: {python_cmd}")

    # Check if we're in the right directory
    if not Path("src").exists():
        print("❌ Error: Not in Jeffrey OS root directory (missing src/)")
        return False

    # Check if data directory exists
    Path("data").mkdir(exist_ok=True)

    print("✅ Environment OK")
    return True


def generate_data():
    """Generate training dataset."""
    if not check_environment():
        return False

    script_path = Path(__file__).parent / "generate_training_dataset.py"
    cmd = f"PYTHONPATH=src python {script_path}"
    return run_command(cmd, "Generating training YAML dataset")


def train_model():
    """Train the prototype classifier."""
    if not check_environment():
        return False

    # Check if training data exists
    conv_dir = Path("data/conversations")
    if not conv_dir.exists() or not list(conv_dir.glob("*.yaml")):
        print("⚠️  No training data found. Generating dataset first...")
        if not generate_data():
            return False

    script_path = Path(__file__).parent / "train_prototypes_optimized.py"
    cmd = f"PYTHONPATH=src python {script_path}"
    return run_command(cmd, "Training prototype classifier")


def run_smoke_test():
    """Run smoke tests to validate model."""
    if not check_environment():
        return False

    # Check if model exists
    if not Path("data/prototypes.npz").exists():
        print("⚠️  No trained model found. Training model first...")
        if not train_model():
            return False

    script_path = Path(__file__).parent / "smoke_test_fr_en.py"
    cmd = f"PYTHONPATH=src python {script_path}"
    return run_command(cmd, "Running smoke tests")


def verify_parity():
    """Verify training-inference parity."""
    if not check_environment():
        return False

    # Check if model exists
    if not Path("data/prototypes.npz").exists():
        print("❌ No trained model found. Run 'train' first.")
        return False

    script_path = Path(__file__).parent / "verify_training_inference_parity.py"
    cmd = f"PYTHONPATH=src python {script_path}"
    return run_command(cmd, "Verifying training-inference parity")


def monitor_emotions():
    """Start emotion detection monitoring."""
    if not check_environment():
        return False

    script_path = Path(__file__).parent / "monitor_emotions.py"
    cmd = f"PYTHONPATH=src python {script_path}"
    return run_command(cmd, "Starting emotion monitoring", check=False)


def show_status():
    """Show current model status."""
    print("📊 Jeffrey OS v2.4.2 - ML Model Status")
    print("=" * 50)

    # Check training data
    conv_dir = Path("data/conversations")
    if conv_dir.exists():
        yaml_files = list(conv_dir.glob("*.yaml"))
        print(f"Training data: {len(yaml_files)} YAML files")
    else:
        print("Training data: ❌ Not generated")

    # Check trained model
    proto_path = Path("data/prototypes.npz")
    meta_path = Path("data/prototypes.meta.json")

    if proto_path.exists() and meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)

            print(f"✅ Trained model: v{meta.get('version', 'unknown')}")
            print(f"   Created: {meta.get('created_at', 'unknown')}")
            print(f"   Encoder: {meta.get('encoder_model', 'unknown')}")
            print(f"   Examples: {meta.get('num_examples_total', 'unknown')}")
            print(f"   F1 LOSO: {meta.get('validation', {}).get('f1_macro', 'N/A'):.3f}")
            print(f"   Accuracy: {meta.get('validation', {}).get('accuracy', 'N/A'):.3f}")

        except Exception as e:
            print(f"⚠️  Model exists but metadata error: {e}")
    else:
        print("❌ Trained model: Not found")

    # Check confusion matrix
    cm_path = Path("data/confusion_matrix.png")
    if cm_path.exists():
        print("✅ Confusion matrix: Available")
    else:
        print("❌ Confusion matrix: Not generated")

    print("=" * 50)


def full_pipeline():
    """Run the complete ML pipeline from scratch."""
    print("🚀 Starting complete ML pipeline...")

    steps = [
        ("Generating training data", generate_data),
        ("Training model", train_model),
        ("Running smoke tests", run_smoke_test),
        ("Verifying parity", verify_parity),
    ]

    for step_name, step_func in steps:
        print(f"\n📋 Step: {step_name}")
        if not step_func():
            print(f"❌ Pipeline failed at: {step_name}")
            return False

    print("\n🎉 Complete ML pipeline finished successfully!")
    show_status()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Jeffrey OS v2.4.2 ML Pipeline Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s status           # Show current model status
  %(prog)s generate-data    # Generate training YAML dataset
  %(prog)s train            # Train prototype classifier
  %(prog)s test             # Run smoke tests
  %(prog)s verify           # Check training-inference parity
  %(prog)s pipeline         # Run complete pipeline from scratch
  %(prog)s monitor          # Start emotion monitoring
        """,
    )

    parser.add_argument(
        'command',
        choices=['status', 'generate-data', 'train', 'test', 'verify', 'monitor', 'pipeline'],
        help='Command to execute',
    )

    args = parser.parse_args()

    print(f"Jeffrey OS v2.4.2 ML Pipeline - {args.command}")

    commands = {
        'status': show_status,
        'generate-data': generate_data,
        'train': train_model,
        'test': run_smoke_test,
        'verify': verify_parity,
        'monitor': monitor_emotions,
        'pipeline': full_pipeline,
    }

    success = commands[args.command]()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
