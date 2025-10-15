#!/usr/bin/env python3
"""Test all imports work correctly"""

import os
import sys
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_imports():
    """Test all module imports"""
    modules = [
        "src.jeffrey.utils.logger",
        "src.jeffrey.core.memory.unified_memory",
        "src.jeffrey.core.learning.jeffrey_meta_learning_integration",
        "src.jeffrey.core.learning.theory_of_mind",
        "src.jeffrey.core.learning.unified_curiosity_engine",
        "src.jeffrey.core.learning.auto_learner",
        "src.jeffrey.core.learning.contextual_learning_engine",
    ]

    print("🔍 Testing imports...")
    print("-" * 50)

    failed = []

    for module_name in modules:
        try:
            # Try to import
            module = __import__(module_name, fromlist=[""])

            # Check for key classes
            parts = module_name.split(".")
            if "meta_learning" in parts[-1]:
                assert hasattr(module, "MetaLearningIntegration")
            elif "theory_of_mind" in parts[-1]:
                assert hasattr(module, "TheoryOfMind")
            elif "curiosity" in parts[-1]:
                assert hasattr(module, "UnifiedCuriosityEngine")
            elif "auto_learner" in parts[-1]:
                assert hasattr(module, "AutoLearner")
            elif "contextual" in parts[-1]:
                assert hasattr(module, "ContextualLearningEngine")
            elif "unified_memory" in parts[-1]:
                assert hasattr(module, "UnifiedMemory")

            print(f"✅ {module_name}")

        except Exception as e:
            print(f"❌ {module_name}")
            print(f"   Error: {e}")
            traceback.print_exc()
            failed.append(module_name)

    print("-" * 50)

    if failed:
        print(f"\n❌ Failed imports: {len(failed)}")
        for m in failed:
            print(f"  - {m}")
        return False
    else:
        print("\n✅ All imports successful!")
        return True


if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
