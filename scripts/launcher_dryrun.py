#!/usr/bin/env python3
"""
Bundle 1 Launcher with --dry-run mode
Tests module compatibility before actual launch
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path


class DryRunLauncher:
    def __init__(self, bundle_file: str):
        with open(bundle_file) as f:
            self.data = json.load(f)

        self.bundle = self.data["bundle1_recommendations"]
        self.errors = []
        self.warnings = []

    def check_paths(self) -> bool:
        """Verify all module paths exist"""
        print("🔍 Checking module paths...")

        for module in self.bundle["modules"]:
            path = Path(module["path"])
            if not path.exists():
                self.errors.append(f"Missing file: {path}")
            elif not path.is_file():
                self.errors.append(f"Not a file: {path}")
            else:
                print(f"  ✅ {module['name']}")

        return len(self.errors) == 0

    def check_imports(self) -> bool:
        """Try importing each module"""
        print("\n📦 Checking imports...")

        for module in self.bundle["modules"]:
            try:
                spec = importlib.util.spec_from_file_location(module["name"], module["path"])
                if spec and spec.loader:
                    test_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(test_module)
                    print(f"  ✅ {module['name']} imported")
                else:
                    self.errors.append(f"Cannot import {module['name']}")
            except Exception as e:
                self.errors.append(f"{module['name']}: {str(e)[:50]}")

        return len(self.errors) == 0

    def check_interfaces(self) -> bool:
        """Check required methods exist"""
        print("\n🔌 Checking interfaces...")

        required_methods = ["process", "health_check"]

        for module_data in self.bundle["modules"]:
            try:
                spec = importlib.util.spec_from_file_location(module_data["name"], module_data["path"])
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Check for required methods
                    found_methods = []

                    # Check module level
                    for method in required_methods:
                        if hasattr(module, method):
                            found_methods.append(method)

                    # Check in classes
                    if len(found_methods) < len(required_methods):
                        for attr_name in dir(module):
                            if not attr_name.startswith("_"):
                                attr = getattr(module, attr_name)
                                if isinstance(attr, type):  # It's a class
                                    for method in required_methods:
                                        if hasattr(attr, method):
                                            found_methods.append(method)

                    missing = set(required_methods) - set(found_methods)
                    if missing:
                        self.warnings.append(f"{module_data['name']}: missing {missing}")
                    else:
                        print(f"  ✅ {module_data['name']} has all interfaces")

            except Exception as e:
                self.errors.append(f"{module_data['name']}: {str(e)[:50]}")

        return True  # Warnings OK, errors not

    def run(self, dry_run: bool = True) -> bool:
        """Run all checks"""

        print("🚀 BUNDLE 1 DRY-RUN CHECK")
        print("=" * 50)

        # Run checks
        path_ok = self.check_paths()
        import_ok = self.check_imports() if path_ok else False
        interface_ok = self.check_interfaces() if import_ok else False

        # Report
        print("\n" + "=" * 50)
        print("📊 DRY-RUN REPORT")
        print("=" * 50)

        if self.errors:
            print("\n❌ ERRORS:")
            for error in self.errors:
                print(f"  - {error}")

        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings:
                print(f"  - {warning}")

        if not self.errors and not self.warnings:
            print("\n✅ All checks passed!")
            print("Ready to launch Bundle 1 for real!")

            if not dry_run:
                print("\n🎯 Launching Bundle 1...")
                # Import and run the real launcher
                from launcher import JeffreyLauncher

                launcher = JeffreyLauncher()
                return launcher.boot()

        return len(self.errors) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Test only, don't launch")
    parser.add_argument("--bundle", default="artifacts/inventory_ultimate.json", help="Bundle config file")

    args = parser.parse_args()

    launcher = DryRunLauncher(args.bundle)
    success = launcher.run(dry_run=args.dry_run)

    sys.exit(0 if success else 1)
