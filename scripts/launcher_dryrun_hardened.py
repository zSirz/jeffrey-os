#!/usr/bin/env python3
"""
Bundle 1 Launcher with --dry-run mode HARDENED
Tests module compatibility in sandboxed subprocess before actual launch
"""

import argparse
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

_RUNNER = textwrap.dedent(
    r"""
import importlib.util, sys, os, json, socket
# Block network
class _BlockedSocket(socket.socket):
    def __init__(self,*a,**k): raise RuntimeError("Network blocked")
socket.socket = _BlockedSocket
path = sys.argv[1]
req = ['process','health_check']
proj_root = os.environ.get("PYTHONPATH")
if proj_root and proj_root not in sys.path:
    sys.path.insert(0, proj_root)
out = {"import_ok": False, "interfaces_ok": False, "warnings": []}
try:
    spec = importlib.util.spec_from_file_location("mod", path)
    if not spec or not spec.loader: raise RuntimeError("spec failed")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    out["import_ok"] = True
    found = set()
    # module-level
    for r in req:
        if hasattr(m, r): found.add(r)
    # classes
    if len(found)<len(req):
        for name in dir(m):
            if name.startswith('_'): continue
            attr = getattr(m, name)
            if isinstance(attr, type):
                for r in req:
                    if hasattr(attr, r): found.add(r)
    missing = [r for r in req if r not in found]
    out["interfaces_ok"] = (len(missing)==0)
    if missing: out["warnings"].append(f"missing:{','.join(missing)}")
except Exception as e:
    out["error"] = str(e)[:200]
print(json.dumps(out))
"""
)


class DryRunLauncher:
    def __init__(self, bundle_file: str):
        with open(bundle_file) as f:
            self.data = json.load(f)

        self.bundle = self.data["bundle1_recommendations"]
        self.errors = []
        self.warnings = []

    def _sandbox_check(self, path: str) -> dict:
        """Run module check in isolated subprocess"""
        # Calculate PYTHONPATH for project root
        proj_root = None
        p = Path(path)
        for parent in p.parents:
            if parent.name == "src":
                proj_root = str(parent.parent)
                break

        env = dict(os.environ)
        if proj_root:
            env["PYTHONPATH"] = proj_root

        cmd = [sys.executable, "-I", "-S", "-E", "-c", _RUNNER, path]

        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=1.5, env=env)
            if r.returncode != 0 or not r.stdout.strip():
                return {
                    "import_ok": False,
                    "interfaces_ok": False,
                    "error": (r.stderr or "subprocess_failed")[:200],
                }
            return json.loads(r.stdout)
        except subprocess.TimeoutExpired:
            return {"import_ok": False, "interfaces_ok": False, "error": "timeout"}
        except Exception as e:
            return {"import_ok": False, "interfaces_ok": False, "error": str(e)[:100]}

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
        """Try importing each module with proper PYTHONPATH"""
        print("\n📦 Checking imports (sandboxed)...")

        import os
        import subprocess
        import sys
        from pathlib import Path

        def _src_root() -> Path:
            """Find src/ directory"""
            here = Path(__file__).resolve()
            for parent in here.parents:
                if (parent / "src").exists():
                    return parent / "src"
            return Path.cwd() / "src"

        def _dotted_from_path(p: str, src: Path) -> str:
            """Convert path to dotted module name
            src/jeffrey/core/bus/kernel_adapter.py -> jeffrey.core.bus.kernel_adapter
            """
            try:
                rel = Path(p).resolve().relative_to(src.resolve())
                return ".".join(rel.with_suffix("").parts)
            except ValueError:
                # If path is not relative to src, use filename
                return Path(p).stem

        def try_import_dotted(dotted: str, relaxed: bool = False) -> tuple[bool, str]:
            """Try importing module in subprocess with PYTHONPATH"""

            # Network blocking code
            network_block = """
import socket
class _BlockedSocket(socket.socket):
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Network blocked in dry-run")
socket.socket = _BlockedSocket
"""

            code = f"""
import os, sys, importlib
{network_block}
# Add PYTHONPATH to sys.path
pythonpath = os.environ.get('PYTHONPATH', '.')
if pythonpath not in sys.path:
    sys.path.insert(0, pythonpath)
try:
    importlib.import_module('{dotted}')
    print('OK')
except Exception as e:
    import traceback
    print(f'ERROR: {{e}}', file=sys.stderr)
    sys.exit(1)
"""

            cmd = [sys.executable, "-I", "-E"]
            if not relaxed:  # Strict mode: no site-packages
                cmd.append("-S")
            cmd += ["-c", code]

            # Setup environment with PYTHONPATH
            env = dict(os.environ)
            src_path = str(_src_root())
            env["PYTHONPATH"] = src_path
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONDONTWRITEBYTECODE"] = "1"

            # Block network in subprocess
            env["JEFFREY_NO_NETWORK"] = "1"
            env["JEFFREY_SAFE_MODE"] = "1"

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=3.0,  # Slightly longer timeout
                    env=env,
                )

                if result.returncode == 0:
                    return True, ""
                else:
                    error = result.stderr or result.stdout or "unknown error"
                    # Clean up error message
                    error = error.strip()
                    if "ERROR:" in error:
                        error = error.split("ERROR:", 1)[1].strip()
                    return False, error[:200]

            except subprocess.TimeoutExpired:
                return False, "timeout (>3s)"
            except Exception as e:
                return False, str(e)[:100]

        # Process each module
        src = _src_root()

        # Make sure src exists
        if not src.exists():
            print(f"  ❌ Source directory not found: {src}")
            self.errors.append(f"Source directory not found: {src}")
            return False

        errors = []
        warnings = []
        strict_pass = 0
        relaxed_pass = 0

        for module in self.bundle["modules"]:
            module_name = module["name"]
            module_path = Path(module["path"])

            # Skip if file doesn't exist
            if not module_path.exists():
                errors.append(f"{module_name}: File not found")
                print(f"  ❌ {module_name}: File not found")
                continue

            # Convert to dotted name
            dotted = _dotted_from_path(str(module_path), src)

            # First try: strict sandbox (-S flag)
            success, error = try_import_dotted(dotted, relaxed=False)

            if success:
                print(f"  ✅ {module_name} imported (strict sandbox)")
                strict_pass += 1
                continue

            # Check if it's a missing third-party library
            error_lower = error.lower()
            is_missing_module = any(
                x in error_lower for x in ["no module named", "cannot import", "modulenotfounderror"]
            )

            if is_missing_module:
                # Extract module name if possible
                missing_lib = None
                if "no module named" in error_lower:
                    parts = error.split("'")
                    if len(parts) >= 2:
                        missing_lib = parts[1].split(".")[0]

                # Second try: relaxed (with site-packages)
                success2, error2 = try_import_dotted(dotted, relaxed=True)

                if success2:
                    msg = "needs site-packages"
                    if missing_lib:
                        msg += f" ({missing_lib})"
                    print(f"  ⚠️  {module_name} imported (relaxed - {msg})")
                    warnings.append(f"{module_name}: {msg}")
                    relaxed_pass += 1
                else:
                    # Real import error even with site-packages
                    final_error = error2 or error
                    errors.append(f"{module_name}: {final_error}")
                    print(f"  ❌ {module_name}: {final_error}")
            else:
                # Other error (not a missing module)
                errors.append(f"{module_name}: {error}")
                print(f"  ❌ {module_name}: {error}")

        # Summary
        print("\n📊 Import Summary:")
        print(f"  Strict sandbox: {strict_pass}/{len(self.bundle['modules'])}")
        print(f"  With site-packages: {relaxed_pass}/{len(self.bundle['modules'])}")
        print(f"  Failed: {len(errors)}/{len(self.bundle['modules'])}")

        # Store results
        self.errors.extend(errors)
        self.warnings.extend(warnings)

        # Pass if no errors (warnings are OK)
        return len(errors) == 0

    def check_interfaces(self) -> bool:
        """Check required methods exist (sandboxed)"""
        print("\n🔌 Checking interfaces (sandboxed)...")

        for module_data in self.bundle["modules"]:
            result = self._sandbox_check(module_data["path"])

            if not result.get("import_ok"):
                # Already reported in check_imports
                continue

            if result.get("interfaces_ok"):
                print(f"  ✅ {module_data['name']} has all interfaces")
            else:
                warnings = result.get("warnings", [])
                for warning in warnings:
                    self.warnings.append(f"{module_data['name']}: {warning}")
                    print(f"  ⚠️  {module_data['name']}: {warning}")

        return True  # Warnings OK, errors not

    def run(self, dry_run: bool = True) -> bool:
        """Run all checks"""

        print("🚀 BUNDLE 1 DRY-RUN CHECK (HARDENED)")
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
            for error in self.errors[:10]:  # Limit to first 10
                print(f"  - {error}")
            if len(self.errors) > 10:
                print(f"  ... and {len(self.errors) - 10} more")

        if self.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in self.warnings[:10]:  # Limit to first 10
                print(f"  - {warning}")
            if len(self.warnings) > 10:
                print(f"  ... and {len(self.warnings) - 10} more")

        # Bundle status check
        status = self.bundle.get("status", "unknown")
        measured = self.bundle.get("measured_modules", 0)
        p95 = self.bundle.get("total_p95_budget_ms", 0)

        print("\n📈 Bundle Metrics:")
        print(f"  Status: {status}")
        print(f"  Measured modules: {measured}")
        print(f"  P95 budget: {p95}ms")

        if status != "ready":
            print(f"\n⚠️  Bundle status is '{status}', not 'ready'")
            if not dry_run:
                print("Cannot launch Bundle 1 until status is 'ready'")
                return False

        if not self.errors and not self.warnings:
            print("\n✅ All checks passed!")
            print("Ready to launch Bundle 1 for real!")

            if not dry_run:
                print("\n🎯 Launching Bundle 1...")
                # Import and run the real launcher
                try:
                    from launcher import JeffreyLauncher

                    launcher = JeffreyLauncher()
                    return launcher.boot()
                except ImportError:
                    print("⚠️  launcher.py not found, skipping actual launch")
                    return True

        return len(self.errors) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Test only, don't launch")
    parser.add_argument("--bundle", default="artifacts/inventory_ultimate.json", help="Bundle config file")

    args = parser.parse_args()

    launcher = DryRunLauncher(args.bundle)
    success = launcher.run(dry_run=args.dry_run)

    sys.exit(0 if success else 1)
