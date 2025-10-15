#!/usr/bin/env python3
"""Environment checker for P1 deployment"""

import importlib
import platform
import sys


def check_environment():
    """Verify environment is ready for P1"""
    results = {"python_version": sys.version, "platform": platform.system(), "checks": {}}

    # Check Python version
    if sys.version_info < (3, 9):
        results["checks"]["python"] = "❌ Python 3.9+ required"
    else:
        results["checks"]["python"] = "✅ Python OK"

    # Check critical imports
    required = ["pydantic", "asyncio", "ast"]
    optional = ["sklearn", "rich", "cryptography", "psutil"]

    for module in required:
        try:
            importlib.import_module(module)
            results["checks"][module] = "✅ Installed"
        except ImportError:
            results["checks"][module] = "❌ Missing (REQUIRED)"

    for module in optional:
        try:
            importlib.import_module(module)
            results["checks"][module] = "✅ Installed"
        except ImportError:
            results["checks"][module] = "⚠️ Missing (optional)"

    # Check Pydantic version
    try:
        import pydantic

        version = getattr(pydantic, "__version__", "unknown")
        major = int(version.split(".")[0]) if version != "unknown" else 0
        results["checks"]["pydantic_version"] = f"v{version} ({'✅ OK' if major >= 1 else '⚠️ v1+ required'})"
    except:
        pass

    # Check file locking capability
    if platform.system() == "Windows":
        results["checks"]["file_locking"] = "⚠️ Windows - using msvcrt"
    else:
        results["checks"]["file_locking"] = "✅ Unix - using fcntl"

    return results


if __name__ == "__main__":
    results = check_environment()
    print(f"Platform: {results['platform']}")
    print(f"Python: {results['python_version']}")
    print("\nEnvironment checks:")
    for check, status in results["checks"].items():
        print(f"  {check}: {status}")

    # Exit code based on required checks - GPT fix for explicit check
    required_ok = (
        "✅" in results["checks"].get("python", "")
        and "✅" in results["checks"].get("pydantic", "")
        and "✅" in results["checks"].get("asyncio", "")
        and "✅" in results["checks"].get("ast", "")
    )
    sys.exit(0 if required_ok else 1)
