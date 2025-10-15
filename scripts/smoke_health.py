#!/usr/bin/env python3
"""Smoke test live pour Bundle 1 - Import + health_check"""

import importlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTHONPATH", str(Path.cwd() / "src"))
sys.path.insert(0, os.environ["PYTHONPATH"])

# Suppression des warnings
import warnings

warnings.filterwarnings("ignore")


def dotted_from(path: str, src: Path) -> str:
    """Convertir un path en module dotted notation"""
    try:
        rel = Path(path).resolve().relative_to(src.resolve())
        return ".".join(rel.with_suffix("").parts)
    except:
        # Fallback
        parts = Path(path).with_suffix("").parts
        if "src" in parts:
            idx = parts.index("src")
            return ".".join(parts[idx + 1 :])
        return ".".join(parts[-3:])


def main():
    print("\n" + "=" * 60)
    print("🩺 JEFFREY OS BUNDLE 1 - SMOKE TEST LIVE")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")

    inv = json.load(open("artifacts/inventory_ultimate.json"))
    mods = inv["bundle1_recommendations"]["modules"]
    src = Path(os.environ["PYTHONPATH"])

    ok, warn, err = 0, 0, 0
    results = []

    print("\n📋 Testing 10 Bundle 1 modules...\n")

    for m in mods:
        name = m["name"]
        start = time.time()

        try:
            dotted = dotted_from(m["path"], src)
            mod = importlib.import_module(dotted)
            import_time = (time.time() - start) * 1000

            hc = getattr(mod, "health_check", None)
            if callable(hc):
                try:
                    hc_start = time.time()
                    res = hc() or {}
                    hc_time = (time.time() - hc_start) * 1000
                    status = (res.get("status") or "ok").lower()

                    if status == "healthy":
                        print(f"  ✅ {name:<30} import={import_time:.1f}ms health={hc_time:.1f}ms")
                        ok += 1
                        results.append(
                            {
                                "module": name,
                                "status": "ok",
                                "times": {"import": import_time, "health": hc_time},
                            }
                        )
                    else:
                        print(f"  ⚠️  {name:<30} health status: {status}")
                        warn += 1
                        results.append({"module": name, "status": "warning", "reason": f"health={status}"})
                except Exception as e:
                    print(f"  ⚠️  {name:<30} health_check() raised: {str(e)[:30]}")
                    warn += 1
                    results.append({"module": name, "status": "warning", "reason": "health_check error"})
            else:
                print(f"  ⚠️  {name:<30} no health_check() function")
                warn += 1
                results.append({"module": name, "status": "warning", "reason": "no health_check"})
        except Exception as e:
            print(f"  ❌ {name:<30} import failed: {str(e)[:50]}")
            err += 1
            results.append({"module": name, "status": "error", "reason": str(e)})

    # Summary
    print("\n" + "-" * 60)
    print(f"📊 Summary: ✅ OK={ok}  ⚠️ WARN={warn}  ❌ ERR={err}")

    # Performance stats
    total_import = sum(r.get("times", {}).get("import", 0) for r in results if "times" in r)
    total_health = sum(r.get("times", {}).get("health", 0) for r in results if "times" in r)
    print(f"⏱️  Total import time: {total_import:.1f}ms")
    print(f"⏱️  Total health time: {total_health:.1f}ms")
    print(f"⚡ Total boot time: {total_import + total_health:.1f}ms")

    # Save results
    with open("artifacts/smoke_test_results.json", "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "summary": {"ok": ok, "warn": warn, "err": err},
                "performance": {
                    "total_import_ms": total_import,
                    "total_health_ms": total_health,
                    "total_boot_ms": total_import + total_health,
                },
                "modules": results,
            },
            f,
            indent=2,
        )

    print("\n✅ Results saved to artifacts/smoke_test_results.json")
    print("=" * 60)

    sys.exit(1 if err else 0)


if __name__ == "__main__":
    main()
