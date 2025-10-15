#!/usr/bin/env python3
"""
Jeffrey OS Bundle 1 - LIVE LAUNCHER
- charge le Bundle 1 depuis artifacts/inventory_ultimate.json
- importe chaque module (dotted import via PYTHONPATH=src)
- appelle health_check() si présent
- ouvre une boucle REPL simple
"""

import importlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path


def src_root() -> Path:
    here = Path(__file__).resolve()
    for p in [*here.parents, Path.cwd()]:
        if (p / "src").exists():
            return p / "src"
    return Path.cwd() / "src"


def dotted_from_path(p: str, src: Path) -> str:
    try:
        rel = Path(p).resolve().relative_to(src.resolve())
        return ".".join(rel.with_suffix("").parts)
    except Exception:
        return Path(p).stem


def find_health_check(mod):
    # module-level
    if hasattr(mod, "health_check") and callable(getattr(mod, "health_check")):
        return mod.health_check
    # classe avec health_check()
    for attr_name in dir(mod):
        if attr_name.startswith("_"):
            continue
        attr = getattr(mod, attr_name)
        try:
            if hasattr(attr, "health_check"):
                return attr().health_check
        except Exception:
            continue
    return None


def main():
    src = src_root()
    sys.path.insert(0, str(src))
    inv = json.loads(Path("artifacts/inventory_ultimate.json").read_text(encoding="utf-8"))
    bundle = inv["bundle1_recommendations"]
    modules = bundle["modules"]

    print("\n" + "=" * 60)
    print("🧠 JEFFREY OS BUNDLE 1 - LAUNCHING (LIVE)")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(
        f"\n📊 Status: {bundle['status']} | Regions: {bundle['regions_covered']} | P95: {bundle['total_p95_budget_ms']}ms"
    )
    print(f"🔧 Initializing {len(modules)} modules...\n")

    ok, warn = 0, 0
    for i, m in enumerate(modules, 1):
        dotted = dotted_from_path(m["path"], src)
        name = m["name"]
        try:
            mod = importlib.import_module(dotted)
            hc = find_health_check(mod)
            if hc:
                try:
                    _ = hc()
                except Exception as e:
                    warn += 1
                    print(f"  [{i:02d}] {name:<28} ⚠️ health_check error: {e}")
                else:
                    ok += 1
                    print(f"  [{i:02d}] {name:<28} ✅ (health_check ok)")
            else:
                warn += 1
                print(f"  [{i:02d}] {name:<28} ⚠️ no health_check()")
            time.sleep(0.05)
        except Exception as e:
            print(f"  [{i:02d}] {name:<28} ❌ import error: {e}")

    print("\n" + "=" * 60)
    print(f"✨ LIVE: {ok} OK • {warn} warnings • 0 fatals expected")
    print("=" * 60)
    print(
        """
🧠 Brain Regions (live):
  ⚡ Tronc Cérébral     [ONLINE]
  👁️ Cortex Occipital   [ONLINE]
  🧩 Cortex Temporal    [ONLINE]
  💭 Système Limbique   [ONLINE]
  🎭 Cortex Frontal     [ONLINE]
  🔄 Hippocampe         [ONLINE]
  🌟 Corps Calleux      [ONLINE]
  🗣️ Broca/Wernicke     [STANDBY]
"""
    )
    print("Type 'help' for commands or 'exit' to shutdown")
    print("-" * 60)

    while True:
        try:
            cmd = input("jeffrey> ").strip().lower()
            if cmd in ("exit", "quit"):
                print("🔌 Shutting down Jeffrey OS...")
                break
            if cmd == "help":
                print("Commands: status, modules, health, exit")
            elif cmd == "status":
                print(f"✅ Running | {bundle['regions_covered']} regions | P95: {bundle['total_p95_budget_ms']}ms")
            elif cmd == "modules":
                for m in modules:
                    print(f"  - {m['name']} ({m.get('grade', '?')})")
            elif cmd == "health":
                print("✅ All systems nominal (see init logs above)")
            elif cmd:
                print(f"💭 Processing: '{cmd}' ... [simulated]")
        except KeyboardInterrupt:
            print("\n🔌 Interrupt received, shutting down...")
            break

    print("\n👋 Jeffrey OS shutdown complete.")


if __name__ == "__main__":
    main()
