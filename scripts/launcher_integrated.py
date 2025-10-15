#!/usr/bin/env python3
"""Jeffrey OS Launcher - Version ULTIMATE avec monitoring complet"""

import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime

os.environ["KIVY_NO_FILELOG"] = "1"
os.environ["KIVY_LOG_LEVEL"] = "error"
os.environ["KIVY_NO_CONSOLELOG"] = "1"
sys.path.insert(0, "src")

print("\n" + "=" * 60)
print("🧠 JEFFREY OS - CONSCIOUSNESS LOOP ULTIMATE")
print("=" * 60)
print("Version: Production Final")
print(f"Timestamp: {datetime.now().isoformat()}")

try:
    from jeffrey.core.consciousness_loop import ConsciousnessLoop

    loop = ConsciousnessLoop()
    print("✅ Consciousness Loop imported")
except ImportError as e:
    print(f"❌ Failed to import: {e}")
    sys.exit(1)

global_stats = defaultdict(list)


async def interactive_repl():
    print("\n🔧 Initializing consciousness loop...")
    await loop.initialize()

    health = loop.health_check()
    print("\n📊 System Status:")
    print(f"  Regions active: {health['regions_active']}")
    print(f"  Real modules: {health['real_modules']}")
    print(f"  Stub modules: {health['stub_modules']}")
    print(f"  Module failures: {health.get('module_failures', {})}")

    if health["stub_modules"] > health["real_modules"]:
        print("  ⚠️  WARNING: More stubs than real modules!")

    print("\n" + "=" * 60)
    print("💬 Jeffrey Consciousness Loop Active")
    print("Commands: help, status, regions, stats, perf, cache, exit")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("jeffrey> ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "bye"]:
                print("👋 Au revoir! Mémoires sauvegardées.")
                break

            elif user_input.lower() == "help":
                print(
                    """
Commands:
  help     - Show this help
  status   - System status
  regions  - Brain regions detail
  stats    - Performance statistics
  perf     - Performance breakdown
  cache    - Memory cache status
  health   - Full health check
  exit     - Quit

Just type to chat with Jeffrey!
"""
                )

            elif user_input.lower() == "status":
                health = loop.health_check()
                perf = health.get("performance", {})
                print(
                    f"""
Status: {health['status']}
Uptime: {health['uptime_seconds']:.0f}s
Cycles: {health['cycles_completed']}
Regions: {health['regions_active']} ({health['real_modules']} real, {health['stub_modules']} stubs)
Failures: {len(health.get('module_failures', {}))} modules with issues
Avg: {perf.get('avg_latency_ms', 0):.2f}ms
P95: {perf.get('p95_latency_ms', 0):.2f}ms
Cache: {perf.get('cache_size', 0)} memories
"""
                )

            elif user_input.lower() == "health":
                health = loop.health_check()
                print(json.dumps(health, indent=2))

            else:
                print("🧠 Processing...", end="", flush=True)
                start = time.time()

                result = await loop.process_input(user_input)
                loop.last_result = result

                elapsed = (time.time() - start) * 1000
                print(f" ({elapsed:.0f}ms)")

                global_stats["latencies"].append(elapsed)

                if result["success"]:
                    print(f"\n{result['response']}\n")

                    if os.environ.get("JEFFREY_DEBUG"):
                        print("[Debug]")
                        print(f"  Intent: {result['comprehension'].get('intent')}")
                        print(f"  Memories: {result['memory']['related_memories']}")
                        print(f"  Cache: {'HIT' if result['memory'].get('from_cache') else 'MISS'}")
                else:
                    print(f"\n❌ {result.get('response', 'Erreur')}\n")

        except KeyboardInterrupt:
            print("\n👋 Interruption...")
            break
        except Exception as e:
            print(f"\n❌ Erreur: {e}\n")


def main():
    try:
        asyncio.run(interactive_repl())
    except KeyboardInterrupt:
        print("\n👋 Arrêt propre...")
    except Exception as e:
        print(f"❌ Erreur fatale: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
