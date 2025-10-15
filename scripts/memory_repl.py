#!/usr/bin/env python3
"""REPL interactif pour tester la mémoire persistante"""

import asyncio
import json
import sys
from datetime import datetime

sys.path.insert(0, "src")
from jeffrey.bundle2.memory.sqlite_store import SQLiteMemoryStore


async def main():
    store = SQLiteMemoryStore()

    print("\n" + "=" * 60)
    print("🧠 JEFFREY MEMORY REPL - Bundle 2 Preview")
    print("=" * 60)
    print("Commands: store <text>, recall <id>, search <query>, stats, exit")
    print("")

    while True:
        try:
            cmd = input("memory> ").strip()

            if not cmd:
                continue

            if cmd.lower() in ["exit", "quit"]:
                print("👋 Mémoires sauvegardées. Au revoir!")
                break

            elif cmd.lower() == "stats":
                stats = await store.consolidate()
                print(f"📊 Stats: {json.dumps(stats, indent=2)}")

            elif cmd.startswith("store "):
                content = cmd[6:]
                mid = await store.store(
                    content,
                    metadata={
                        "timestamp": datetime.now().isoformat(),
                        "source": "repl",
                        "importance": 0.5,
                    },
                )
                print(f"✅ Stored as {mid}")

            elif cmd.startswith("recall "):
                mid = cmd[7:]
                memory = await store.recall(mid)
                if memory:
                    print(f"🔍 Found: {json.dumps(memory, indent=2)}")
                else:
                    print("❌ Memory not found")

            elif cmd.startswith("search "):
                query = cmd[7:]
                results = await store.search(query, limit=5)
                print(f"🔍 Found {len(results)} memories:")
                for r in results:
                    print(f"  - {r['id']}: {str(r['content'])[:60]}...")

            else:
                print("❓ Unknown command. Try: store, recall, search, stats, exit")

        except KeyboardInterrupt:
            print("\n👋 Arrêt...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
