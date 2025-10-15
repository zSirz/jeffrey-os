#!/usr/bin/env python3
"""
Debug: Pourquoi l'enregistrement mémoire ne se déclenche pas ?
"""

from jeffrey.core.orchestration.agi_orchestrator import AGIOrchestrator

print("🔍 DEBUG: Pourquoi pas d'enregistrement mémoire ?")

orch = AGIOrchestrator()

print(f"\n1. memory_v2_interface existe ? {hasattr(orch, 'memory_v2_interface')}")
if hasattr(orch, 'memory_v2_interface'):
    mem_v2 = getattr(orch, 'memory_v2_interface')
    print(f"   Type : {type(mem_v2).__name__}")
    print(f"   a save_fact ? {hasattr(mem_v2, 'save_fact')}")

print(f"\n2. memory existe ? {hasattr(orch, 'memory')}")
if hasattr(orch, 'memory'):
    mem = getattr(orch, 'memory')
    print(f"   Type : {type(mem).__name__}")
    print(f"   a save_fact ? {hasattr(mem, 'save_fact')}")

# Test exact du code dans life simulation
print("\n3. Test exact du code :")
mem_store = getattr(orch, "memory_v2_interface", None)
if mem_store is None:
    mem_store = getattr(orch, "memory", None)

print(f"   mem_store final : {type(mem_store).__name__ if mem_store else 'None'}")
print(f"   mem_store and hasattr(mem_store, 'save_fact') : {mem_store and hasattr(mem_store, 'save_fact')}")

if mem_store:
    print("   Méthodes de mem_store :")
    methods = [m for m in dir(mem_store) if not m.startswith('_') and 'save' in m.lower()]
    for m in methods:
        print(f"      - {m}")

print("\n✅ Diagnostic terminé")
