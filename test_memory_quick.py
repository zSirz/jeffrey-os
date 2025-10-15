#!/usr/bin/env python3
"""
Test rapide pour vérifier l'enregistrement mémoire
"""

from jeffrey.core.orchestration.agi_orchestrator import AGIOrchestrator

print("🚀 TEST RAPIDE ENREGISTREMENT MÉMOIRE")

orch = AGIOrchestrator()

# Même logique que dans life simulation
mem_store = getattr(orch, "memory", None)
if mem_store is None:
    mem_store = getattr(orch, "memory_v2_interface", None)

print(f"mem_store sélectionné : {type(mem_store).__name__}")
print(f"a save_fact ? {hasattr(mem_store, 'save_fact') if mem_store else False}")

if mem_store and hasattr(mem_store, 'save_fact'):
    print("\n✅ TEST ENREGISTREMENT :")
    try:
        user_id = "life_simulation_user"
        category = "test"
        fact = "Message: Salut Jeffrey ! | Émotion: joie"

        mem_store.save_fact(user_id, category, fact)
        print("✅ Enregistrement réussi !")

        # Vérifier
        memories = mem_store.search_memories(user_id, "Salut")
        print(f"📋 {len(memories)} souvenirs trouvés")

    except Exception as e:
        print(f"❌ Erreur : {e}")
else:
    print("❌ Pas d'enregistrement possible")

print("\n✅ Test terminé")
