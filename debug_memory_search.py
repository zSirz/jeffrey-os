#!/usr/bin/env python3
"""
Debug: Pourquoi search_memories ne trouve rien ?
"""

from jeffrey.core.orchestration.agi_orchestrator import AGIOrchestrator

print("🔍 DEBUG: Pourquoi search_memories ne fonctionne pas ?")

orch = AGIOrchestrator()
mem = getattr(orch, "memory", None)

if mem:
    user_id = "life_simulation_user"

    print("\n1. ✅ Enregistrement d'un test...")
    try:
        mem.save_fact(user_id, "test", "Jeffrey est un assistant IA très intelligent")
        print("   ✅ save_fact réussi")
    except Exception as e:
        print(f"   ❌ save_fact échoué : {e}")
        exit()

    print("\n2. 🔍 Tests de recherche...")

    # Test 1: Recherche exacte
    try:
        results = mem.search_memories(user_id, "Jeffrey")
        print(f"   Test 'Jeffrey' : {len(results)} résultats")
        if results:
            for i, r in enumerate(results[:3]):
                print(f"      {i + 1}. {r}")
    except Exception as e:
        print(f"   ❌ Erreur recherche 'Jeffrey' : {e}")

    # Test 2: Recherche partielle
    try:
        results = mem.search_memories(user_id, "intelligent")
        print(f"   Test 'intelligent' : {len(results)} résultats")
    except Exception as e:
        print(f"   ❌ Erreur recherche 'intelligent' : {e}")

    # Test 3: Recherche très générale
    try:
        results = mem.search_memories(user_id, "")
        print(f"   Test chaîne vide : {len(results)} résultats")
    except Exception as e:
        print(f"   ❌ Erreur recherche vide : {e}")

    # Test 4: get_all_memories
    print("\n3. 📊 Test get_all_memories...")
    try:
        all_memories = mem.get_all_memories(user_id)
        print(f"   get_all_memories : {len(all_memories)} résultats")
        if all_memories:
            for i, m in enumerate(all_memories[:3]):
                print(f"      {i + 1}. {m}")
    except Exception as e:
        print(f"   ❌ Erreur get_all_memories : {e}")

    # Test 5: Signature de search_memories
    print("\n4. 🔧 Inspection de search_memories...")
    import inspect

    try:
        sig = inspect.signature(mem.search_memories)
        print(f"   Signature : search_memories{sig}")
    except Exception as e:
        print(f"   ❌ Erreur signature : {e}")

else:
    print("❌ Pas de memory")

print("\n✅ Debug terminé")
