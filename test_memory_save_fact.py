#!/usr/bin/env python3
"""
Test pour save_fact avec les bons paramètres
"""

from jeffrey.core.orchestration.agi_orchestrator import AGIOrchestrator

print("🧪 TEST save_fact avec bons paramètres")

orch = AGIOrchestrator()
mem = getattr(orch, 'memory', None)

if mem:
    try:
        # Bon ordre: user_id, category, fact
        result = mem.save_fact("user_test", "conversation", "Je me sens joyeux aujourd'hui")
        print(f"✅ SUCCÈS save_fact ! Retour : {result}")

        # Vérifier dans la mémoire
        memories = mem.search_memories("joyeux")
        print(f"📋 Souvenirs trouvés : {len(memories)}")

        if memories:
            for i, memory in enumerate(memories):
                print(f"   {i + 1}. {memory}")

        # Compter total
        try:
            all_memories = mem.get_all_memories("user_test")
            print(f"📊 Total mémoires pour user_test : {len(all_memories)}")
        except Exception as e:
            print(f"⚠️ get_all_memories erreur : {e}")

    except Exception as e:
        print(f"❌ Erreur : {e}")
        import traceback

        traceback.print_exc()
else:
    print("❌ Pas de memory")

print("\n✅ Test terminé !")
