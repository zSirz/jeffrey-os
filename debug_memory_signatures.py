#!/usr/bin/env python3
"""
Script pour analyser les signatures de save_fact et save_persistent_data
"""

import inspect

from jeffrey.core.orchestration.agi_orchestrator import AGIOrchestrator

print("=" * 80)
print("🔍 ANALYSE DES SIGNATURES MÉMOIRE")
print("=" * 80)

orch = AGIOrchestrator()
mem = getattr(orch, 'memory', None)

if mem:
    print(f"✅ memory existe : {type(mem).__name__}")

    # Analyser save_fact
    if hasattr(mem, 'save_fact'):
        print("\n📋 SIGNATURE save_fact :")
        method = getattr(mem, 'save_fact')
        try:
            sig = inspect.signature(method)
            print(f"   save_fact{sig}")

            # Doc
            if method.__doc__:
                print("\n   📖 Documentation :")
                for line in method.__doc__.strip().split('\n')[:10]:
                    print(f"      {line}")
        except Exception as e:
            print(f"   ⚠️ Erreur : {e}")

    # Analyser save_persistent_data
    if hasattr(mem, 'save_persistent_data'):
        print("\n📋 SIGNATURE save_persistent_data :")
        method = getattr(mem, 'save_persistent_data')
        try:
            sig = inspect.signature(method)
            print(f"   save_persistent_data{sig}")

            # Doc
            if method.__doc__:
                print("\n   📖 Documentation :")
                for line in method.__doc__.strip().split('\n')[:10]:
                    print(f"      {line}")
        except Exception as e:
            print(f"   ⚠️ Erreur : {e}")

    # Test fonctionnel
    print("\n🧪 TEST FONCTIONNEL")

    # Test save_fact
    if hasattr(mem, 'save_fact'):
        print("\n🔬 Test save_fact :")
        try:
            result = mem.save_fact("Test conversation Jeffrey", category="test")
            print(f"   ✅ SUCCÈS save_fact ! Retour : {result}")

            # Vérifier si c'est maintenant en mémoire
            memories = mem.search_memories("Test conversation")
            print(f"   📋 Relecture : {len(memories)} souvenirs trouvés")
            if memories:
                print(f"      Premier souvenir : {memories[0]}")

        except Exception as e:
            print(f"   ⚠️ Échec save_fact : {e}")
            import traceback

            traceback.print_exc()

    # Test save_persistent_data
    if hasattr(mem, 'save_persistent_data'):
        print("\n🔬 Test save_persistent_data :")
        try:
            result = mem.save_persistent_data("Test persistent", "test_key")
            print(f"   ✅ SUCCÈS save_persistent_data ! Retour : {result}")

        except Exception as e:
            print(f"   ⚠️ Échec save_persistent_data : {e}")

    # Compter le total en mémoire maintenant
    try:
        all_memories = mem.get_all_memories()
        print(f"\n📊 Total en mémoire après tests : {len(all_memories)} souvenirs")
    except Exception as e:
        print(f"   ⚠️ Impossible de compter : {e}")

print("\n" + "=" * 80)
print("✅ Analyse terminée !")
print("=" * 80)
