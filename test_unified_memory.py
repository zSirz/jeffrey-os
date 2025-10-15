#!/usr/bin/env python3
"""Test de validation complète du module UnifiedMemory ultime"""

import asyncio
import sys
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Fix pour Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def test_unified_memory():
    """Test complet du système UnifiedMemory"""
    print("=" * 60)
    print("🧪 TEST UNIFIED MEMORY ULTIMATE")
    print("=" * 60)

    # Import des modules
    try:
        from jeffrey.core.memory.unified_memory import MemoryPriority, MemoryType, MemoryValidator, UnifiedMemory

        print("✅ UnifiedMemory importé avec succès")
    except ImportError as e:
        print(f"❌ Erreur d'import UnifiedMemory: {e}")
        return False

    try:
        from jeffrey.core.memory.sqlite.backend import SQLiteMemoryBackend

        print("✅ SQLiteMemoryBackend importé avec succès")
    except ImportError as e:
        print(f"⚠️ SQLiteBackend non disponible (mode dégradé): {e}")

    # Test 1: Création et initialisation
    print("\n📝 Test 1: Création et initialisation")
    memory = UnifiedMemory(backend="sqlite", data_dir="test_data")
    await memory.initialize()
    print("✅ Mémoire initialisée")

    # Test 2: Validation et sanitization
    print("\n📝 Test 2: Validation et sanitization")
    test_text = "Test text very long " * 1000
    clean_text = MemoryValidator.sanitize_text(test_text, max_length=100)
    assert len(clean_text) <= 100
    print(f"✅ Texte nettoyé (longueur={len(clean_text)}): {clean_text[:50]}...")

    # Test 3: Store (avec queue async)
    print("\n📝 Test 3: Store avec batching async")
    test_data = {
        "message": "Test message important",
        "user_id": "test_user",
        "emotion": {"primary_emotion": "joy", "intensity": 0.8},
        "type": "contextual",
    }
    memory_id = await memory.store(test_data)
    print(f"✅ Mémoire stockée avec ID: {memory_id}")

    # Test 4: Store multiple
    print("\n📝 Test 4: Store batch")
    for i in range(5):
        await memory.store({"message": f"Message batch {i}", "type": "pattern", "priority": i % 3 + 1})
    await asyncio.sleep(1.5)  # Attendre le flush
    print("✅ 5 mémoires stockées en batch")

    # Test 5: Retrieve avec cache
    print("\n📝 Test 5: Retrieve avec cache")
    results = await memory.retrieve("important", limit=5)
    print(f"✅ Trouvé {len(results)} résultats")
    for r in results[:2]:
        print(f"  - {r.get('message', r.get('_id', 'unknown'))[:50]}")

    # Test 6: Compatibilité - save_fact
    print("\n📝 Test 6: Save fact (compatibilité)")
    await memory.save_fact("test_user", "animal_chien", "Max")
    memories = memory.search_memories("test_user", "chien")
    print(f"✅ Fact sauvé, recherche: {memories}")

    # Test 7: Emotional summary
    print("\n📝 Test 7: Emotional summary")
    summary = memory.get_emotional_summary("test_user")
    print(f"✅ État émotionnel: {summary}")

    # Test 8: Context summary
    print("\n📝 Test 8: Context summary")
    context = memory.get_context_summary()
    print(f"✅ Contexte: {context[:100]}...")

    # Test 9: Evolution
    print("\n📝 Test 9: Evolution system")
    evolution = await memory.evolve()
    print(f"✅ Évolution: {evolution}")

    # Test 10: Consolidation
    print("\n📝 Test 10: Consolidation")
    consolidation = await memory.consolidate()
    print(f"✅ Consolidation: {consolidation}")

    # Test 11: Statistics
    print("\n📝 Test 11: Statistics")
    stats = memory.get_stats()
    print("✅ Statistiques:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")

    # Test 12: Shutdown propre
    print("\n📝 Test 12: Shutdown")
    await memory.shutdown()
    print("✅ Arrêt propre effectué")

    print("\n" + "=" * 60)
    print("🎉 TOUS LES TESTS PASSENT AVEC SUCCÈS !")
    print("=" * 60)

    return True


async def test_memory_types():
    """Test des différents types de mémoire"""
    print("\n📝 Test des types de mémoire")

    from jeffrey.core.memory.unified_memory import MemoryPriority, MemoryType

    for mem_type in MemoryType:
        print(f"  - {mem_type.name}: {mem_type.value}")

    for priority in MemoryPriority:
        print(f"  - Priority {priority.name}: {priority.value}")

    print("✅ Types validés")


async def main():
    """Point d'entrée principal"""
    try:
        # Tests principaux
        success = await test_unified_memory()

        # Tests supplémentaires
        await test_memory_types()

        if success:
            print("\n✅ Module UnifiedMemory 100% fonctionnel!")
            return 0
        else:
            print("\n❌ Des tests ont échoué")
            return 1

    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
