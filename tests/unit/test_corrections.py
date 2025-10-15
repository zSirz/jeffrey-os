#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de test pour valider les corrections du projet Jeffrey
"""

import sys
import os

# Ajouter le répertoire racine au PYTHONPATH
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

print(f"🔧 Test des corrections Jeffrey")
print(f"📁 Répertoire de base : {BASE_DIR}")
print("=" * 60)


    def test_emotional_core():
    """Test du cœur émotionnel"""
    print("\n🧠 Test JeffreyEmotionalCore...")

        try:
        from Orchestrateur_IA.core.jeffrey_emotional_core import JeffreyEmotionalCore

        # Créer une instance
        emotional_core = JeffreyEmotionalCore()
        print("✅ JeffreyEmotionalCore instancié avec succès")

        # Test de la méthode detecter_emotion
        emotions = emotional_core.detecter_emotion("Je suis très heureux aujourd'hui!")
        print(f"✅ detecter_emotion fonctionne: {emotions}")

        # Test de la nouvelle méthode get_meteo_interieure
        meteo = emotional_core.get_meteo_interieure()
        print(f"✅ get_meteo_interieure fonctionne: {meteo}")

        # Test update_emotional_state
        emotional_core.update_emotional_state(emotions)
        print("✅ update_emotional_state fonctionne")

        # Test get_emotional_response_modifier
        modifiers = emotional_core.get_emotional_response_modifier()
        print(f"✅ get_emotional_response_modifier fonctionne: {modifiers}")

        # Test de la nouvelle méthode generer_phrase_cadeau_emotionnelle
        phrase = emotional_core.generer_phrase_cadeau_emotionnelle(
            emotion="joie", intensite=0.8, souvenirs=["Tu es formidable"]
        )
        print(f"✅ generer_phrase_cadeau_emotionnelle fonctionne: {phrase}")

        print("🧠 ✅ Tous les tests du cœur émotionnel RÉUSSIS")
            return True

            except Exception as e:
        print(f"🧠 ❌ Erreur dans JeffreyEmotionalCore: {e}")
        import traceback

        traceback.print_exc()
                return False


                def test_unified_memory():
    """Test de la mémoire unifiée"""
    print("\n💾 Test JeffreyUnifiedMemory...")

                    try:
        from core.jeffrey_memory_integration import JeffreyUnifiedMemory
        from Orchestrateur_IA.core.jeffrey_emotional_core import JeffreyEmotionalCore

        # Créer les instances
        emotional_core = JeffreyEmotionalCore()
        memory = JeffreyUnifiedMemory()
        print("✅ JeffreyUnifiedMemory instancié avec succès")

        # Test connect_emotional_core
        memory.connect_emotional_core(emotional_core)
        print("✅ connect_emotional_core fonctionne")

        # Test process_interaction
        memory_id = memory.process_interaction(
            "Bonjour Jeffrey!", "Bonjour ! Je suis ravi de te parler !"
        )
        print(f"✅ process_interaction fonctionne: {memory_id}")

        # Test consolidate_memories (évite l'erreur KeyError 'conversation')
        report = memory.consolidate_memories()
        print(
            f"✅ consolidate_memories fonctionne: {report.get('consolidation_status', 'unknown')}"
        )

        # Test clean_context (évite l'erreur KeyError 'conversations')
        success = memory.clean_context()
        print(f"✅ clean_context fonctionne: {success}")

        # Test get_relevant_memories
        memories = memory.get_relevant_memories("test query")
        print(f"✅ get_relevant_memories fonctionne: {len(memories)} mémoires trouvées")

        print("💾 ✅ Tous les tests de mémoire unifiée RÉUSSIS")
                        return True

                        except Exception as e:
        print(f"💾 ❌ Erreur dans JeffreyUnifiedMemory: {e}")
        import traceback

        traceback.print_exc()
                            return False


                            def test_imports():
    """Test des imports problématiques"""
    print("\n📦 Test des imports...")

    imports_to_test = [
        "core.jeffrey_emotional_core.JeffreyEmotionalCore",
        "core.jeffrey_memory_integration.JeffreyUnifiedMemory",
        "core.personality.conversation_personality.ConversationPersonality",
        "core.voice.jeffrey_voice_system.JeffreyVoiceSystem",
        "core.emotions.emotional_sync.EmotionalSync",
        "core.humeur_detector.HumeurDetector",
    ]

    success_count = 0

                                for import_path in imports_to_test:
                                    try:
            module_path, class_name = import_path.rsplit(".", 1)
            module = __import__(module_path, fromlist=[class_name])
            getattr(module, class_name)
            print(f"✅ {import_path}")
            success_count += 1
                                        except Exception as e:
            print(f"❌ {import_path}: {e}")

    print(f"📦 {success_count}/{len(imports_to_test)} imports réussis")
                                            return success_count == len(imports_to_test)


                                            def test_integration():
    """Test d'intégration simple"""
    print("\n🔗 Test d'intégration...")

                                                try:
        # Simuler l'initialisation comme dans lancer_jeffrey_chat.py
        from Orchestrateur_IA.core.jeffrey_emotional_core import JeffreyEmotionalCore
        from core.jeffrey_memory_integration import JeffreyUnifiedMemory

        # Créer les composants principaux
        emotional_core = JeffreyEmotionalCore(memory_path="Jeffrey_Memoire")
        unified_memory = JeffreyUnifiedMemory(base_path="Jeffrey_Memoire")
        unified_memory.connect_emotional_core(emotional_core)

        # Test d'une séquence d'interaction complète
        user_message = "Je me sens un peu triste aujourd'hui"

        # 1. Détection émotionnelle
        emotions = emotional_core.detecter_emotion(user_message)
        emotional_core.update_emotional_state(emotions)

        # 2. Génération de la météo intérieure
        meteo = emotional_core.get_meteo_interieure()

        # 3. Enregistrement en mémoire
        jeffrey_response = f"Je comprends, {meteo['meteo']}. Je suis là pour toi."
        memory_id = unified_memory.process_interaction(user_message, jeffrey_response)

        print(f"✅ Test d'intégration réussi:")
        print(f"   Émotions détectées: {emotions}")
        print(f"   Météo intérieure: {meteo['meteo']}")
        print(f"   Mémoire sauvegardée: {memory_id}")

                                                    return True

                                                    except Exception as e:
        print(f"🔗 ❌ Erreur dans le test d'intégration: {e}")
        import traceback

        traceback.print_exc()
                                                        return False


                                                        def main():
    """Point d'entrée principal des tests"""
    print("🚀 Début des tests de correction...")

    tests = [
        ("Imports", test_imports),
        ("Cœur émotionnel", test_emotional_core),
        ("Mémoire unifiée", test_unified_memory),
        ("Intégration", test_integration),
    ]

    results = []

                                                            for test_name, test_func in tests:
                                                                try:
            result = test_func()
            results.append((test_name, result))
                                                                    except Exception as e:
            print(f"💥 Erreur fatale dans {test_name}: {e}")
            results.append((test_name, False))

    # Résumé des résultats
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 60)

    success_count = 0
                                                                        for test_name, success in results:
        status = "✅ RÉUSSI" if success else "❌ ÉCHEC"
        print(f"{test_name:20} : {status}")
                                                                            if success:
            success_count += 1

    print(f"\n🎯 {success_count}/{len(results)} tests réussis")

                                                                                if success_count == len(results):
        print("🎉 TOUTES LES CORRECTIONS VALIDÉES AVEC SUCCÈS !")
        print("✨ Jeffrey est prêt à fonctionner !")
                                                                                    else:
        print("⚠️  Il reste des problèmes à corriger.")

                                                                                        return success_count == len(results)


                                                                                        if __name__ == "__main__":
    main()
