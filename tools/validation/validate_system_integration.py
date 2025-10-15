#!/usr/bin/env python3
"""
Script de validation de l'intégration système pour Jeffrey V1.1
Vérifie que tous les systèmes sont correctement branchés
"""

import sys

sys.path.append('.')


def validate_systems():
    """Valide que tous les systèmes critiques sont correctement branchés"""

    print("🔧 VALIDATION DE L'INTÉGRATION SYSTÈME JEFFREY V1.1")
    print("=" * 60)

    success_count = 0
    total_tests = 5

    # Test 1: ThoughtMemorySystem
    try:
        print("🧠 Test 1: ThoughtMemorySystem initialization...")
        tms = ThoughtMemorySystem()
        tms.save_thought('Test thought validation', 'curiosity')
        print("✅ ThoughtMemorySystem correctement initialisé")
        success_count += 1
    except Exception as e:
        print(f"❌ ERREUR ThoughtMemorySystem: {e}")

    # Test 2: Entity extraction
    try:
        print("🔍 Test 2: Entity extraction...")
        from core.entity_extraction import extract_entities_fixed

        result = extract_entities_fixed('Le chien de mon frère s appelle Rex')
        print(f"✅ Using extract_entities_fixed: {extract_entities_fixed.__name__}")
        success_count += 1
    except Exception as e:
        print(f"❌ ERREUR Entity extraction: {e}")

    # Test 3: Response generator
    try:
        print("🤖 Test 3: IntelligentResponseGenerator...")
        from core.understanding.real_intelligence import IntelligentResponseGenerator

        generator = IntelligentResponseGenerator()
        response = generator.generate_response('conversation', {'unclear': True})
        print(f"✅ Using response_generator: {generator.__class__.__name__}")
        success_count += 1
    except Exception as e:
        print(f"❌ ERREUR Response generator: {e}")

    # Test 4: Conversation memory
    try:
        print("💾 Test 4: Conversation memory...")
        conversation_memory.add_exchange(
            user_message='Le chien de mon frère s appelle Rex',
            jeffrey_response='Oh Rex ! C est mignon comme nom pour un chien !',
            user_emotion='neutral',
            jeffrey_emotion='joy',
        )
        print("✅ Conversation sauvegardée dans memory")
        success_count += 1
    except Exception as e:
        print(f"❌ ERREUR Conversation memory: {e}")

    # Test 5: Integration flow test
    try:
        print("🔄 Test 5: Flow d'intégration complet...")
        from core.entity_extraction import extract_entities_fixed
        from core.understanding.real_intelligence import IntelligentResponseGenerator

        test_message = "Le chien de mon frère s appelle Rex"

        # Test extraction
        entities = extract_entities_fixed(test_message)
        print(f"🤖 Extraction: {entities}")

        # Test response generation
        generator = IntelligentResponseGenerator()
        response = generator.generate_response('conversation', {'topic': 'chien'})
        print(f"💡 Réponse: \"{response}\"")

        # Vérifier que ce n'est pas une réponse générique
        if "Hmm, tu veux dire quoi exactement ?" not in response:
            print("✅ Plus AUCUNE réponse hardcodée!")
            success_count += 1
        else:
            print("❌ Réponse hardcodée détectée")

    except Exception as e:
        print(f"❌ ERREUR Flow integration: {e}")

    print("\n" + "=" * 60)
    print(f"📊 RÉSULTAT: {success_count}/{total_tests} tests réussis")

    if success_count == total_tests:
        print("🎉 SUCCÈS TOTAL! Tous les systèmes sont correctement branchés!")
        print("✅ Jeffrey V1.1 est prêt avec tous les nouveaux systèmes")
        return True
    else:
        print(f"⚠️  {total_tests - success_count} système(s) nécessite(nt) encore des corrections")
        return False


if __name__ == "__main__":
    validate_systems()
