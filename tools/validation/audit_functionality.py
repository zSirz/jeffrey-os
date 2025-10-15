#!/usr/bin/env python3
"""
🧪 AUDIT FONCTIONNALITÉ RÉELLE JEFFREY V2
Teste chaque module pour déterminer ce qui fonctionne vraiment
"""

import asyncio
import importlib
import os
import time

# Configuration pour bypass SecureConfig
os.environ['JEFFREY_BYPASS_SECURECONFIG'] = 'true'


def test_jeffrey_core():
    """Test du système core Jeffrey"""
    print('\n1. TEST INSTANCIATION JEFFREY CORE')
    try:
        from jeffrey_v2_integration import JeffreyV2

        start_time = time.time()
        jeffrey = JeffreyV2()
        init_time = time.time() - start_time
        status = jeffrey.get_status()

        print(f'✅ Instanciation réussie en {init_time:.2f}s')
        print(f'📊 Modules: {sum(status["modules_loaded"].values())}/{len(status["modules_loaded"])}')

        for module, loaded in status['modules_loaded'].items():
            icon = '✅' if loaded else '❌'
            print(f'   {icon} {module}')

        return jeffrey, status

    except Exception as e:
        print(f'❌ Échec instanciation: {e}')
        return None, {}


def test_voice_system():
    """Test du système vocal"""
    print('\n2. TEST SYSTÈME VOCAL')
    try:
        from core.voice.elevenlabs_v3_engine import ElevenLabsV3Engine

        voice = ElevenLabsV3Engine()

        # Test connectivité API
        has_api_key = hasattr(voice, 'api_key') and voice.api_key
        print(f'   API Key: {"✅" if has_api_key else "❌"}')

        # Test méthodes disponibles
        can_synthesize = hasattr(voice, 'synthesize_speech')
        can_get_voices = hasattr(voice, 'get_available_voices')

        print(f'   Synthèse speech: {"✅" if can_synthesize else "❌"}')
        print(f'   Liste voix: {"✅" if can_get_voices else "❌"}')

        # Test réel si possible
        if can_get_voices:
            try:
                voices = voice.get_available_voices()
                print(f'✅ ElevenLabs connecté: {len(voices)} voix disponibles')
                return True
            except Exception as e:
                print(f'⚠️ API non accessible: {str(e)[:50]}...')
                return False

        return can_synthesize and can_get_voices

    except Exception as e:
        print(f'❌ Système vocal défaillant: {e}')
        return False


def test_emotional_core():
    """Test du cœur émotionnel"""
    print('\n3. TEST CŒUR ÉMOTIONNEL')
    try:
        from core.agi_fusion.emotional_core import EmotionalCore

        emotion_core = EmotionalCore()

        # Test analyse émotionnelle
        test_inputs = [
            "Je suis très heureux!",
            "C'est frustrant cette situation",
            "Texte neutre sans émotion particulière",
        ]

        results = []
        for text in test_inputs:
            try:
                result = emotion_core.analyze_and_resonate(text, {})
                results.append(result)
                print(f'   ✅ Analyse: "{text[:20]}..." -> {type(result).__name__}')

                if isinstance(result, dict):
                    emotion = result.get('primary_emotion', 'unknown')
                    intensity = result.get('intensity', 0)
                    print(f'      Émotion: {emotion} (intensité: {intensity})')

            except Exception as e:
                print(f'   ❌ Erreur analyse: {e}')

        success_rate = len([r for r in results if r is not None]) / len(test_inputs)
        print(f'✅ Cœur émotionnel: {success_rate * 100:.0f}% de réussite')
        return success_rate > 0.5

    except Exception as e:
        print(f'❌ Cœur émotionnel défaillant: {e}')
        return False


def test_memory_system():
    """Test du système mémoire"""
    print('\n4. TEST SYSTÈME MÉMOIRE')
    try:
        from core.agi_fusion.unified_memory import UnifiedMemory

        memory = UnifiedMemory()

        # Test stockage/récupération
        test_data = {
            'user_id': 'test_user',
            'message': 'test message',
            'timestamp': time.time(),
            'context': {'test': True},
        }

        # Test stockage
        try:
            memory.store_context('test_session', test_data)
            print('   ✅ Stockage mémoire réussi')
            store_ok = True
        except Exception as e:
            print(f'   ❌ Stockage échoué: {e}')
            store_ok = False

        # Test récupération
        try:
            retrieved = memory.get_context_summary('test_session')
            print(f'   ✅ Récupération mémoire: {retrieved is not None}')
            retrieve_ok = retrieved is not None
        except Exception as e:
            print(f'   ❌ Récupération échouée: {e}')
            retrieve_ok = False

        # Test capacités avancées
        has_search = hasattr(memory, 'search_memories')
        has_consolidate = hasattr(memory, 'consolidate_memories')

        print(f'   Recherche mémoire: {"✅" if has_search else "❌"}')
        print(f'   Consolidation: {"✅" if has_consolidate else "❌"}')

        print(f'✅ Système mémoire: {(store_ok + retrieve_ok) / 2 * 100:.0f}% fonctionnel')
        return store_ok and retrieve_ok

    except Exception as e:
        print(f'❌ Système mémoire défaillant: {e}')
        return False


def test_learning_manager():
    """Test du gestionnaire d'apprentissage"""
    print('\n5. TEST GESTIONNAIRE APPRENTISSAGE')
    try:
        from core.learning.immediate_learning_manager import ImmediateLearningManager

        learning = ImmediateLearningManager()

        # Test capacités
        can_record = hasattr(learning, 'record_interaction')
        can_apply = hasattr(learning, 'apply_learnings')
        can_analyze = hasattr(learning, 'analyze_conversation')

        print(f'   Enregistrement: {"✅" if can_record else "❌"}')
        print(f'   Application: {"✅" if can_apply else "❌"}')
        print(f'   Analyse: {"✅" if can_analyze else "❌"}')

        # Test enregistrement réel
        if can_record:
            try:
                learning.record_interaction(
                    'test_user',
                    'Quelle est la capitale de la France?',
                    'La capitale de la France est Paris.',
                    {'emotion': 'curious', 'confidence': 0.9},
                )
                print('   ✅ Enregistrement interaction réussi')
                record_ok = True
            except Exception as e:
                print(f'   ❌ Enregistrement échoué: {e}')
                record_ok = False
        else:
            record_ok = False

        # Vérifier données apprises
        try:
            stats = learning.get_stats() if hasattr(learning, 'get_stats') else {}
            learned_count = stats.get('total_learned', 0)
            print(f'   📚 Éléments appris: {learned_count}')
        except:
            pass

        capabilities = sum([can_record, can_apply, can_analyze])
        print(f'✅ Learning Manager: {capabilities}/3 capacités disponibles')
        return capabilities >= 2

    except Exception as e:
        print(f'❌ Gestionnaire apprentissage défaillant: {e}')
        return False


def test_conversation_flow():
    """Test du flux de conversation complet"""
    print('\n6. TEST FLUX CONVERSATION COMPLET')
    try:
        from jeffrey_v2_integration import JeffreyV2

        jeffrey = JeffreyV2()

        test_messages = [
            "Bonjour Jeffrey, comment ça va?",
            "Peux-tu m'expliquer ce que tu fais?",
            "Raconte-moi une blague",
        ]

        success_count = 0
        total_time = 0

        for i, message in enumerate(test_messages, 1):
            try:
                print(f'   Test {i}: "{message[:30]}..."')
                start_time = time.time()

                # Traiter la conversation
                result = jeffrey.process_conversation(message, f'test_user_{i}')

                # Si c'est une coroutine, l'exécuter
                if hasattr(result, '__await__'):
                    result = asyncio.run(result)

                response_time = time.time() - start_time
                total_time += response_time

                if result and 'response' in result:
                    response = result['response']
                    print(f'      ✅ Réponse en {response_time:.2f}s: "{response[:50]}..."')
                    success_count += 1
                else:
                    print('      ❌ Pas de réponse valide')

            except Exception as e:
                print(f'      ❌ Erreur: {str(e)[:60]}...')

        avg_time = total_time / len(test_messages) if test_messages else 0
        success_rate = success_count / len(test_messages) * 100

        print(f'✅ Flux conversation: {success_rate:.0f}% réussite, {avg_time:.2f}s/msg')
        return success_rate >= 70

    except Exception as e:
        print(f'❌ Test conversation échoué: {e}')
        return False


def test_experimental_modules():
    """Test rapide des modules expérimentaux"""
    print('\n7. TEST MODULES EXPÉRIMENTAUX')

    experimental_modules = [
        ('future_modules.cognitive_expansion.jeffrey_brain_orchestrator', 'JeffreyBrainOrchestrator'),
        ('future_modules.emotion_engine.living_soul_engine', 'LivingSoulEngine'),
        ('future_modules.dream_modules.dream_engine', 'DreamEngine'),
        ('Orchestrateur_IA.core.orchestration.multi_agent_controller', 'MultiAgentController'),
    ]

    working_modules = 0
    total_modules = len(experimental_modules)

    for module_path, class_name in experimental_modules:
        try:
            module = importlib.import_module(module_path)
            if hasattr(module, class_name):
                # Tenter instanciation simple
                cls = getattr(module, class_name)
                instance = cls()
                print(f'   ✅ {class_name}: Instanciation OK')
                working_modules += 1
            else:
                print(f'   ⚠️ {class_name}: Classe manquante')
        except ImportError:
            print(f'   ❌ {module_path}: Module non trouvé')
        except Exception as e:
            print(f'   ⚠️ {class_name}: Erreur instanciation - {str(e)[:40]}...')

    experimental_rate = working_modules / total_modules * 100
    print(f'✅ Modules expérimentaux: {experimental_rate:.0f}% fonctionnels ({working_modules}/{total_modules})')
    return experimental_rate


def main():
    """Audit principal"""
    print('🧪 AUDIT FONCTIONNALITÉ RÉELLE JEFFREY V2')
    print('=' * 60)

    # Tests des modules core
    jeffrey, status = test_jeffrey_core()
    voice_ok = test_voice_system()
    emotion_ok = test_emotional_core()
    memory_ok = test_memory_system()
    learning_ok = test_learning_manager()
    conversation_ok = test_conversation_flow()
    experimental_rate = test_experimental_modules()

    # Calcul score global
    core_tests = [bool(jeffrey), voice_ok, emotion_ok, memory_ok, learning_ok, conversation_ok]

    core_score = sum(core_tests) / len(core_tests) * 100

    print('\n' + '=' * 60)
    print('📊 RÉSUMÉ AUDIT FONCTIONNALITÉ')
    print('=' * 60)
    print(f'🎯 Score Core: {core_score:.0f}%')
    print(f'🧪 Score Expérimental: {experimental_rate:.0f}%')
    print(f'📈 Score Global: {(core_score * 0.8 + experimental_rate * 0.2):.0f}%')

    print('\n🔍 DÉTAIL CORE:')
    tests_names = [
        'Jeffrey Core',
        'Voice System',
        'Emotional Core',
        'Memory System',
        'Learning Manager',
        'Conversation Flow',
    ]
    for name, result in zip(tests_names, core_tests):
        icon = '✅' if result else '❌'
        print(f'   {icon} {name}')

    print(f'\n🎭 Modules expérimentaux fonctionnels: {experimental_rate:.0f}%')

    # Recommandations
    print('\n💡 RECOMMANDATIONS:')
    if core_score >= 80:
        print('   🎉 Core excellent - Focus sur optimisation et UX')
    elif core_score >= 60:
        print('   ⚠️ Core bon - Corriger modules défaillants')
    else:
        print('   🚨 Core critique - Refactoring majeur nécessaire')

    if experimental_rate < 30:
        print('   🧹 Purger modules expérimentaux non fonctionnels')

    return core_score, experimental_rate


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('\n👋 Audit interrompu')
    except Exception as e:
        print(f'\n❌ Erreur fatale audit: {e}')
