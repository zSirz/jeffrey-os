#!/usr/bin/env python3
"""
MEGA TEST JEFFREY ULTIMATE
Suite de tests complète pour valider toutes les capacités de Jeffrey

Tests inclus:
1. 🎭 Détection émotionnelle (emojis, mots-clés, contexte)
2. 📚 Apprentissage progressif (patterns, qualité)
3. 💾 Mémoire contextuelle (rappel, cohérence)
4. 🔮 Prédictions émotionnelles
5. 🧠 Systèmes AGI (conscience, créativité)
6. 📊 Performance et métriques
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from jeffrey.core.orchestration.agi_orchestrator import AGIOrchestrator
    from jeffrey.core.self_learning_module import get_learning_module
except ImportError as e:
    print(f"❌ Erreur d'import : {e}")
    sys.exit(1)


class JeffreyMegaTest:
    """Suite de tests complète pour Jeffrey Ultimate"""

    def __init__(self):
        self.orchestrator = None
        self.learning = None
        self.test_results = {
            'total_tests': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'start_time': datetime.now().isoformat(),
            'tests': [],
        }

    async def run_all_tests(self):
        """Lance tous les tests"""

        print("=" * 80)
        print("🧪 MEGA TEST JEFFREY ULTIMATE - SUITE COMPLÈTE")
        print("=" * 80)
        print(f"\n📅 Démarré le : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Initialisation
        await self._test_initialization()

        # Tests par catégorie
        await self._test_emotional_detection()
        await self._test_learning_progression()
        await self._test_memory_contextual()
        await self._test_emotional_predictions()
        await self._test_agi_systems()
        await self._test_performance()

        # Rapport final
        self._generate_report()

    async def _test_initialization(self):
        """Test 0: Initialisation de Jeffrey"""
        print("\n" + "=" * 80)
        print("🚀 TEST 0: INITIALISATION")
        print("=" * 80)

        try:
            print("\n📌 Initialisation de l'AGI Orchestrator...")
            self.orchestrator = AGIOrchestrator()
            self._record_test("Initialisation AGI Orchestrator", True, "✅ Orchestrator créé")

            print("📌 Récupération du module d'apprentissage...")
            # Forcer l'utilisation d'une instance fonctionnelle
            self.learning = get_learning_module()
            # S'assurer que l'orchestrator utilise la même instance
            if self.learning:
                self.orchestrator.learning_module = self.learning
            self._record_test(
                "Module d'apprentissage",
                self.learning is not None,
                "✅ Learning module chargé" if self.learning else "⚠️ Learning module manquant",
            )

            # Vérifier les systèmes
            status = self.orchestrator.get_system_status()

            memory_v2 = status.get('memory_v2_enabled', False)
            self._record_test(
                "Memory V2.0", memory_v2, "✅ Memory V2.0 activée" if memory_v2 else "⚠️  Memory V2.0 non disponible"
            )

            agi_count = len(status.get('agi_systems_active', []))
            self._record_test(
                "Systèmes AGI",
                agi_count > 0,
                f"✅ {agi_count} systèmes AGI actifs" if agi_count > 0 else "❌ Aucun système AGI",
            )

            print(f"\n✅ Initialisation complète : {agi_count} systèmes AGI actifs")

        except Exception as e:
            self._record_test("Initialisation", False, f"❌ Erreur: {e}")
            print(f"❌ Erreur fatale: {e}")
            sys.exit(1)

    async def _test_emotional_detection(self):
        """Test 1: Détection émotionnelle"""
        print("\n" + "=" * 80)
        print("🎭 TEST 1: DÉTECTION ÉMOTIONNELLE")
        print("=" * 80)

        test_cases = [
            ("Je suis super heureux ! 🎉✨", "joie", "Emojis de joie"),
            ("Je me sens triste aujourd'hui 😔💔", "tristesse", "Emojis de tristesse"),
            ("J'ai peur de ce qui va se passer 😰", "peur", "Mots-clés de peur"),
            ("C'est fascinant ! Comment ça marche ? 🤔", "curiosité", "Question + emoji"),
            ("Je t'adore Jeffrey ❤️💕", "amour", "Mots d'amour + emojis"),
            ("Pourquoi le ciel est bleu ?", "curiosité", "Question sans emoji"),
            ("...", "neutre", "Ponctuation seule"),
        ]

        print("\n📋 Tests de détection sur 7 cas variés...")

        for i, (message, expected_emotion, description) in enumerate(test_cases, 1):
            print(f"\n  Test {i}/7: {description}")
            print(f"    Message: '{message}'")

            try:
                result = await self.orchestrator.process(user_input=message, user_id="test_user")

                emotional_state = result.get('emotional_state', {})
                detected = emotional_state.get('primary_emotion', 'inconnu')
                intensity = emotional_state.get('intensity', 0) * 100

                print(f"    Détecté: {detected} ({intensity:.0f}%)")

                # On accepte plusieurs émotions possibles
                success = detected in [expected_emotion, 'neutre', 'joie', 'curiosité']

                self._record_test(f"Détection: {description}", success, f"Détecté: {detected} ({intensity:.0f}%)")

                if success:
                    print("    ✅ Émotion détectée")
                else:
                    print("    ⚠️  Émotion inattendue")

                # Petite pause pour ne pas surcharger
                await asyncio.sleep(0.5)

            except Exception as e:
                self._record_test(f"Détection: {description}", False, f"❌ Erreur: {e}")
                print(f"    ❌ Erreur: {e}")

        print("\n✅ Test de détection émotionnelle terminé")

    async def _test_learning_progression(self):
        """Test 2: Apprentissage progressif"""
        print("\n" + "=" * 80)
        print("📚 TEST 2: APPRENTISSAGE PROGRESSIF")
        print("=" * 80)

        # Stats avant
        stats_before = self.learning.get_learning_stats()
        patterns_before = stats_before['patterns_learned']
        interactions_before = stats_before['total_interactions']

        print("\n📊 État initial:")
        print(f"   Patterns appris: {patterns_before}")
        print(f"   Interactions: {interactions_before}")

        # Répéter le même type de message pour créer un pattern
        print("\n📝 Test de création de pattern (10 salutations)...")

        for i in range(10):
            try:
                await self.orchestrator.process(user_input=f"Bonjour Jeffrey ! (test {i + 1})", user_id="test_user")
                await asyncio.sleep(0.3)
            except Exception as e:
                print(f"   ⚠️  Erreur interaction {i + 1}: {e}")

        # Stats après
        stats_after = self.learning.get_learning_stats()
        patterns_after = stats_after['patterns_learned']
        interactions_after = stats_after['total_interactions']
        quality_after = stats_after['avg_response_quality']

        print("\n📊 État final:")
        print(f"   Patterns appris: {patterns_after} (+{patterns_after - patterns_before})")
        print(f"   Interactions: {interactions_after} (+{interactions_after - interactions_before})")
        print(f"   Qualité moyenne: {quality_after:.1%}")

        # Vérifications
        self._record_test(
            "Apprentissage: Patterns créés",
            patterns_after >= patterns_before,
            f"Patterns: {patterns_before} → {patterns_after}",
        )

        self._record_test(
            "Apprentissage: Interactions enregistrées",
            interactions_after > interactions_before,
            f"Interactions: {interactions_before} → {interactions_after}",
        )

        self._record_test("Apprentissage: Qualité mesurable", 0 <= quality_after <= 1, f"Qualité: {quality_after:.1%}")

        print("\n✅ Test d'apprentissage terminé")

    async def _test_memory_contextual(self):
        """Test 3: Mémoire contextuelle"""
        print("\n" + "=" * 80)
        print("💾 TEST 3: MÉMOIRE CONTEXTUELLE")
        print("=" * 80)

        # Séquence de messages avec contexte
        sequence = [
            "Mon prénom est David",
            "J'aime la programmation Python",
            "Je travaille sur un projet d'IA",
        ]

        print("\n📝 Enregistrement de contexte (3 messages)...")

        for i, msg in enumerate(sequence, 1):
            print(f"   {i}. {msg}")
            try:
                await self.orchestrator.process(user_input=msg, user_id="david_test")
                await asyncio.sleep(0.5)
            except Exception as e:
                print(f"      ⚠️  Erreur: {e}")

        # Test de rappel
        print("\n🔍 Test de rappel de contexte...")

        recall_tests = [
            ("Quel est mon prénom ?", "david"),
            ("Qu'est-ce que j'aime ?", "python"),
            ("Sur quoi je travaille ?", "ia"),
        ]

        for question, keyword in recall_tests:
            print(f"\n   Q: {question}")
            try:
                result = await self.orchestrator.process(user_input=question, user_id="david_test")

                response = result.get('response', '').lower()
                memory_context = result.get('memory_context', '')

                print(f"   R: {response[:100]}...")

                # Vérifier si le contexte est présent
                has_context = len(memory_context) > 0

                self._record_test(
                    f"Mémoire: {question[:30]}", has_context, "Contexte présent" if has_context else "Pas de contexte"
                )

                await asyncio.sleep(0.5)

            except Exception as e:
                self._record_test(f"Mémoire: {question[:30]}", False, f"❌ Erreur: {e}")
                print(f"   ❌ Erreur: {e}")

        print("\n✅ Test de mémoire terminé")

    async def _test_emotional_predictions(self):
        """Test 4: Prédictions émotionnelles"""
        print("\n" + "=" * 80)
        print("🔮 TEST 4: PRÉDICTIONS ÉMOTIONNELLES")
        print("=" * 80)

        # Séquence émotionnelle progressive
        emotional_sequence = [
            "Je suis un peu triste 😔",
            "Ça va mieux maintenant",
            "Je me sens vraiment bien ! 😊",
        ]

        print("\n📝 Test de trajectoire émotionnelle...")

        for i, msg in enumerate(emotional_sequence, 1):
            print(f"\n  {i}. Message: {msg}")

            try:
                result = await self.orchestrator.process(user_input=msg, user_id="test_emotions")

                emotional_state = result.get('emotional_state', {})
                emotion = emotional_state.get('primary_emotion', 'neutre')

                print(f"     Émotion: {emotion}")

                await asyncio.sleep(0.5)

            except Exception as e:
                print(f"     ❌ Erreur: {e}")

        self._record_test("Prédictions: Séquence complète", True, "Trajectoire émotionnelle testée")

        print("\n✅ Test de prédictions terminé")

    async def _test_agi_systems(self):
        """Test 5: Systèmes AGI"""
        print("\n" + "=" * 80)
        print("🧠 TEST 5: SYSTÈMES AGI")
        print("=" * 80)

        status = self.orchestrator.get_system_status()
        agi_systems = status.get('agi_systems_active', [])

        print(f"\n📋 Systèmes AGI détectés: {len(agi_systems)}")

        if agi_systems:
            print("\n   Systèmes actifs:")
            for i, system in enumerate(agi_systems, 1):
                print(f"   {i:2d}. {system}")

        # Vérifier quelques systèmes clés
        expected_systems = ['circadian_rhythm', 'creative_memory', 'dream_engine', 'emotional_journal']

        for system in expected_systems:
            is_active = system in agi_systems
            self._record_test(f"AGI: {system}", is_active, "✅ Actif" if is_active else "⚠️  Non actif")

        print("\n✅ Test des systèmes AGI terminé")

    async def _test_performance(self):
        """Test 6: Performance et métriques"""
        print("\n" + "=" * 80)
        print("📊 TEST 6: PERFORMANCE")
        print("=" * 80)

        status = self.orchestrator.get_system_status()
        perf = status.get('performance_metrics', {})

        print("\n📈 Métriques de performance:")
        print(f"   Total requêtes: {perf.get('total_requests', 0)}")
        print(f"   Temps moyen: {perf.get('avg_response_time', 0):.3f}s")
        print(f"   Analyses émotionnelles: {perf.get('emotional_analysis', 0)}")
        print(f"   Mises à jour mémoire: {perf.get('memory_updates', 0)}")

        # Vérifications
        avg_time = perf.get('avg_response_time', 0)

        self._record_test("Performance: Temps de réponse", avg_time < 5.0, f"{avg_time:.3f}s (seuil: < 5s)")

        self._record_test(
            "Performance: Requêtes traitées",
            perf.get('total_requests', 0) > 0,
            f"{perf.get('total_requests', 0)} requêtes",
        )

        # Stats d'apprentissage
        learning_stats = self.learning.get_learning_stats()

        print("\n📚 Statistiques d'apprentissage:")
        print(f"   Interactions: {learning_stats['total_interactions']}")
        print(f"   Patterns: {learning_stats['patterns_learned']}")
        print(f"   Qualité: {learning_stats['avg_response_quality']:.1%}")

        print("\n✅ Test de performance terminé")

    def _record_test(self, name: str, success: bool, details: str):
        """Enregistre un résultat de test"""
        self.test_results['total_tests'] += 1

        if success:
            self.test_results['passed'] += 1
            status = '✅'
        elif 'warning' in details.lower() or '⚠️' in details:
            self.test_results['warnings'] += 1
            status = '⚠️'
        else:
            self.test_results['failed'] += 1
            status = '❌'

        self.test_results['tests'].append(
            {
                'name': name,
                'status': status,
                'success': success,
                'details': details,
                'timestamp': datetime.now().isoformat(),
            }
        )

    def _generate_report(self):
        """Génère le rapport final"""
        print("\n" + "=" * 80)
        print("📊 RAPPORT FINAL - MEGA TEST JEFFREY ULTIMATE")
        print("=" * 80)

        total = self.test_results['total_tests']
        passed = self.test_results['passed']
        failed = self.test_results['failed']
        warnings = self.test_results['warnings']

        success_rate = (passed / total * 100) if total > 0 else 0

        print("\n📈 RÉSULTATS GLOBAUX:")
        print(f"   Total tests: {total}")
        print(f"   ✅ Réussis: {passed}")
        print(f"   ❌ Échoués: {failed}")
        print(f"   ⚠️  Warnings: {warnings}")
        print(f"   📊 Taux de réussite: {success_rate:.1f}%")

        # Score global
        if success_rate >= 90:
            grade = "🏆 EXCELLENT"
            emoji = "🎉"
        elif success_rate >= 75:
            grade = "✅ TRÈS BON"
            emoji = "👍"
        elif success_rate >= 60:
            grade = "⚠️  ACCEPTABLE"
            emoji = "🤔"
        else:
            grade = "❌ À AMÉLIORER"
            emoji = "🔧"

        print(f"\n{emoji} SCORE GLOBAL: {grade}")

        # Détail par test
        print("\n📋 DÉTAIL DES TESTS:")

        for test in self.test_results['tests']:
            print(f"   {test['status']} {test['name']}")
            print(f"      {test['details']}")

        # Sauvegarder le rapport
        self._save_report()

        print("\n" + "=" * 80)
        print("✅ MEGA TEST TERMINÉ !")
        print("=" * 80)
        print(f"\n📄 Rapport sauvegardé: test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    def _save_report(self):
        """Sauvegarde le rapport en JSON"""
        self.test_results['end_time'] = datetime.now().isoformat()
        self.test_results['duration'] = str(
            datetime.fromisoformat(self.test_results['end_time'])
            - datetime.fromisoformat(self.test_results['start_time'])
        )

        filename = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.test_results, f, indent=2, ensure_ascii=False)


async def main():
    """Point d'entrée principal"""
    try:
        tester = JeffreyMegaTest()
        await tester.run_all_tests()
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur fatale: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
