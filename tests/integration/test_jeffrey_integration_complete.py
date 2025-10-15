#!/usr/bin/env python3
"""
Test d'intégration complète du système Jeffrey
Vérifie que tous les composants fonctionnent ensemble
"""

import time
import json
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_jeffrey_complete():
    """Test que tout fonctionne ensemble"""

    print("🔍 Test d'intégration complète Jeffrey...")

    results = {"tests_passed": 0, "tests_failed": 0, "errors": []}

    # Test 1: Import des modules critiques
    print("\n1️⃣ Test des imports...")
    try:
        from Orchestrateur_IA.core.jeffrey_emotional_core import JeffreyEmotionalCore
        from core.consciousness.jeffrey_living_consciousness import JeffreyLivingConsciousness
        from core.emotions.jeffrey_curiosity_engine import JeffreyCuriosityEngine

        print("✅ Tous les modules critiques importés avec succès")
        results["tests_passed"] += 1
        except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        results["tests_failed"] += 1
        results["errors"].append(f"Import error: {str(e)}")

    # Test 2: Initialisation du core émotionnel
    print("\n2️⃣ Test du core émotionnel...")
            try:
        emotional_core = JeffreyEmotionalCore()
        assert hasattr(
            emotional_core, "curiosity_engine"
        ), "Curiosity engine non trouvé dans emotional core"
        assert emotional_core.emotional_state["curiosité"] > 0, "Curiosité non initialisée"
        print("✅ Core émotionnel initialisé avec curiosité")
        results["tests_passed"] += 1
                except Exception as e:
        print(f"❌ Erreur core émotionnel: {e}")
        results["tests_failed"] += 1
        results["errors"].append(f"Emotional core error: {str(e)}")

    # Test 3: Initialisation de la conscience vivante
    print("\n3️⃣ Test de la conscience vivante...")
                    try:
        consciousness = JeffreyLivingConsciousness()
        assert hasattr(
            consciousness, "curiosity_engine"
        ), "Curiosity engine non trouvé dans consciousness"
        assert consciousness.proactive_mode, "Mode proactif non activé"
        print("✅ Conscience vivante initialisée avec curiosité proactive")
        results["tests_passed"] += 1
                        except Exception as e:
        print(f"❌ Erreur conscience: {e}")
        results["tests_failed"] += 1
        results["errors"].append(f"Consciousness error: {str(e)}")

    # Test 4: Test de la curiosité
    print("\n4️⃣ Test du moteur de curiosité...")
                            try:
        curiosity = JeffreyCuriosityEngine()

        # Test de détection d'intérêt
        response = curiosity.generate_curious_response(
            "J'adore jouer au tennis le weekend", {"humeur": "curieuse"}
        )
        assert "tennis" in response.lower() or "?" in response, "Pas de question curieuse générée"
        print(f"✅ Réponse curieuse générée: {response[:100]}...")

        # Test de conversation proactive
        proactive = curiosity.proactive_conversation_starter(35, "curieuse")
        assert proactive is not None, "Pas de message proactif généré"
        print(f"✅ Message proactif: {proactive[:100]}...")

        results["tests_passed"] += 2
                                except Exception as e:
        print(f"❌ Erreur curiosité: {e}")
        results["tests_failed"] += 1
        results["errors"].append(f"Curiosity error: {str(e)}")

    # Test 5: Intégration consciousness + curiosity
    print("\n5️⃣ Test d'intégration conscience + curiosité...")
                                    try:
        # Simuler une conversation
        user_input = "J'ai couru 10km ce matin, je suis épuisé mais content!"
        response = consciousness.respond_naturally(user_input)

        # Vérifier que la réponse contient de la curiosité
        assert len(response) > 20, "Réponse trop courte"
        print(f"✅ Réponse intégrée: {response[:150]}...")

        # Vérifier que la curiosité a enregistré l'info
        curiosity_level = consciousness.curiosity_engine.get_curiosity_level()
        assert curiosity_level > 0, "Niveau de curiosité non mis à jour"
        print(f"✅ Niveau de curiosité: {curiosity_level:.2f}")

        results["tests_passed"] += 2
                                        except Exception as e:
        print(f"❌ Erreur intégration: {e}")
        results["tests_failed"] += 1
        results["errors"].append(f"Integration error: {str(e)}")

    # Test 6: Mémoire et persistance
    print("\n6️⃣ Test de la mémoire...")
                                            try:
        # Vérifier que le système peut sauvegarder l'état
        state = consciousness.get_consciousness_state()
        assert "emotional_layers" in state, "État émotionnel manquant"
        assert "humeur_actuelle" in state, "Humeur actuelle manquante"

        # Sauvegarder et recharger
        consciousness._save_consciousness_state()
        print("✅ État de conscience sauvegardé")

        # Créer une nouvelle instance et vérifier la persistance
        new_consciousness = JeffreyLivingConsciousness()
        new_consciousness._load_consciousness_state()
        print("✅ État de conscience rechargé avec succès")

        results["tests_passed"] += 2
                                                except Exception as e:
        print(f"❌ Erreur mémoire: {e}")
        results["tests_failed"] += 1
        results["errors"].append(f"Memory error: {str(e)}")

    # Test 7: Performance
    print("\n7️⃣ Test de performance...")
                                                    try:
        start_time = time.time()

        # Simuler 10 interactions
                                                        for i in range(10):
            response = consciousness.respond_naturally(f"Test message {i}")

        elapsed = time.time() - start_time
        avg_time = elapsed / 10

        assert avg_time < 1.0, f"Temps de réponse trop lent: {avg_time:.2f}s"
        print(f"✅ Performance OK: {avg_time:.3f}s par réponse en moyenne")

        results["tests_passed"] += 1
                                                            except Exception as e:
        print(f"❌ Erreur performance: {e}")
        results["tests_failed"] += 1
        results["errors"].append(f"Performance error: {str(e)}")

    # Résumé final
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DES TESTS D'INTÉGRATION")
    print("=" * 60)
    print(f"✅ Tests réussis: {results['tests_passed']}")
    print(f"❌ Tests échoués: {results['tests_failed']}")

                                                                if results["errors"]:
        print("\n⚠️ Erreurs détectées:")
                                                                    for error in results["errors"]:
            print(f"  - {error}")

    # Déterminer le statut global
    total_tests = results["tests_passed"] + results["tests_failed"]
    success_rate = results["tests_passed"] / total_tests if total_tests > 0 else 0

                                                                        if success_rate == 1.0:
        print("\n🎉 TOUS LES TESTS PASSENT ! Le système est pleinement intégré.")
                                                                            elif success_rate > 0.8:
        print("\n✅ La plupart des tests passent. Quelques ajustements nécessaires.")
                                                                                elif success_rate > 0.5:
        print("\n⚠️ Intégration partielle. Des corrections sont nécessaires.")
                                                                                    else:
        print("\n❌ Intégration défaillante. Révision majeure requise.")

    print("=" * 60)

    # Sauvegarder le rapport
                                                                                        with open("integration_test_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "results": results,
                "success_rate": success_rate,
            },
            f,
            indent=2,
        )

                                                                                            return success_rate > 0.8


                                                                                            if __name__ == "__main__":
    # Exécuter les tests
    success = test_jeffrey_complete()

    # Code de sortie basé sur le succès
    sys.exit(0 if success else 1)
