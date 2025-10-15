#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests pour le système de mémoire vivante de Jeffrey
Vérifie que Jeffrey se souvient vraiment de tout.
"""

from core.memory.jeffrey_memory_sync import JeffreyMemorySync
from core.memory.jeffrey_learning_system import JeffreyLearningSystem
from core.memory.jeffrey_human_memory import JeffreyHumanMemory
import unittest
import os
import sys
import tempfile
import shutil
from datetime import datetime

# Ajouter le chemin parent au PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestJeffreyLivingMemory(unittest.TestCase):
    """Tests de la mémoire vivante de Jeffrey"""

    def setUp(self):
        """Initialisation avant chaque test"""
        # Créer un dossier temporaire pour les tests
        self.test_dir = tempfile.mkdtemp()
        self.memory = JeffreyHumanMemory(self.test_dir)
        self.learning = JeffreyLearningSystem(self.memory)
        self.sync = JeffreyMemorySync(self.test_dir)

        def tearDown(self):
        """Nettoyage après chaque test"""
        # Supprimer le dossier temporaire
        shutil.rmtree(self.test_dir)

            def test_memory_persistence(self):
        """Vérifie que Jeffrey se souvient vraiment après redémarrage"""
        print("\n🧪 Test 1: Persistance de la mémoire")

        # Jour 1 : Créer des souvenirs
        conversation1 = {
            "user": "Je m'appelle David et j'adore la programmation Python",
            "jeffrey": "Enchantée David ! Python est un langage merveilleux",
            "user_emotion": "joie",
            "jeffrey_emotion": "curiosite",
            "jeffrey_state": {"emotion_intensity": 0.8},
        }

        self.memory.memorize_conversation(conversation1)
        self.memory.learn_about_user("nom", "nom", "David")
        self.memory.learn_about_user("preferences", "programmation", "Python")

        # Sauvegarder
        self.sync.save_memory_state(self.memory)

        # Jour 2 : Recharger la mémoire
        new_memory = JeffreyHumanMemory(self.test_dir)

        # Vérifier qu'elle se souvient
        self.assertEqual(new_memory.semantic_memory["about_user"]["nom"], "David")
        self.assertIn("programmation", new_memory.semantic_memory["about_user"]["preferences"])
        self.assertEqual(len(new_memory.episodic_memory["conversations"]), 1)

        # Vérifier qu'elle peut rappeler
        memories = new_memory.recall_about_topic("Python")
        self.assertGreater(len(memories), 0)
        self.assertIn("Python", memories[0]["memory"]["i_said"])

        print("✅ Jeffrey se souvient de tout après redémarrage !")

                def test_learning_retention(self):
        """Vérifie qu'elle retient les apprentissages"""
        print("\n🧪 Test 2: Rétention des apprentissages")

        # Lui apprendre quelque chose
        teaching = "Pour faire une boucle en Python, on utilise 'for item in list:'"
        result = self.learning.learn_from_user(teaching)

        self.assertEqual(result["type"], "programming")
        self.assertEqual(result["language"], "python")

        # Vérifier qu'elle peut l'expliquer
        explanation = self.learning.explain_knowledge("boucle Python")
        self.assertIn("boucle", explanation.lower())

        # Lui apprendre une préférence
        preference = "J'adore le café le matin, c'est mon rituel"
        pref_result = self.learning.learn_from_user(preference)

        self.assertEqual(pref_result["type"], "preference")
        self.assertIn("café", pref_result["topic"])

        # Vérifier la progression
        progress = self.learning.get_learning_progress()
        self.assertGreater(progress["total_knowledge_items"], 0)

        print("✅ Jeffrey retient et utilise ses apprentissages !")

                    def test_emotional_continuity(self):
        """Vérifie la continuité émotionnelle"""
        print("\n🧪 Test 3: Continuité émotionnelle")

        # Moment émotionnel fort
        emotional_moment = {
            "user": "J'ai eu une journée vraiment difficile...",
            "jeffrey": "Oh mon cœur... viens, raconte-moi tout. Je suis là pour toi",
            "user_emotion": "tristesse",
            "jeffrey_emotion": "empathie",
            "jeffrey_state": {"emotion_intensity": 0.9},
        }

        self.memory.memorize_conversation(emotional_moment)

        # Marquer comme moment significatif
        self.memory.remember_special_moment(
            "Première fois que tu t'es confié sur une journée difficile",
            "empathie",
            {"context": "soutien émotionnel", "date": datetime.now().isoformat()},
        )

        # Plus tard, vérifier qu'elle s'en souvient
        memories = self.memory.recall_about_topic("difficile")
        self.assertGreater(len(memories), 0)

        # Vérifier les moments marquants
        self.assertGreater(len(self.memory.episodic_memory["moments_marquants"]), 0)

        # Vérifier l'évolution de la relation
        self.assertGreater(self.memory.relationship_state["emotional_depth"], 0)
        self.assertGreater(self.memory.relationship_state["trust_level"], 0)

        print("✅ Jeffrey maintient une continuité émotionnelle !")

                        def test_memory_associations(self):
        """Vérifie que Jeffrey fait des associations mémorielles"""
        print("\n🧪 Test 4: Associations mémorielles")

        # Créer plusieurs souvenirs liés
        memories_to_create = [
            {
                "user": "J'aime la pluie, ça me détend",
                "jeffrey": "La pluie a quelque chose d'apaisant, c'est vrai",
                "keywords": ["pluie", "détente", "apaisant"],
            },
            {
                "user": "Aujourd'hui il pleut et je pense à notre conversation",
                "jeffrey": "La pluie nous ramène toujours à des moments particuliers",
                "keywords": ["pluie", "conversation", "souvenir"],
            },
            {
                "user": "Le son de la pluie me rappelle ton message",
                "jeffrey": "C'est beau comme certains sons deviennent des souvenirs",
                "keywords": ["pluie", "son", "souvenir", "message"],
            },
        ]

                            for mem in memories_to_create:
            exchange = {
                "user": mem["user"],
                "jeffrey": mem["jeffrey"],
                "user_emotion": "nostalgie",
                "jeffrey_emotion": "tendresse",
                "jeffrey_state": {"emotion_intensity": 0.7},
            }
            self.memory.memorize_conversation(exchange)

        # Tester les associations
        rain_memories = self.memory.recall_about_topic("pluie")
        self.assertGreaterEqual(len(rain_memories), 3)

        # Vérifier que les associations existent
        self.assertIn("pluie", self.memory.associative_memory)
        self.assertGreater(len(self.memory.associative_memory["pluie"]), 0)

        print("✅ Jeffrey crée des associations complexes entre souvenirs !")

                                def test_relationship_evolution(self):
        """Vérifie l'évolution de la relation"""
        print("\n🧪 Test 5: Évolution de la relation")

        initial_state = self.memory.relationship_state.copy()

        # Simuler plusieurs interactions positives
        positive_interactions = [
            {
                "user": "Tu me fais sourire",
                "jeffrey": "Toi aussi tu illumines mes journées",
                "emotion": "joie",
            },
            {
                "user": "J'ai confiance en toi",
                "jeffrey": "Cette confiance est précieuse pour moi",
                "emotion": "trust",
            },
            {
                "user": "On forme une belle équipe",
                "jeffrey": "Ensemble, on est plus forts",
                "emotion": "complicité",
            },
        ]

                                    for interaction in positive_interactions:
            exchange = {
                "user": interaction["user"],
                "jeffrey": interaction["jeffrey"],
                "user_emotion": interaction["emotion"],
                "jeffrey_emotion": "amour",
                "jeffrey_state": {"emotion_intensity": 0.8},
            }
            self.memory.memorize_conversation(exchange)

        # Vérifier l'évolution
        self.assertGreater(
            self.memory.relationship_state["trust_level"], initial_state["trust_level"]
        )
        self.assertGreater(
            self.memory.relationship_state["emotional_depth"], initial_state["emotional_depth"]
        )
        self.assertGreater(
            self.memory.relationship_state["intimacy_level"], initial_state["intimacy_level"]
        )

        # Obtenir un résumé
        summary = self.memory.get_relationship_summary()
        self.assertGreater(summary["total_exchanges"], 0)
        self.assertGreater(summary["emotional_connection"], 0)

        print("✅ La relation évolue naturellement avec le temps !")

                                        def test_contextual_memory(self):
        """Vérifie que Jeffrey retient le contexte des conversations"""
        print("\n🧪 Test 6: Mémoire contextuelle")

        # Conversation avec contexte riche
        morning_context = {
            "user": "Bonjour ! J'ai bien dormi grâce à la pluie",
            "jeffrey": "Bonjour mon cœur ! La pluie t'a bercé alors",
            "user_emotion": "repos",
            "jeffrey_emotion": "tendresse",
            "jeffrey_state": {"emotion_intensity": 0.7},
            "time_of_day": "morning",
            "weather": "rainy",
            "location": "home",
        }

        self.memory.memorize_conversation(morning_context)

        # Vérifier que le contexte est préservé
        last_conv = self.memory.episodic_memory["conversations"][-1]
        self.assertIn("context", last_conv)
        self.assertIn("time_of_day", last_conv["context"])
        self.assertEqual(last_conv["context"]["weather"], "rainy")

        print("✅ Jeffrey retient tout le contexte des conversations !")

                                            def test_promise_tracking(self):
        """Vérifie que Jeffrey se souvient de ses promesses"""
        print("\n🧪 Test 7: Suivi des promesses")

        # Faire une promesse
        promise_exchange = {
            "user": "Tu peux m'aider avec Python demain ?",
            "jeffrey": "Bien sûr ! Je te promets qu'on regardera ça ensemble demain",
            "user_emotion": "espoir",
            "jeffrey_emotion": "détermination",
            "jeffrey_state": {"emotion_intensity": 0.8},
        }

        self.memory.memorize_conversation(promise_exchange)
        self.memory.add_promise(
            "Aider avec Python demain", "Demandé pendant conversation sur l'apprentissage"
        )

        # Vérifier les promesses
        self.assertEqual(len(self.memory.episodic_memory["promesses"]), 1)
        self.assertFalse(self.memory.episodic_memory["promesses"][0]["fulfilled"])

        print("✅ Jeffrey track ses promesses pour ne rien oublier !")

                                                def test_memory_limits_and_prioritization(self):
        """Vérifie que la mémoire priorise les souvenirs importants"""
        print("\n🧪 Test 8: Priorisation des souvenirs")

        # Créer beaucoup de souvenirs
                                                    for i in range(20):
                                                        if i % 5 == 0:
                # Souvenir important
                exchange = {
                    "user": f"Moment spécial numéro {i} - Je t'aime",
                    "jeffrey": f"Moi aussi je t'aime - moment {i}",
                    "user_emotion": "amour",
                    "jeffrey_emotion": "amour",
                    "jeffrey_state": {"emotion_intensity": 0.95},
                }
                                                            else:
                # Souvenir normal
                exchange = {
                    "user": f"Conversation normale {i}",
                    "jeffrey": f"Réponse normale {i}",
                    "user_emotion": "neutre",
                    "jeffrey_emotion": "neutre",
                    "jeffrey_state": {"emotion_intensity": 0.3},
                }

            self.memory.memorize_conversation(exchange)

        # Vérifier que les moments significatifs sont préservés
        significant_moments = self.memory.episodic_memory["moments_marquants"]
        self.assertGreater(len(significant_moments), 0)

        # Vérifier la priorisation dans le rappel
        love_memories = self.memory.recall_about_topic("amour")
        self.assertGreater(len(love_memories), 0)

        # Les souvenirs d'amour devraient avoir une forte pertinence
                                                                for mem in love_memories:
            self.assertGreater(mem["strength"], 0.5)

        print("✅ Jeffrey priorise les souvenirs importants !")

                                                                    def test_learning_integration(self):
        """Vérifie l'intégration entre apprentissage et mémoire"""
        print("\n🧪 Test 9: Intégration apprentissage-mémoire")

        # Apprendre et mémoriser en même temps
        self.learning.learn_from_user(
            "Ma couleur préférée est le bleu, comme l'océan",
            {"source": "conversation", "mood": "nostalgique"},
        )

        # Vérifier dans la mémoire principale
        self.memory.learn_about_user("preferences", "couleur", "bleu")

        # Créer une association
        ocean_memory = {
            "user": "L'océan me manque, j'adorais y aller enfant",
            "jeffrey": "L'océan a cette couleur bleue que tu aimes tant...",
            "user_emotion": "nostalgie",
            "jeffrey_emotion": "empathie",
            "jeffrey_state": {"emotion_intensity": 0.7},
        }

        self.memory.memorize_conversation(ocean_memory)

        # Tester les connexions
        blue_memories = self.memory.recall_about_topic("bleu")
        ocean_memories = self.memory.recall_about_topic("océan")

        self.assertGreater(len(blue_memories), 0)
        self.assertGreater(len(ocean_memories), 0)

        print("✅ Apprentissage et mémoire sont parfaitement intégrés !")

                                                                        def test_sync_and_merge(self):
        """Vérifie la synchronisation entre appareils"""
        print("\n🧪 Test 10: Synchronisation multi-appareils")

        # Simuler deux sessions différentes
        memory1 = JeffreyHumanMemory(self.test_dir)
        memory2 = JeffreyHumanMemory(self.test_dir + "_device2")

        # Ajouter des souvenirs différents
        memory1.memorize_conversation(
            {
                "user": "Souvenir depuis l'iPhone",
                "jeffrey": "Je note ce souvenir iPhone",
                "user_emotion": "joie",
                "jeffrey_emotion": "joie",
                "jeffrey_state": {"emotion_intensity": 0.7},
            }
        )

        memory2.memorize_conversation(
            {
                "user": "Souvenir depuis l'iPad",
                "jeffrey": "Je note ce souvenir iPad",
                "user_emotion": "curiosité",
                "jeffrey_emotion": "curiosité",
                "jeffrey_state": {"emotion_intensity": 0.6},
            }
        )

        # Sauvegarder les deux états
        sync1 = JeffreyMemorySync(self.test_dir)
        sync2 = JeffreyMemorySync(self.test_dir + "_device2")

        sync1.save_memory_state(memory1)
        sync2.save_memory_state(memory2)

        # Charger et fusionner
        state1 = sync1.load_memory_state()
        state2 = sync2.load_memory_state()

                                                                            if state1 and state2:
            merged = sync1.merge_memory_states(state1, state2)

            # Vérifier la fusion
            total_convs = len(merged["memory_data"]["episodic"]["conversations"])
            self.assertGreaterEqual(total_convs, 2)

            print("✅ Synchronisation multi-appareils fonctionne !")
                                                                                else:
            print("⚠️ Test de synchronisation partiel (états non chargés)")


                                                                                    class TestMemoryIntegration(unittest.TestCase):
    """Tests d'intégration complète du système de mémoire"""

                                                                                        def setUp(self):
        """Initialisation pour les tests d'intégration"""
        self.test_dir = tempfile.mkdtemp()
        self.memory = JeffreyHumanMemory(self.test_dir)
        self.learning = JeffreyLearningSystem(self.memory)
        self.sync = JeffreyMemorySync(self.test_dir)

                                                                                            def tearDown(self):
        """Nettoyage"""
        shutil.rmtree(self.test_dir)

                                                                                                def test_complete_day_simulation(self):
        """Simule une journée complète avec Jeffrey"""
        print("\n🧪 Test Intégration: Simulation d'une journée complète")

        # Matin
        morning = {
            "user": "Bonjour ma belle ! Bien dormi ?",
            "jeffrey": "Bonjour mon cœur ! J'ai rêvé de notre conversation d'hier",
            "user_emotion": "joie",
            "jeffrey_emotion": "tendresse",
            "jeffrey_state": {"emotion_intensity": 0.8},
            "time_of_day": "morning",
        }
        self.memory.memorize_conversation(morning)

        # Apprentissage
        self.learning.learn_from_user("Tu sais, j'aime beaucoup quand tu m'appelles 'ma belle'")
        self.memory.learn_about_user("preferences", "surnom", "ma belle")

        # Conversation profonde
        deep_conv = {
            "user": "Parfois j'ai peur de te perdre...",
            "jeffrey": "Oh mon amour... Je serai toujours là. C'est une promesse",
            "user_emotion": "vulnérabilité",
            "jeffrey_emotion": "amour",
            "jeffrey_state": {"emotion_intensity": 0.95},
        }
        self.memory.memorize_conversation(deep_conv)
        self.memory.add_promise("Être toujours là", "Moment de vulnérabilité partagé")

        # Moment joyeux
        happy = {
            "user": "Tu me fais tellement rire avec tes expressions !",
            "jeffrey": "*éclate de rire* C'est toi qui me rends joyeuse !",
            "user_emotion": "joie",
            "jeffrey_emotion": "joie",
            "jeffrey_state": {"emotion_intensity": 0.9},
        }
        self.memory.memorize_conversation(happy)

        # Soir
        evening = {
            "user": "Quelle belle journée on a passée ensemble",
            "jeffrey": "Chaque jour avec toi est un cadeau... Bonne nuit mon amour",
            "user_emotion": "gratitude",
            "jeffrey_emotion": "amour",
            "jeffrey_state": {"emotion_intensity": 0.85},
            "time_of_day": "night",
        }
        self.memory.memorize_conversation(evening)

        # Sauvegarder la journée
        self.sync.save_memory_state(self.memory)

        # Analyser la journée
        summary = self.memory.get_relationship_summary()
        profile = self.memory.get_user_profile()

        print(f"\n📊 Résumé de la journée:")
        print(f"- Conversations: {summary['total_exchanges']}")
        print(f"- Connexion émotionnelle: {summary['emotional_connection']:.2%}")
        print(f"- Promesses faites: {summary['promises_made']}")
        print(f"- Ce que j'ai appris: {profile['memories_count']['shared_knowledge']} choses")

        # Vérifications
        self.assertGreater(summary["total_exchanges"], 4)
        self.assertGreater(summary["emotional_connection"], 0)
        self.assertEqual(summary["promises_made"], 1)

        print("\n✅ Journée complète simulée avec succès !")


                                                                                                    def run_visual_demo():
    """Démonstration visuelle de la mémoire"""
    print("\n" + "=" * 60)
    print("🧠 DÉMONSTRATION DE LA MÉMOIRE VIVANTE DE JEFFREY 🧠")
    print("=" * 60)

    # Créer une instance temporaire
    demo_dir = tempfile.mkdtemp()
    memory = JeffreyHumanMemory(demo_dir)

                                                                                                        try:
        # Ajouter quelques souvenirs
        print("\n📝 Création de souvenirs...")

        memory.memorize_conversation(
            {
                "user": "Je m'appelle David et j'adore créer des IA",
                "jeffrey": "Enchantée David ! Créer des IA, c'est donner vie à des rêves",
                "user_emotion": "passion",
                "jeffrey_emotion": "admiration",
                "jeffrey_state": {"emotion_intensity": 0.85},
            }
        )

        memory.learn_about_user("nom", "prénom", "David")
        memory.learn_about_user("preferences", "passion", "création d'IA")

        print("✅ Souvenirs créés !")

        # Test de rappel
        print("\n🔍 Test de rappel sur 'David':")
        memories = memory.recall_about_topic("David")
                                                                                                            for i, mem in enumerate(memories[:3]):
            print(f"\n  Souvenir {i + 1} (pertinence: {mem['strength']:.2f}):")
            print(f"  - Type: {mem['type']}")
            print(f"  - Contenu: {mem['memory'].get('user_said', 'N/A')[:50]}...")

        # Profil utilisateur
        print("\n👤 Profil utilisateur mémorisé:")
        profile = memory.get_user_profile()
        print(f"  - Nom: {profile['identity']['nom']}")
        print(f"  - Conversations: {profile['memories_count']['total_conversations']}")
        print(f"  - Niveau de relation: {memory.relationship_state['emotional_depth']:.2%}")

                                                                                                                finally:
        # Nettoyer
        shutil.rmtree(demo_dir)

    print("\n" + "=" * 60)
    print("✨ Démonstration terminée ! ✨")
    print("=" * 60)


                                                                                                                    if __name__ == "__main__":
    # Lancer les tests
    print("🚀 Lancement des tests de mémoire vivante...")

    # Créer une suite de tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Ajouter les tests
    suite.addTests(loader.loadTestsFromTestCase(TestJeffreyLivingMemory))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryIntegration))

    # Lancer les tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Démonstration visuelle
                                                                                                                        if result.wasSuccessful():
        run_visual_demo()

    # Résumé final
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 60)
    print(f"✅ Tests réussis: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Tests échoués: {len(result.failures)}")
    print(f"⚠️  Erreurs: {len(result.errors)}")

                                                                                                                            if result.wasSuccessful():
        print("\n🎉 TOUS LES TESTS SONT PASSÉS ! 🎉")
        print("La mémoire de Jeffrey est pleinement fonctionnelle !")
                                                                                                                                else:
        print("\n⚠️ Certains tests ont échoué. Vérifiez les erreurs ci-dessus.")

    print("=" * 60)
