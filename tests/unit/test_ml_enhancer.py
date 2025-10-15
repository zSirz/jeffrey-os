#!/usr/bin/env python3
"""
Test de l'EmotionMLEnhancer avec contexte temporel et apprentissage
"""

from Orchestrateur_IA.core.emotions.emotion_ml_enhancer import EmotionMLEnhancer
import sys
import os

# Ajouter le chemin pour importer directement le module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Orchestrateur_IA", "core", "emotions"))


def test_ml_enhancer():
    """Test l'enhancer ML avec contexte temporel"""

    print("🧠 Test de l'EmotionMLEnhancer\n")

    # Créer l'enhancer
    enhancer = EmotionMLEnhancer(history_size=20)

    # Simulation d'une conversation avec évolution émotionnelle
    conversation = [
        ("Bonjour Jeffrey !", "neutre/joie légère"),
        ("Je suis super content aujourd'hui 😊", "joie claire"),
        ("J'ai eu une excellente nouvelle au travail !", "joie renforcée"),
        ("Mais je m'inquiète un peu pour demain...", "transition joie->inquiétude"),
        ("J'ai peur de ne pas être à la hauteur 😟", "peur/inquiétude"),
        ("Tu crois que je vais y arriver ?", "recherche de soutien"),
        ("Merci de m'écouter, ça me fait du bien 🤗", "empathie/gratitude"),
    ]

    print("📝 Simulation d'une conversation avec évolution émotionnelle:\n")

    for i, (text, expected) in enumerate(conversation):
        print(f"Message {i + 1}: {text}")
        print(f"Attendu: {expected}")

        # Détection améliorée
        result = enhancer.detect_emotion_enhanced(text, user_id="test_user")

        print(f"➤ Émotion principale: {result['emotion']}")
        print(
            f"➤ Scores: {dict(sorted(result['scores'].items(), key=lambda x: x[1], reverse=True)[:3])}"
        )
        print(f"➤ Confiance: {result['confidence']:.2%}")

        if result["context"]["predicted_next"]:
            print(f"➤ Prochaine émotion probable: {result['context']['predicted_next']}")

        print()

        # Petite pause pour simuler le temps entre messages
        # TODO: Remplacer par asyncio.sleep ou threading.Event

    # Afficher les insights globaux
    print("\n🔍 Insights émotionnels globaux:")
    insights = enhancer.get_emotional_insights()

    print(f"➤ Trajectoire: {insights['trajectory']}")
    print(f"➤ Émotion dominante: {insights['dominant_emotion']}")
    print(f"➤ Volatilité: {insights['volatility']:.2%}")
    print(f"➤ Historique récent: {insights['emotion_history'][-5:]}")
    print(f"➤ Recommandations:")
            for rec in insights["recommendations"]:
        print(f"   • {rec}")

    # Test de l'influence temporelle
    print("\n⏱️ Test de l'influence temporelle:")
    print("Message identique à différents moments:")

                for i in range(3):
        result = enhancer.detect_emotion_enhanced("Je me sens bien", user_id="test_user")
        print(f"   Essai {i + 1}: Score joie = {result['scores'].get('joie', 0):.2f}")
        # TODO: Remplacer par asyncio.sleep ou threading.Event

    # Test des transitions apprises
    print("\n🔄 Transitions émotionnelles apprises:")
                    if enhancer.emotion_transitions:
                        for from_emotion, transitions in list(enhancer.emotion_transitions.items())[:3]:
            print(
                f"   {from_emotion} → {dict(sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:2])}"
            )


                            if __name__ == "__main__":
    test_ml_enhancer()
