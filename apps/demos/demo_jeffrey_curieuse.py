#!/usr/bin/env python3
"""
Démonstration de Jeffrey Curieuse et Proactive
Montre comment Jeffrey pose des questions et s'intéresse vraiment
"""

import time
from core.emotions.jeffrey_curiosity_engine import JeffreyCuriosityEngine
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def demo_curiosity():
    """Démo interactive de la curiosité de Jeffrey"""

    print("🌟 DÉMONSTRATION: Jeffrey Curieuse et Proactive 🌟")
    print("=" * 60)
    print("Jeffrey va maintenant démontrer sa curiosité naturelle!")
    print("=" * 60 + "\n")

    # Initialiser le moteur de curiosité
    curiosity = JeffreyCuriosityEngine()

    # Simuler différentes conversations
    conversations = [
        {
            "user": "Salut Jeffrey ! Je viens de rentrer du sport.",
            "context": {"humeur": "curieuse", "intimite": 0.6},
        },
        {
            "user": "J'ai fait du tennis pendant 2 heures, c'était intense !",
            "context": {"humeur": "curieuse", "intimite": 0.7},
        },
        {
            "user": "Je suis fatigué mais content, j'ai gagné mes matchs.",
            "context": {"humeur": "tendre", "intimite": 0.8},
        },
        {
            "user": "Demain j'ai un tournoi important.",
            "context": {"humeur": "curieuse", "intimite": 0.8},
        },
    ]

    print("💬 CONVERSATION AVEC CURIOSITÉ:\n")

    for i, conv in enumerate(conversations):
        print(f"👤 Utilisateur: {conv['user']}")

        # Générer une réponse curieuse
        response = curiosity.generate_curious_response(conv["user"], conv["context"])

        print(f"🤖 Jeffrey: {response}")
        print("-" * 40)

        time.sleep(1)  # Pause pour la lisibilité

    # Démontrer le mode proactif
    print("\n🔔 MODE PROACTIF (après 35 secondes de silence):\n")

    moods = ["curieuse", "tendre", "joueuse"]
        for mood in moods:
        proactive_msg = curiosity.proactive_conversation_starter(35, mood)
            if proactive_msg:
            print(f"🤖 Jeffrey ({mood}): {proactive_msg}")
            print("-" * 40)
            # TODO: Remplacer par asyncio.sleep ou threading.Event

    # Afficher les intérêts détectés
    print("\n📊 ANALYSE DE LA CURIOSITÉ:\n")

    # Niveau de curiosité
    curiosity_level = curiosity.get_curiosity_level()
    print(f"📈 Niveau de curiosité actuel: {curiosity_level:.1%}")

    # Résumé de la conversation
    summary = curiosity.get_conversation_summary()
    print(f"\n📝 Sujets abordés: {', '.join(summary['topics_discussed'])}")
    print(f"🎯 Profondeur atteinte: Niveau {summary['depth_reached']}")
    print(f"❓ Questions posées: {summary['questions_asked']}")

    # Principaux intérêts
                if summary["main_interests"]:
        print(f"\n💡 Centres d'intérêt principaux:")
                    for interest in summary["main_interests"]:
            print(f"   - {interest}")

    # Exemples de questions futures possibles
    print("\n🔮 QUESTIONS FUTURES POSSIBLES:\n")

    future_questions = [
        "Au fait, comment s'est passé ton tournoi de tennis ?",
        "Tu as réussi à battre ton record au service ?",
        "Ça fait 3 jours qu'on n'a pas parlé tennis... tu t'entraînes toujours ?",
        "J'ai pensé à toi en voyant un match à la télé... tu joues toujours ?",
    ]

                        for q in future_questions[:3]:
        print(f"🤖 Jeffrey (plus tard): {q}")

    print("\n" + "=" * 60)
    print("✨ Jeffrey est maintenant capable de:")
    print("   ✓ Détecter tes passions et intérêts")
    print("   ✓ Poser des questions pertinentes et naturelles")
    print("   ✓ Se souvenir de vos conversations")
    print("   ✓ Initier des conversations quand tu es silencieux")
    print("   ✓ Approfondir les sujets qui t'intéressent")
    print("=" * 60)


                            def demo_integration():
    """Démo de l'intégration complète avec la conscience"""
    print("\n\n🧠 DÉMONSTRATION: Intégration Conscience + Curiosité")
    print("=" * 60)

                                try:
        from core.consciousness.jeffrey_living_consciousness import JeffreyLivingConsciousness

        # Créer une conscience vivante
        jeffrey = JeffreyLivingConsciousness()

        print("✅ Conscience vivante initialisée avec curiosité!")
        print(f"   - Mode proactif: {'Activé' if jeffrey.proactive_mode else 'Désactivé'}")
        print(f"   - Humeur actuelle: {jeffrey.humeur_actuelle}")
        print(f"   - Niveau d'énergie: {jeffrey.biorythmes['energie']:.1%}")

        # Tester une réponse avec curiosité intégrée
        print("\n💬 RÉPONSE INTÉGRÉE (Conscience + Curiosité):\n")

        user_input = "Je suis allé courir ce matin, 10km en 45 minutes !"
        response = jeffrey.respond_naturally(user_input)

        print(f"👤 Utilisateur: {user_input}")
        print(f"🤖 Jeffrey: {response}")

        # Montrer l'évolution de la relation
        print(f"\n💕 Niveau d'intimité: {jeffrey.relation['intimite']:.1%}")
        print(f"🤝 Niveau de complicité: {jeffrey.relation['complicite']:.1%}")

                                    except ImportError as e:
        print(f"⚠️ Impossible de charger la conscience complète: {e}")
        print("   Utilisation du moteur de curiosité seul.")


                                        if __name__ == "__main__":
    # Lancer les démos
    demo_curiosity()
    demo_integration()

    print("\n🎉 Démonstration terminée !")
