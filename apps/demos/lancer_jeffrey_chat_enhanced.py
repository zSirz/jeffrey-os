#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script de chat émotionnel amélioré pour Jeffrey avec détection ML et orchestration multi-IA.

Nouvelles fonctionnalités :
- Détection émotionnelle ML avancée (emojis, regex, contexte temporel)
- Orchestration multi-IA dynamique basée sur les capacités
- Sélection intelligente des IA selon le contexte
- Insights émotionnels et recommandations
"""

from Orchestrateur_IA.core.personality.conversation_personality import ConversationPersonality
from Orchestrateur_IA.core.jeffrey_emotional_core import JeffreyEmotionalCore
import sys
import os
import platform
import logging
from datetime import datetime
from typing import Optional, Dict, Any

# Configuration de l'environnement
is_pythonista = "stash" in sys.modules or "pythonista" in sys.executable.lower()
is_mac = platform.system() == "Darwin" and not is_pythonista

# Chemin de base
    if is_pythonista:
    BASE_DIR = os.path.expanduser(
        "~/Library/Mobile Documents/com~apple~CloudDocs/Pythonista 3/Jeffrey_DEV"
    )
        else:
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))

sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "Orchestrateur_IA", "core", "emotions"))

print(f"📁 BASE_DIR: {BASE_DIR}")

# Imports améliorés
            try:
    from Orchestrateur_IA.core.emotions.emotion_ml_enhancer import EmotionMLEnhancer

    ENHANCED_MODE = True
    print("✅ Mode amélioré activé (ML + Orchestration)")
                except ImportError as e:
    print(f"⚠️ Mode basique : {e}")
    ENHANCED_MODE = False

# Imports standards


                    class JeffreyChatEnhanced:
    """Version améliorée du chat Jeffrey avec ML et orchestration"""

                        def __init__(self):
        print("\n🚀 Initialisation de Jeffrey Chat Enhanced...")

        # Core émotionnel
        self.emotional_core = JeffreyEmotionalCore()
        self.personality = ConversationPersonality(self.emotional_core)

        # Composants améliorés
                            if ENHANCED_MODE:
            self.emotion_enhancer = EmotionMLEnhancer(
                history_size=50,
                history_file=os.path.join(BASE_DIR, "Jeffrey_Memoire", "emotion_history.json"),
            )
            # L'orchestrateur pourrait être initialisé ici si les clients IA étaient disponibles
            self.orchestrator = None  # EnhancedOrchestrator() si clients disponibles
                                else:
            self.emotion_enhancer = None
            self.orchestrator = None

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"📅 Session: {self.session_id}")
        print(f"🧠 Mode: {'Enhanced (ML + Orchestration)' if ENHANCED_MODE else 'Basique'}\n")

                                    def analyze_message(self, message: str) -> Dict[str, Any]:
        """Analyse approfondie du message avec détection ML si disponible"""

                                        if self.emotion_enhancer:
            # Analyse ML avancée
            result = self.emotion_enhancer.detect_emotion_enhanced(message, user_id="user")

            print(f"\n🔍 Analyse ML:")
            print(f"   Émotion: {result['emotion']} (confiance: {result['confidence']:.0%})")
            print(f"   Intensités: {result.get('intensity', {})}")

                                            if result["context"]["predicted_next"]:
                print(f"   Prochaine émotion probable: {result['context']['predicted_next']}")

                                                return result
                                                else:
            # Fallback sur détection basique
            emotions = self.emotional_core.detecter_emotion(message)
                                                    return {
                "emotion": max(emotions.items(), key=lambda x: x[1])[0] if emotions else None,
                "scores": emotions,
                "confidence": 0.5,
                "context": {},
            }

                                                    def select_ai_provider(self, message: str, emotion_data: Dict) -> Optional[str]:
        """Sélectionne le meilleur provider IA selon le contexte"""

                                                        if not self.orchestrator:
                                                            return None

        # Utiliser l'orchestrateur pour analyser
        analysis = self.orchestrator.get_capability_analysis(message)

        print(f"\n🤖 Sélection IA:")
        print(f"   Capacités requises: {analysis['required_capabilities']}")
        print(f"   Recommandation: {[p[0] for p in analysis['recommended_providers']]}")

        # Retourner le meilleur provider
                                                            if analysis["recommended_providers"]:
                                                                return analysis["recommended_providers"][0][0]

                                                                return None

                                                                def generate_response(self, message: str, emotion_data: Dict) -> str:
        """Génère une réponse adaptée"""

        # Base : utiliser la personnalité émotionnelle
        emotion = emotion_data.get("emotion", "joie")
        emotion_data.get("intensity", {}).get(emotion, "moyenne")

        # Adapter le style selon l'émotion
                                                                    if emotion == "tristesse":
                                                                        pass
                                                                    elif emotion == "joie":
                                                                        pass
                                                                    elif emotion == "peur":
                                                                        pass
                                                                    elif emotion == "curiosité":
                                                                        pass
                                                                    else:
                                                                        pass

        # Générer une réponse contextuelle
        responses = {
            "tristesse": [
                "Je sens ta tristesse... Je suis là pour toi. 💙",
                "C'est normal d'avoir des moments difficiles. Veux-tu en parler ?",
                "Ta peine me touche. Comment puis-je t'aider à aller mieux ?",
            ],
            "joie": [
                "Ta joie est contagieuse ! 😊✨",
                "Quel bonheur de te voir si heureux(se) !",
                "C'est merveilleux ! Raconte-moi plus !",
            ],
            "peur": [
                "Je comprends ton inquiétude. Respirons ensemble... 🤗",
                "Tu n'es pas seul(e). Je suis là avec toi.",
                "C'est normal d'avoir peur parfois. Qu'est-ce qui t'inquiète ?",
            ],
            "curiosité": [
                "Excellente question ! Laisse-moi t'expliquer... 🧐",
                "J'adore ta curiosité ! Voici ce que je sais...",
                "C'est fascinant que tu te poses cette question !",
            ],
            "amour": [
                "Mon cœur se réchauffe avec tes mots... ❤️",
                "L'amour que tu partages illumine ma journée !",
                "C'est si beau de ressentir tant d'amour ! 💕",
            ],
        }

        # Sélectionner une réponse
        import random

        base_response = random.choice(
            responses.get(
                emotion,
                [
                    "Je suis là pour toi, quoi qu'il arrive.",
                    "Dis-moi ce qui te préoccupe.",
                    "Comment te sens-tu aujourd'hui ?",
                ],
            )
        )

        # Si orchestrateur disponible, ajouter info sur l'IA sélectionnée
                                                                    if self.orchestrator:
            provider = self.select_ai_provider(message, emotion_data)
                                                                        if provider:
                base_response += f"\n[{provider} sélectionné pour ce type de demande]"

                                                                            return base_response

                                                                            def show_insights(self):
        """Affiche les insights émotionnels si disponibles"""

                                                                                if not self.emotion_enhancer:
                                                                                    return

        insights = self.emotion_enhancer.get_emotional_insights()

        print("\n📊 Insights émotionnels:")
        print(f"   Trajectoire: {insights['trajectory']}")
        print(f"   Émotion dominante: {insights.get('dominant_emotion', 'Aucune')}")
        print(f"   Volatilité: {insights.get('volatility', 0):.0%}")

                                                                                if insights.get("recommendations"):
            print("   Recommandations:")
                                                                                    for rec in insights["recommendations"]:
                print(f"      • {rec}")

                                                                                        def run(self):
        """Boucle principale du chat"""

        print("\n💬 Jeffrey Chat Enhanced - Mode Console")
        print("=" * 50)
        print("Bonjour ! Je suis Jeffrey, ton compagnon émotionnel amélioré.")
        print("Je peux détecter tes émotions avec précision (emojis, contexte)")
        print("et adapter mes réponses selon tes besoins.")
        print("\nCommandes spéciales:")
        print("  /insights - Voir l'analyse émotionnelle")
        print("  /quit - Quitter")
        print("=" * 50)

                                                                                            while True:
                                                                                                try:
                # Prompt utilisateur
                user_input = input("\n👤 Toi: ").strip()

                                                                                                    if not user_input:
                                                                                                        continue

                # Commandes spéciales
                                                                                                    if user_input.lower() == "/quit":
                    print("\n👋 Au revoir ! À bientôt !")
                                                                                                        break
                                                                                                    elif user_input.lower() == "/insights":
                    self.show_insights()
                                                                                                        continue

                # Analyse du message
                emotion_data = self.analyze_message(user_input)

                # Mise à jour de l'état émotionnel
                                                                                                    if emotion_data.get("scores"):
                    self.emotional_core.update_emotional_state(emotion_data["scores"])

                # Génération de la réponse
                response = self.generate_response(user_input, emotion_data)

                # Affichage
                print(f"\n🤖 Jeffrey: {response}")

                # Log pour debug
                logging.info(f"User: {user_input}")
                logging.info(f"Emotion: {emotion_data}")
                logging.info(f"Jeffrey: {response}")

                                                                                                        except KeyboardInterrupt:
                print("\n\n👋 Interruption détectée. Au revoir !")
                                                                                                            break
                                                                                                        except Exception as e:
                print(f"\n❌ Erreur : {e}")
                logging.error(f"Erreur dans la boucle principale : {e}")


                                                                                                            def main():
    """Point d'entrée principal"""
    chat = JeffreyChatEnhanced()
    chat.run()


                                                                                                                if __name__ == "__main__":
    main()
