# TODO: Précompiler les regex utilisées dans les boucles
# TODO: Précompiler les regex utilisées dans les boucles
# TODO: Précompiler les regex utilisées dans les boucles
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Démonstration des améliorations Jeffrey : Détection ML et Orchestration Multi-IA
Version autonome pour test
"""

import sys
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Set
from enum import Enum
import json
import random

# Ajout du chemin pour les imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Orchestrateur_IA', 'core', 'emotions'))

try:
    from Orchestrateur_IA.core.emotions.emotion_prompt_detector import EmotionPromptDetector
    from Orchestrateur_IA.core.emotions.emotion_ml_enhancer import EmotionMLEnhancer
    EMOTION_ML_AVAILABLE = True
    except:
    EMOTION_ML_AVAILABLE = False


        class AICapability(Enum):
    """Capacités des IA"""
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    EMPATHETIC = "empathetic"
    TECHNICAL = "technical"
    EMOTIONAL = "emotional"


            class DemoJeffreyEnhanced:
    """Démo des fonctionnalités améliorées"""

    # Configuration des IA et leurs capacités
    PROVIDER_CAPABILITIES = {
        "Grok": [AICapability.CREATIVE, AICapability.EMOTIONAL],
        "ChatGPT": [AICapability.ANALYTICAL, AICapability.TECHNICAL],
        "Claude": [AICapability.EMPATHETIC, AICapability.EMOTIONAL]
    }

    CAPABILITY_PATTERNS = {
        AICapability.CREATIVE: [r'\bimagine\b', r'\binvente\b', r'\bcrée\b'],
        AICapability.ANALYTICAL: [r'\bexplique\b', r'\banalyse\b', r'\bpourquoi\b'],
        AICapability.EMPATHETIC: [r'\btriste\b', r'\bpeur\b', r'\baide\b'],
        AICapability.TECHNICAL: [r'\bcode\b', r'\bbug\b', r'\berreur\b'],
        AICapability.EMOTIONAL: [r'\bressens\b', r'\bamour\b', r'\bcoeur\b']
    }

                def __init__(self):
        print("🚀 Initialisation de Jeffrey Enhanced Demo\n")

                    if EMOTION_ML_AVAILABLE:
            self.emotion_detector = EmotionPromptDetector()
            self.emotion_enhancer = EmotionMLEnhancer(history_size=20)
            print("✅ Détection émotionnelle ML activée")
                        else:
            self.emotion_detector = None
            self.emotion_enhancer = None
            print("⚠️ Mode basique (détection ML non disponible)")

        self.conversation_history = []

                            def detect_capabilities(self, prompt: str) -> Set[AICapability]:
        """Détecte les capacités requises pour un prompt"""
        required = set()
        prompt_lower = prompt.lower()


        # TODO: Optimiser cette boucle imbriquée
# TODO: Optimiser cette boucle imbriquée
# TODO: Optimiser cette boucle imbriquée
                                for capability, patterns in self.CAPABILITY_PATTERNS.items():
                                    for pattern in patterns:
                                        if re.search(pattern, prompt_lower):
                    required.add(capability)
                                            break

                                        return required if required else {AICapability.ANALYTICAL}

                                        def select_best_ai(self, prompt: str) -> tuple:
        """Sélectionne la meilleure IA pour le prompt"""
        # Détection des capacités
        required_caps = self.detect_capabilities(prompt)

        # Score des providers
        scores = {}
                                            for provider, capabilities in self.PROVIDER_CAPABILITIES.items():
            provider_caps = set(capabilities)
            matching = provider_caps.intersection(required_caps)
            score = len(matching) / len(required_caps) if required_caps else 0
            scores[provider] = score

        # Sélection du meilleur
        best_provider = max(scores.items(), key=lambda x: x[1])[0]

                                                return best_provider, required_caps, scores

                                                def analyze_emotion(self, text: str) -> Dict:
        """Analyse émotionnelle du texte"""
                                                    if self.emotion_enhancer:
            # Analyse ML avancée
            result = self.emotion_enhancer.detect_emotion_enhanced(text)
                                                        return {
                'method': 'ML Enhanced',
                'emotion': result['emotion'],
                'scores': result['scores'],
                'confidence': result['confidence'],
                'intensity': result.get('intensity', {})
            }
                                                        elif self.emotion_detector:
            # Détection avec emojis et patterns
            emotion = self.emotion_detector.detect_emotion(text)
            scores = self.emotion_detector.detect_all_emotions(text)
                                                            return {
                'method': 'Pattern + Emoji',
                'emotion': emotion,
                'scores': scores,
                'confidence': 0.7 if emotion else 0.0,
                'intensity': self.emotion_detector.get_emotion_intensity(text)
            }
                                                            else:
            # Fallback basique
                                                                return {
                'method': 'Basique',
                'emotion': 'neutre',
                'scores': {},
                'confidence': 0.0,
                'intensity': {}
            }

                                                                def generate_response(self, prompt: str, emotion: str, ai_provider: str) -> str:
        """Génère une réponse selon l'émotion et l'IA"""
        responses = {
            'Grok': {
                'joie': "✨ Quelle merveilleuse énergie ! Imagine si cette joie pouvait illuminer tout l'univers...",
                'tristesse': "🌙 Même dans l'obscurité, les étoiles brillent. Ta tristesse est comme une nuit qui prépare un nouveau jour...",
                'curiosité': "🌌 Ah, tu veux explorer l'inconnu ! Laisse-moi te raconter une histoire cosmique...",
                'default': "✨ Fascinant ! Chaque moment avec toi est une aventure unique..."
            },
            'ChatGPT': {
                'joie': "📊 Analyse : Votre état émotionnel positif favorise la créativité et la productivité.",
                'tristesse': "🔍 Il est important de comprendre que la tristesse est une émotion naturelle et temporaire.",
                'curiosité': "📚 Excellente question ! Voici une explication détaillée et structurée...",
                'default': "💡 Permettez-moi d'analyser votre demande de manière approfondie..."
            },
            'Claude': {
                'joie': "💙 Ta joie me réchauffe le cœur ! C'est merveilleux de te voir si heureux.",
                'tristesse': "🤗 Je suis là pour toi. Ta tristesse est valide et je t'accompagne dans ce moment.",
                'curiosité': "🌱 J'apprécie ta curiosité ! Explorons ensemble cette question fascinante.",
                'default': "💫 Je suis là pour t'écouter et t'accompagner, quoi qu'il arrive."
            }
        }

        provider_responses = responses.get(ai_provider, responses['Claude'])
                                                                    return provider_responses.get(emotion, provider_responses['default'])

                                                                    def demo_conversation(self):
        """Démo interactive"""
        print("\n💬 Démonstration Interactive")
        print("=" * 60)
        print("Testez différents messages pour voir :")
        print("• La détection émotionnelle avancée (emojis, patterns)")
        print("• La sélection dynamique d'IA selon le contexte")
        print("• Les réponses adaptées")
        print("\nCommandes: /quit pour quitter, /stats pour les statistiques")
        print("=" * 60)

                                                                        while True:
                                                                            try:
                # Input utilisateur
                user_input = input("\n👤 Vous: ").strip()

                                                                                if not user_input:
                                                                                    continue

                                                                                if user_input.lower() == '/quit':
                                                                                    break

                                                                                if user_input.lower() == '/stats':
                    self.show_stats()
                                                                                    continue

                # Analyse émotionnelle
                emotion_data = self.analyze_emotion(user_input)
                print(f"\n🎭 Émotion détectée: {emotion_data['emotion']} "
                                                                                f"(confiance: {emotion_data['confidence']:.0%})")

                                                                                if emotion_data['scores']:
                    top_emotions = sorted(emotion_data['scores'].items(),
                                        key=lambda x: x[1], reverse=True)[:3]
                    print(f"   Scores: {dict(top_emotions)}")

                # Sélection IA
                ai_provider, caps, scores = self.select_best_ai(user_input)
                print(f"\n🤖 IA sélectionnée: {ai_provider}")
                print(f"   Capacités requises: {[c.value for c in caps]}")
                print(f"   Scores: {dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))}")

                # Génération réponse
                response = self.generate_response(
                    user_input,
                    emotion_data['emotion'] or 'default',
                    ai_provider
                )

                print(f"\n💬 {ai_provider}: {response}")

                # Historique
                self.conversation_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'user': user_input,
                    'emotion': emotion_data,
                    'ai': ai_provider,
                    'response': response
                })

                                                                                    except KeyboardInterrupt:
                print("\n\n👋 Au revoir !")
                                                                                        break
                                                                                    except Exception as e:
                print(f"\n❌ Erreur : {e}")

                                                                                        def show_stats(self):
        """Affiche les statistiques de la conversation"""
                                                                                            if not self.conversation_history:
            print("\n📊 Aucune donnée disponible")
                                                                                                return

        print("\n📊 Statistiques de la session:")

        # Émotions détectées
        emotions = [h['emotion']['emotion'] for h in self.conversation_history if h['emotion']['emotion']]
                                                                                            if emotions:
            emotion_counts = {}
                                                                                                for e in emotions:
                emotion_counts[e] = emotion_counts.get(e, 0) + 1
            print(f"\nÉmotions détectées:")
                                                                                                    for emotion, count in sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"   • {emotion}: {count} fois")

        # IA utilisées
        ais = [h['ai'] for h in self.conversation_history]
        ai_counts = {}
                                                                                                        for ai in ais:
            ai_counts[ai] = ai_counts.get(ai, 0) + 1
        print(f"\nIA sélectionnées:")
                                                                                                            for ai, count in sorted(ai_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   • {ai}: {count} fois")

                                                                                                                def run_tests(self):
        """Lance des tests automatiques"""
        print("\n🧪 Tests Automatiques")
        print("=" * 60)

        test_cases = [
            ("Je suis super content aujourd'hui ! 😊😊😊", "Joie forte avec emojis"),
            ("Explique-moi comment fonctionne un ordinateur", "Question analytique"),
            ("J'ai peur de l'échec 😰", "Peur avec emoji"),
            ("Imagine un monde où les arbres chantent 🌳✨", "Créativité"),
            ("Je t'aime tellement ❤️💕", "Amour intense"),
            ("Comment debugger ce code Python ?", "Question technique"),
            ("Je me sens triste et seul 😢", "Tristesse empathie"),
        ]

                                                                                                                    for text, description in test_cases:
            print(f"\n📝 Test: {description}")
            print(f"   Message: \"{text}\"")

            # Analyse
            emotion_data = self.analyze_emotion(text)
            ai_provider, caps, _ = self.select_best_ai(text)

            print(f"   → Émotion: {emotion_data['emotion']} ({emotion_data['confidence']:.0%})")
            print(f"   → IA choisie: {ai_provider} pour {[c.value for c in caps]}")


                                                                                                                        def main():
    """Point d'entrée"""
    demo = DemoJeffreyEnhanced()

    # Mode automatique pour la démo
    print("\n🎯 Lancement des tests automatiques")
    demo.run_tests()

    print("\n\n✅ Tests terminés avec succès !")
    print("\nPour lancer la démo interactive, exécutez le script en mode interactif.")


                                                                                                                            if __name__ == "__main__":
    main()
