#!/usr/bin/env python3
"""
Interface de Chat Moderne pour Jeffrey OS V1.1 - VERSION ULTIMATE
Utilise l'architecture AGI Orchestrator avec apprentissage adaptatif
+ Analyse émotionnelle enrichie + Prédictions + Insights avancés
"""

import asyncio
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from jeffrey.core.self_learning import get_learning_module

    from jeffrey.core.orchestration.agi_orchestrator import AGIOrchestrator
except ImportError as e:
    print(f"❌ Erreur d'import : {e}")
    print("Vérifiez que vous êtes dans le bon dossier et que PYTHONPATH est correct.")
    sys.exit(1)


class EmotionAnalyzer:
    """Analyseur émotionnel enrichi avec détection ML-style"""

    def __init__(self):
        # Patterns emojis par émotion
        self.emoji_patterns = {
            'joie': ['😊', '😄', '😁', '🎉', '✨', '💖', '❤️', '😍', '🥰'],
            'tristesse': ['😢', '😭', '😔', '😞', '💔', '😿', '🥺'],
            'peur': ['😨', '😰', '😱', '😓', '😖'],
            'colère': ['😠', '😡', '🤬', '😤', '💢'],
            'surprise': ['😮', '😲', '🤯', '😳'],
            'dégoût': ['🤢', '🤮', '😒', '😑'],
            'curiosité': ['🤔', '🧐', '🤨', '❓', '❔'],
            'amour': ['❤️', '💕', '💖', '💗', '💘', '😍', '🥰'],
        }

        # Patterns de mots-clés
        self.keyword_patterns = {
            'joie': ['heureux', 'content', 'génial', 'super', 'top', 'youpi', 'yes', 'merveilleux', 'excellent'],
            'tristesse': ['triste', 'malheureux', 'déprimé', 'seul', 'vide', 'perdu', 'pleurer', 'chagrin'],
            'peur': ['peur', 'angoisse', 'inquiet', 'stressé', 'anxieux', 'terreur', 'panique'],
            'colère': ['énervé', 'furieux', 'rage', 'injuste', 'agacé', 'irrité', 'colère'],
            'curiosité': ['pourquoi', 'comment', 'qu\'est-ce', 'intéressant', 'curieux', 'fascinant'],
            'amour': ['aimer', 'adorer', 'amour', 'affection', 'tendresse', 'chérir'],
        }

        # Historique pour prédiction
        self.emotion_history = deque(maxlen=20)

    def analyze_enhanced(self, text: str) -> dict:
        """Analyse enrichie avec détection multi-niveaux"""

        scores = {}

        # 1. Détection par emojis
        for emotion, emojis in self.emoji_patterns.items():
            emoji_count = sum(text.count(e) for e in emojis)
            if emoji_count > 0:
                scores[emotion] = scores.get(emotion, 0) + emoji_count * 0.4

        # 2. Détection par mots-clés
        text_lower = text.lower()
        for emotion, keywords in self.keyword_patterns.items():
            keyword_count = sum(text_lower.count(k) for k in keywords)
            if keyword_count > 0:
                scores[emotion] = scores.get(emotion, 0) + keyword_count * 0.3

        # 3. Patterns de ponctuation
        if '!' in text:
            scores['joie'] = scores.get('joie', 0) + 0.1
        if '?' in text:
            scores['curiosité'] = scores.get('curiosité', 0) + 0.15
        if text.endswith('...'):
            scores['tristesse'] = scores.get('tristesse', 0) + 0.1

        # 4. Longueur du message (contexte)
        if len(text) > 100:
            scores['curiosité'] = scores.get('curiosité', 0) + 0.05

        # Normaliser les scores
        if scores:
            total = sum(scores.values())
            scores = {k: v / total for k, v in scores.items()}

        # Émotion dominante
        dominant = max(scores.items(), key=lambda x: x[1])[0] if scores else 'neutre'
        confidence = scores.get(dominant, 0)

        # Ajouter à l'historique
        self.emotion_history.append(dominant)

        # Prédire la prochaine émotion
        predicted_next = self._predict_next_emotion()

        # Calculer la trajectoire émotionnelle
        trajectory = self._calculate_trajectory()

        # Calculer la volatilité
        volatility = self._calculate_volatility()

        return {
            'emotion': dominant,
            'confidence': confidence,
            'scores': scores,
            'predicted_next': predicted_next,
            'trajectory': trajectory,
            'volatility': volatility,
            'context': {
                'has_emoji': any(e in text for emojis in self.emoji_patterns.values() for e in emojis),
                'has_question': '?' in text,
                'is_long': len(text) > 100,
            },
        }

    def _predict_next_emotion(self) -> str:
        """Prédire la prochaine émotion probable"""
        if len(self.emotion_history) < 3:
            return None

        # Patterns de transition
        transitions = {
            'tristesse': ['joie', 'neutre'],
            'joie': ['joie', 'neutre'],
            'peur': ['soulagement', 'neutre'],
            'colère': ['calme', 'tristesse'],
            'curiosité': ['joie', 'surprise'],
        }

        last_emotion = self.emotion_history[-1]
        return transitions.get(last_emotion, ['neutre'])[0]

    def _calculate_trajectory(self) -> str:
        """Calculer la trajectoire émotionnelle"""
        if len(self.emotion_history) < 5:
            return "stable"

        recent = list(self.emotion_history)[-5:]
        positive_emotions = ['joie', 'amour', 'curiosité']
        negative_emotions = ['tristesse', 'peur', 'colère']

        positive_count = sum(1 for e in recent if e in positive_emotions)
        negative_count = sum(1 for e in recent if e in negative_emotions)

        if positive_count > negative_count + 1:
            return "ascendante (vers le positif)"
        elif negative_count > positive_count + 1:
            return "descendante (vers le négatif)"
        else:
            return "stable"

    def _calculate_volatility(self) -> float:
        """Calculer la volatilité émotionnelle (0-1)"""
        if len(self.emotion_history) < 5:
            return 0.0

        recent = list(self.emotion_history)[-10:]
        unique_emotions = len(set(recent))

        # Plus il y a d'émotions différentes, plus c'est volatile
        return min(unique_emotions / 5, 1.0)

    def get_insights(self) -> dict:
        """Obtenir des insights émotionnels"""
        if len(self.emotion_history) < 3:
            return {
                'trajectory': 'Données insuffisantes',
                'recommendations': ['Continuez à discuter pour que je vous connaisse mieux !'],
            }

        trajectory = self._calculate_trajectory()
        volatility = self._calculate_volatility()
        dominant = max(set(self.emotion_history), key=list(self.emotion_history).count)

        recommendations = []

        if volatility > 0.7:
            recommendations.append("Vos émotions changent rapidement. Prenez un moment pour respirer.")

        if trajectory == "descendante (vers le négatif)":
            recommendations.append("Je remarque une tendance négative. Voulez-vous en parler ?")

        if dominant == 'tristesse' and list(self.emotion_history).count('tristesse') > 5:
            recommendations.append("Vous semblez triste depuis un moment. Je suis là pour vous.")

        if dominant == 'joie':
            recommendations.append("Votre joie est inspirante ! Continuez ainsi !")

        return {
            'trajectory': trajectory,
            'dominant_emotion': dominant,
            'volatility': volatility,
            'recommendations': recommendations or ["Tout va bien ! Continuons notre conversation."],
        }


class JeffreyChatModern:
    """Interface de chat moderne pour Jeffrey OS - VERSION ULTIMATE"""

    def __init__(self):
        print("\n🚀 Initialisation de Jeffrey OS V1.1 ULTIMATE...")

        try:
            # Initialiser l'orchestrateur AGI
            self.orchestrator = AGIOrchestrator()

            # Récupérer le module d'apprentissage
            self.learning = get_learning_module()

            # Analyseur émotionnel enrichi
            self.emotion_analyzer = EmotionAnalyzer()
            print("✅ Analyseur émotionnel enrichi activé")

            # Configuration utilisateur
            self.user_id = "David"
            self.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Compteurs
            self.interaction_count = 0

            print("✅ Jeffrey OS V1.1 ULTIMATE prêt !\n")

        except Exception as e:
            print(f"❌ Erreur lors de l'initialisation : {e}")
            raise

    async def chat_loop(self):
        """Boucle principale de conversation"""

        self._show_welcome()

        while True:
            try:
                # Prompt utilisateur
                user_input = input("\n💭 Vous: ").strip()

                # Vérifier input vide
                if not user_input:
                    continue

                # Commandes spéciales
                if await self._handle_command(user_input):
                    continue

                # Traiter le message avec Jeffrey
                await self._process_message(user_input)

                # Compteur
                self.interaction_count += 1

                # Stats périodiques
                if self.interaction_count % 5 == 0:
                    self._show_quick_stats()

            except KeyboardInterrupt:
                print("\n\n👋 Au revoir ! À bientôt !")
                break
            except Exception as e:
                print(f"\n❌ Erreur : {e}")
                import traceback

                traceback.print_exc()

    async def _process_message(self, user_input: str):
        """Traite un message utilisateur avec analyse enrichie"""

        try:
            # Analyse émotionnelle enrichie
            emotion_analysis = self.emotion_analyzer.analyze_enhanced(user_input)

            print("\n🔍 Analyse émotionnelle:")
            print(f"   Émotion: {emotion_analysis['emotion']} ({emotion_analysis['confidence']:.0%})")

            if emotion_analysis['predicted_next']:
                print(f"   Prédiction: {emotion_analysis['predicted_next']}")

            print(f"   Trajectoire: {emotion_analysis['trajectory']}")
            print(f"   Volatilité: {emotion_analysis['volatility']:.0%}")

            # Appeler l'orchestrateur AGI
            result = await self.orchestrator.process(
                user_input=user_input,
                user_id=self.user_id,
                metadata={
                    'conversation_id': self.conversation_id,
                    'timestamp': datetime.now().isoformat(),
                    'emotion_analysis': emotion_analysis,
                },
            )

            # Extraire la réponse
            response = result.get('response', 'Hmm... je réfléchis...')

            # Enrichir la réponse selon l'émotion détectée
            response = self._enrich_response(response, emotion_analysis)

            # Afficher la réponse
            print(f"\n🤖 Jeffrey: {response}")

            # Afficher l'état émotionnel de Jeffrey
            emotional_state = result.get('emotional_state', {})
            if emotional_state:
                emotion = emotional_state.get('primary_emotion', 'neutre')
                intensity = emotional_state.get('intensity', 0) * 100
                print(f"\n   💫 [État de Jeffrey: {emotion} ({intensity:.0f}%)]")

        except Exception as e:
            print(f"\n❌ Erreur lors du traitement : {e}")
            import traceback

            traceback.print_exc()

    def _enrich_response(self, response: str, emotion_analysis: dict) -> str:
        """Enrichit la réponse selon l'analyse émotionnelle"""

        emotion = emotion_analysis['emotion']
        confidence = emotion_analysis['confidence']

        # Ajouter des emojis contextuels si confiance élevée
        if confidence > 0.7:
            emoji_map = {
                'joie': ' 😊',
                'tristesse': ' 💙',
                'peur': ' 🤗',
                'curiosité': ' 🤔',
                'amour': ' ❤️',
                'colère': ' 🌸',  # Apaisant
            }

            if emotion in emoji_map and emoji_map[emotion] not in response:
                response += emoji_map[emotion]

        # Adapter le ton selon la trajectoire
        trajectory = emotion_analysis['trajectory']

        if trajectory == "descendante (vers le négatif)" and '?' not in response:
            response += " Comment puis-je t'aider à te sentir mieux ?"

        return response

    async def _handle_command(self, command: str) -> bool:
        """Gère les commandes spéciales. Retourne True si c'était une commande."""

        cmd = command.lower()

        # Quitter
        if cmd in ['/quit', '/exit', '/q']:
            print("\n👋 Au revoir ! À bientôt !")
            sys.exit(0)

        # Effacer l'écran
        if cmd == '/clear':
            import os

            os.system('clear' if os.name != 'nt' else 'cls')
            return True

        # Stats Jeffrey
        if cmd == '/stats':
            self._show_system_stats()
            return True

        # Stats apprentissage
        if cmd in ['/stats_learn', '/learn', '/learning']:
            self._show_learning_stats()
            return True

        # Insights émotionnels
        if cmd in ['/insights', '/emotions', '/analyse']:
            self._show_emotional_insights()
            return True

        # Aide
        if cmd in ['/help', '/aide', '/?']:
            self._show_help()
            return True

        return False

    def _show_welcome(self):
        """Affiche le message de bienvenue"""
        print("=" * 70)
        print("💬 JEFFREY OS V1.1 ULTIMATE - CHAT INTERFACE")
        print("=" * 70)
        print("\n👋 Bonjour ! Je suis Jeffrey, votre compagnon IA émotionnel.")
        print("\n🧠 Capacités activées :")
        print("   ✓ Mémoire contextuelle avancée (Memory V2.0)")
        print("   ✓ 15 systèmes de conscience AGI")
        print("   ✓ Apprentissage adaptatif en temps réel")
        print("   ✓ Analyse émotionnelle hybride ENRICHIE")
        print("   ✓ Prédictions émotionnelles")
        print("   ✓ Insights et recommandations personnalisés")
        self._show_help()

    def _show_help(self):
        """Affiche l'aide"""
        print("\n📖 Commandes disponibles :")
        print("   /stats        → Statistiques du système")
        print("   /learn        → Statistiques d'apprentissage")
        print("   /insights     → Analyse émotionnelle détaillée")
        print("   /clear        → Effacer l'écran")
        print("   /help         → Afficher cette aide")
        print("   /quit         → Quitter")
        print("\n" + "-" * 70)

    def _show_quick_stats(self):
        """Affiche des stats rapides"""
        try:
            stats = self.learning.get_learning_stats()
            print(
                f"\n📊 [{self.interaction_count} msgs] "
                f"Qualité: {stats['avg_response_quality']:.0%} | "
                f"Patterns: {stats['patterns_learned']}"
            )
        except:
            pass

    def _show_system_stats(self):
        """Affiche les statistiques complètes du système"""
        print("\n" + "=" * 70)
        print("📊 STATISTIQUES SYSTÈME JEFFREY OS")
        print("=" * 70)

        try:
            # Stats orchestrateur
            status = self.orchestrator.get_system_status()

            # Mémoire
            memory_stats = status.get('memory_stats', {})
            print("\n🧠 MÉMOIRE:")
            print(f"   Total: {memory_stats.get('total_memories', 0)}")
            print(f"   Récentes: {memory_stats.get('recent_memories', 0)}")

            # Memory V2
            if status.get('memory_v2_enabled'):
                mem_v2 = status.get('memory_v2_stats', {})
                print(f"   Memory V2: {mem_v2.get('total_memories', 0)} mémoires")

            # Performance
            perf = status.get('performance_metrics', {})
            print("\n⚡ PERFORMANCE:")
            print(f"   Requêtes: {perf.get('total_requests', 0)}")
            print(f"   Temps moyen: {perf.get('avg_response_time', 0):.3f}s")

            # Systèmes AGI
            agi = status.get('agi_systems_active', [])
            print(f"\n🤖 SYSTÈMES AGI: {len(agi)} actifs")
            if agi:
                for i, system in enumerate(agi[:8], 1):
                    print(f"   {i}. {system}")
                if len(agi) > 8:
                    print(f"   ... et {len(agi) - 8} autres")

            print("\n" + "=" * 70)

        except Exception as e:
            print(f"\n❌ Erreur stats système : {e}")

    def _show_learning_stats(self):
        """Affiche les statistiques d'apprentissage"""
        print("\n" + "=" * 70)
        print("📚 STATISTIQUES D'APPRENTISSAGE")
        print("=" * 70)

        try:
            stats = self.learning.get_learning_stats()

            # Général
            print("\n📈 GÉNÉRAL:")
            print(f"   Interactions: {stats['total_interactions']}")
            print(f"   Réussies: {stats['successful_interactions']}")
            print(f"   Patterns: {stats['patterns_learned']}")
            print(f"   Qualité moyenne: {stats['avg_response_quality']:.1%}")
            print(f"   Amélioration: {stats['improvement_rate']:.2f}%")

            # Distribution
            dist = stats.get('quality_distribution', {})
            if dist:
                print("\n📊 QUALITÉ DES RÉPONSES:")
                for niveau, count in dist.items():
                    barre = "█" * count
                    print(f"   {niveau:10s}: {barre} ({count})")

            # Top patterns
            top = stats.get('top_patterns', [])
            if top:
                print("\n🏆 TOP 5 PATTERNS:")
                for i, p in enumerate(top[:5], 1):
                    print(f"   {i}. {p['input_type']:20s} ({p['effectiveness']:.0%})")

            # Performance récente
            perf = stats.get('recent_performance', {})
            if perf:
                print("\n📊 PERFORMANCE RÉCENTE:")
                print(f"   Qualité: {perf.get('avg_quality', 0):.1%}")
                print(f"   Consistance: {perf.get('consistency', 0):.1%}")
                print(f"   Tendance: {perf.get('trend', 'N/A')}")

            print("\n" + "=" * 70)

        except Exception as e:
            print(f"\n❌ Erreur stats apprentissage : {e}")

    def _show_emotional_insights(self):
        """Affiche les insights émotionnels détaillés"""
        print("\n" + "=" * 70)
        print("💭 INSIGHTS ÉMOTIONNELS")
        print("=" * 70)

        try:
            insights = self.emotion_analyzer.get_insights()

            print(f"\n📈 Trajectoire: {insights['trajectory']}")
            print(f"🎯 Émotion dominante: {insights.get('dominant_emotion', 'Aucune')}")
            print(f"📊 Volatilité: {insights.get('volatility', 0):.0%}")

            if insights.get('recommendations'):
                print("\n💡 Recommandations:")
                for rec in insights['recommendations']:
                    print(f"   • {rec}")

            # Historique émotionnel
            if len(self.emotion_analyzer.emotion_history) > 0:
                print("\n📜 Historique récent:")
                recent = list(self.emotion_analyzer.emotion_history)[-10:]
                print(f"   {' → '.join(recent)}")

            print("\n" + "=" * 70)

        except Exception as e:
            print(f"\n❌ Erreur insights : {e}")


async def main():
    """Point d'entrée principal"""
    try:
        chat = JeffreyChatModern()
        await chat.chat_loop()
    except Exception as e:
        print(f"\n❌ Erreur fatale : {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Programme interrompu. Au revoir !")
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        import traceback

        traceback.print_exc()
