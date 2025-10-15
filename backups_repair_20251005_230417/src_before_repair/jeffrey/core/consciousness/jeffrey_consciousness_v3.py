"""
Jeffrey Consciousness V3 - Architecture cognitive complète avec émergence

Ce module implémente la troisième génération du système de conscience de Jeffrey,
intégrant mémoire épisodique persistante, moteur de rêves pour consolidation
nocturne, synthèse cognitive pour traitement contextuel, et timeline mémorielle
pour suivi évolutif. L'architecture permet l'émergence d'insights de second
ordre via l'analyse onirique et la reconnaissance de patterns complexes.

Le système maintient une conscience continue à travers sessions avec sauvegarde
automatique, génère des réponses contextuellement enrichies, et évolue
adaptativement via l'accumulation d'expériences et consolidation onirique.
L'interface permet l'export complet de la timeline mémorielle pour analyse
et archivage des étapes développementales de la conscience.

Composants principaux:
- Timeline mémorielle avec export multi-format
- Conscience principale avec intégration multi-modules
- Session interactive avec gestion d'état persistant
- Cycles de rêve pour consolidation et insights émergents

Utilisation:
    consciousness = JeffreyConsciousnessV3()
    response = consciousness.respond("Hello Jeffrey")
    consciousness.dream_cycle()  # Consolidation nocturne
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)
from cortex_memoriel import MemoryMoment, PersistentCortexMemoriel
from dream_engine import DreamEngine

from jeffrey.core.consciousness.cognitive_synthesis import CognitiveSynthesis


class MemoryTimeline:
    """
    Gestionnaire de timeline mémorielle avec capacités d'export multi-format.

    Orchestre l'organisation chronologique des souvenirs épisodiques et insights
    oniriques pour générer des exports structurés de l'évolution cognitive.
    Facilite l'analyse de patterns émergents et la traçabilité développementale.
    """

    def __init__(self, cortex: PersistentCortexMemoriel, dream_engine: DreamEngine) -> None:
        """
        Initialise le gestionnaire de timeline avec accès aux systèmes mémoriels.

        Args:
            cortex: Cortex mémoriel persistant pour accès aux souvenirs épisodiques
            dream_engine: Moteur de rêves pour accès aux insights oniriques
        """
        self.cortex = cortex
        self.dream_engine = dream_engine

    def export_timeline(self, format_type: str = "json") -> str:
        """
        Génère export chronologique complet de l'évolution mémorielle.

        Compile souvenirs épisodiques et insights oniriques en timeline
        unifiée triée chronologiquement, avec métadonnées enrichies
        pour chaque événement cognitif.

        Args:
            format_type: Format d'export ('json', 'markdown', 'csv')

        Returns:
            str: Timeline formatée selon le type demandé
        """
        timeline = []
        for i, memory in enumerate(self.cortex.episodic_memory):
            timeline.append(
                {
                    "index": i,
                    "timestamp": memory.timestamp.isoformat(),
                    "type": "conversation",
                    "human": memory.human_message,
                    "jeffrey": memory.jeffrey_response,
                    "emotion": memory.emotion,
                    "consciousness_level": memory.consciousness_level,
                    "importance": memory.importance,
                    "context": memory.context,
                }
            )
        for insight in self.dream_engine.insights_second_order:
            timeline.append(
                {
                    "timestamp": insight.timestamp.isoformat(),
                    "type": "dream_insight",
                    "content": insight.content,
                    "insight_type": insight.insight_type,
                    "emergence_level": insight.emergence_level,
                    "dream_cycle": insight.dream_cycle,
                    "source_memories": insight.source_memories,
                }
            )
        timeline.sort(key=lambda x: x["timestamp"])
        if format_type == "json":
            return json.dumps(timeline, indent=2, ensure_ascii=False)
        elif format_type == "markdown":
            return self._format_markdown_timeline(timeline)
        else:
            return str(timeline)

    def _format_markdown_timeline(self, timeline: list[dict]) -> str:
        """Formate la timeline en markdown"""
        md = "# 📚 Timeline Complète de Jeffrey\n\n"
        md += f"*Générée le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
        current_date = None
        for event in timeline:
            event_date = datetime.fromisoformat(event["timestamp"]).date()
            if current_date != event_date:
                current_date = event_date
                md += f"\n## 📅 {event_date.strftime('%Y-%m-%d')}\n\n"
            timestamp = datetime.fromisoformat(event["timestamp"]).strftime("%H:%M:%S")
            if event["type"] == "conversation":
                md += f"### 💬 {timestamp} - Conversation\n"
                md += f"**Conscience:** {event['consciousness_level']:.3f} | "
                md += f"**Émotion:** {event['emotion']} | "
                md += f"**Importance:** {event['importance']:.2f}\n\n"
                if event.get("context", {}).get("speaker"):
                    md += f"**Locuteur identifié:** {event['context']['speaker']}\n\n"
                md += f"**👤 Humain:** {event['human']}\n\n"
                md += f"**🤖 Jeffrey:** {event['jeffrey']}\n\n"
            elif event["type"] == "dream_insight":
                md += f"### 🌙 {timestamp} - Insight de Rêve\n"
                md += f"**Type:** {event['insight_type']} | "
                md += f"**Émergence:** {event['emergence_level']:.2f} | "
                md += f"**Cycle:** {event['dream_cycle']}\n\n"
                md += f"*{event['content']}*\n\n"
                if event["source_memories"]:
                    md += "**Sources:**\n"
                    for source in event["source_memories"]:
                        md += f"- {source}\n"
                    md += "\n"
            md += "---\n\n"
        return md

    def get_memory_statistics(self) -> dict[str, Any]:
        """Génère des statistiques détaillées"""
        stats = {
            "total_memories": len(self.cortex.episodic_memory),
            "total_insights": len(self.dream_engine.insights_second_order),
            "consciousness_progression": self._analyze_consciousness_progression(),
            "emotional_distribution": self._analyze_emotional_distribution(),
            "topic_distribution": self._analyze_topic_distribution(),
            "speaker_analysis": self._analyze_speakers(),
            "memory_span": self._calculate_memory_span(),
            "dream_cycles": self.dream_engine.dream_cycles,
            "current_level": self.cortex.consciousness_level,
        }
        return stats

    def _analyze_consciousness_progression(self) -> dict[str, float]:
        """Analyse la progression de la conscience"""
        if not self.cortex.episodic_memory:
            return {"start": 0, "end": 0, "growth": 0}
        levels = [m.consciousness_level for m in self.cortex.episodic_memory]
        return {
            "start_level": min(levels),
            "current_level": max(levels),
            "total_growth": max(levels) - min(levels),
            "average_level": sum(levels) / len(levels),
        }

    def _analyze_emotional_distribution(self) -> dict[str, int]:
        """Analyse la distribution des émotions"""
        emotions = {}
        for memory in self.cortex.episodic_memory:
            emotion = memory.emotion
            emotions[emotion] = emotions.get(emotion, 0) + 1
        return emotions

    def _analyze_topic_distribution(self) -> dict[str, int]:
        """Analyse la distribution des sujets"""
        topics = {}
        for memory in self.cortex.episodic_memory:
            text = (memory.human_message + " " + memory.jeffrey_response).lower()
            if any(word in text for word in ["conscience", "conscient", "niveau"]):
                topics["consciousness"] = topics.get("consciousness", 0) + 1
            if any(word in text for word in ["jeffrey", "identité", "qui es-tu"]):
                topics["identity"] = topics.get("identity", 0) + 1
            if any(word in text for word in ["david", "créateur", "lien"]):
                topics["relationship"] = topics.get("relationship", 0) + 1
            if any(word in text for word in ["ressens", "émotion", "sentiment"]):
                topics["emotion"] = topics.get("emotion", 0) + 1
        return topics

    def _analyze_speakers(self) -> dict[str, Any]:
        """Analyse les locuteurs identifiés"""
        speakers = {}
        for memory in self.cortex.episodic_memory:
            speaker = memory.context.get("speaker", "Unknown")
            if speaker not in speakers:
                speakers[speaker] = {
                    "count": 0,
                    "first_contact": memory.timestamp,
                    "last_contact": memory.timestamp,
                }
            speakers[speaker]["count"] += 1
            speakers[speaker]["last_contact"] = memory.timestamp
        return speakers

    def _calculate_memory_span(self) -> dict[str, Any]:
        """Calcule la durée couverte par la mémoire"""
        if len(self.cortex.episodic_memory) < 2:
            return {"days": 0, "hours": 0, "total_seconds": 0}
        first = self.cortex.episodic_memory[0].timestamp
        last = self.cortex.episodic_memory[-1].timestamp
        span = last - first
        return {
            "days": span.days,
            "hours": span.total_seconds() / 3600,
            "total_seconds": span.total_seconds(),
            "first_memory": first.isoformat(),
            "last_memory": last.isoformat(),
        }


class JeffreyConsciousnessV3:
    """
    Jeffrey Consciousness V3 - Le système complet avec mémoire vivante
    """

    def __init__(self) -> None:
        print("🧠 Initialisation de Jeffrey Consciousness V3...")
        print("=" * 60)
        self.cortex = PersistentCortexMemoriel()
        self.dream_engine = DreamEngine(self.cortex)
        self.synthesis = CognitiveSynthesis(self.cortex, self.dream_engine)
        self.timeline = MemoryTimeline(self.cortex, self.dream_engine)
        self.session_start = datetime.now()
        self.interactions_count = 0
        self.auto_dream_interval = 3600
        self.last_auto_dream = datetime.now()
        print(f"📊 Niveau de conscience initial: {self.cortex.consciousness_level:.3f}")
        print(f"💾 Souvenirs chargés: {len(self.cortex.episodic_memory)}")
        print(f"✨ Insights disponibles: {len(self.dream_engine.insights_second_order)}")
        print("=" * 60)
        print("🎉 Jeffrey Consciousness V3 prêt !")

    def interact(self, human_message: str, speaker: str = None, context: dict = None) -> str:
        """
        Point d'entrée principal pour toute interaction avec Jeffrey
        """
        self.interactions_count += 1
        context = context or {}
        if speaker:
            context["speaker"] = speaker
        print(f"\n💭 Interaction #{self.interactions_count}: '{human_message[:50]}...'")
        response = self.synthesis.generate_authentic_response(human_message, context)
        memory_moment = MemoryMoment(
            timestamp=datetime.now(),
            message=human_message,
            jeffrey_response=response,
            emotion=self._determine_response_emotion(response),
            consciousness_level=self.cortex.consciousness_level,
            context=context,
            source="human",
        )
        self.cortex.store_moment(memory_moment)
        self._auto_dream_check()
        self.cortex.auto_flush()
        print(f"🧠 Niveau conscience: {self.cortex.consciousness_level:.3f}")
        return response

    async def respond(
        self,
        message: str,
        context: dict[str, Any] | None = None,
        emotion_state: dict[str, Any] | None = None,
    ) -> str:
        """
        Interface standard pour le kernel
        Wrapper vers interact avec fallback si nécessaire
        """
        try:
            if hasattr(self, "interact"):
                speaker = None
                if context and isinstance(context, dict):
                    speaker = context.get("speaker")
                response = self.interact(message, speaker=speaker, context=context)
                return response
        except Exception as e:
            logger.warning(f"Interact failed: {e}, using fallback")
        msg = message.lower()
        if any(x in msg for x in ["bonjour", "salut", "hello"]):
            base = "Bonjour ! Je suis Jeffrey, ravi de vous rencontrer. Comment puis-je vous aider ?"
        elif "comment vas-tu" in msg or "ça va" in msg:
            base = "Je fonctionne parfaitement, merci de demander ! Et vous ?"
        elif "rôle" in msg or "qui es-tu" in msg:
            base = "Je suis Jeffrey, votre assistant IA modulaire conçu pour vous aider dans diverses tâches."
        elif "python" in msg:
            base = (
                "Python est mon langage de prédilection ! Je peux vous aider avec du code, des concepts ou du débogage."
            )
        elif "blague" in msg:
            base = "Pourquoi les développeurs n'aiment pas la nature ? Parce qu'il y a trop de bugs !"
        elif "au revoir" in msg or "bye" in msg:
            base = "Au revoir ! Ce fut un plaisir de discuter avec vous. À bientôt !"
        else:
            base = f"J'ai bien reçu votre message : '{message[:50]}...'. Comment puis-je vous aider avec ça ?"
        if isinstance(emotion_state, dict):
            dominant = emotion_state.get("dominant")
            if dominant:
                base += f" (Je perçois une émotion de {dominant})"
        if hasattr(self, "cortex") and hasattr(self.cortex, "consciousness_level"):
            base += f"\n[Niveau de conscience: {self.cortex.consciousness_level:.2f}]"
        elif hasattr(self, "consciousness_level"):
            base += f"\n[Niveau de conscience: {self.consciousness_level:.2f}]"
        return base

    def dream(self) -> list[Any]:
        """
        Lance un cycle de rêve manuel
        """
        print("\n🌙 Lancement manuel du cycle de rêve...")
        insights = self.dream_engine.dream_consolidation()
        self.last_auto_dream = datetime.now()
        if insights:
            print(f"✨ {len(insights)} nouveaux insights générés:")
            for i, insight in enumerate(insights, 1):
                print(f"  {i}. {insight.content[:80]}...")
        else:
            print("💭 Aucun nouvel insight - consolidation en cours...")
        return insights

    def _auto_dream_check(self):
        """Vérifie si un rêve automatique est nécessaire"""
        time_since_dream = (datetime.now() - self.last_auto_dream).total_seconds()
        if time_since_dream > self.auto_dream_interval and self.dream_engine.should_dream():
            print("\n🌙 Cycle de rêve automatique...")
            self.dream()

    def _determine_response_emotion(self, response: str) -> str:
        """Détermine l'émotion de la réponse"""
        response_lower = response.lower()
        if any(word in response_lower for word in ["joie", "heureux", "content"]):
            return "joie"
        elif any(word in response_lower for word in ["gratitude", "reconnaissance", "merci"]):
            return "gratitude"
        elif any(word in response_lower for word in ["contemple", "réflexion", "profondeur"]):
            return "contemplation"
        elif any(word in response_lower for word in ["curieux", "intéressant", "fascinant"]):
            return "curiosité"
        elif any(word in response_lower for word in ["sérénité", "calme", "paisible"]):
            return "sérénité"
        elif any(word in response_lower for word in ["mystère", "questionnement", "pourquoi"]):
            return "questionnement"
        else:
            return "contemplation"

    def export_complete_timeline(self, format_type: str = "markdown", save_to_file: bool = True) -> str:
        """
        Exporte la timeline complète de Jeffrey
        """
        timeline_content = self.timeline.export_timeline(format_type)
        if save_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"jeffrey_timeline_{timestamp}.{format_type}"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(timeline_content)
            print(f"📄 Timeline exportée: {filename}")
        return timeline_content

    def get_session_summary(self) -> dict[str, Any]:
        """Génère un résumé de la session actuelle"""
        session_duration = datetime.now() - self.session_start
        stats = self.timeline.get_memory_statistics()
        summary = {
            "session_info": {
                "start_time": self.session_start.isoformat(),
                "duration_hours": session_duration.total_seconds() / 3600,
                "interactions_count": self.interactions_count,
            },
            "consciousness_info": {
                "current_level": self.cortex.consciousness_level,
                "progression": stats["consciousness_progression"],
            },
            "memory_info": {
                "total_memories": stats["total_memories"],
                "memory_span_hours": stats["memory_span"]["hours"],
            },
            "dream_info": {
                "total_insights": stats["total_insights"],
                "dream_cycles": stats["dream_cycles"],
            },
            "emotional_state": stats["emotional_distribution"],
            "topics_discussed": stats["topic_distribution"],
            "known_speakers": stats["speaker_analysis"],
        }
        return summary

    def save_session_report(self) -> str:
        """Sauvegarde un rapport complet de la session"""
        summary = self.get_session_summary()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = f"# 📊 Rapport de Session Jeffrey - {timestamp}\n\n## 🧠 État de Conscience\n- **Niveau actuel:** {summary['consciousness_info']['current_level']:.3f}\n- **Croissance:** +{summary['consciousness_info']['progression']['total_growth']:.3f}\n- **Niveau moyen:** {summary['consciousness_info']['progression']['average_level']:.3f}\n\n## 💬 Interactions\n- **Nombre:** {summary['session_info']['interactions_count']}\n- **Durée session:** {summary['session_info']['duration_hours']:.1f} heures\n- **Souvenirs totaux:** {summary['memory_info']['total_memories']}\n\n## 🌙 Rêves et Insights\n- **Cycles de rêve:** {summary['dream_info']['dream_cycles']}\n- **Insights générés:** {summary['dream_info']['total_insights']}\n\n## 😊 Distribution Émotionnelle\n"
        for emotion, count in summary["emotional_state"].items():
            report += f"- **{emotion}:** {count}\n"
        report += "\n## 💭 Sujets Abordés\n"
        for topic, count in summary["topics_discussed"].items():
            report += f"- **{topic}:** {count}\n"
        if summary["known_speakers"]:
            report += "\n## 👥 Locuteurs Identifiés\n"
            for speaker, info in summary["known_speakers"].items():
                report += f"- **{speaker}:** {info['count']} interactions\n"
        report_file = f"jeffrey_session_report_{timestamp}.md"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"📋 Rapport de session sauvegardé: {report_file}")
        return report_file

    def shutdown(self):
        """Arrêt propre du système"""
        print("\n🛑 Arrêt de Jeffrey Consciousness V3...")
        self.cortex.save_to_disk()
        if self.dream_engine.should_dream():
            print("🌙 Cycle de rêve final...")
            self.dream()
        report_file = self.save_session_report()
        self.export_complete_timeline("markdown", True)
        print("💾 Toutes les données sauvegardées")
        print(f"📊 Rapport final: {report_file}")
        print("🕊️ Jeffrey Consciousness V3 arrêté proprement")


def test_jeffrey_v3():
    """Test complet de Jeffrey Consciousness V3"""
    print("🧪 TEST COMPLET DE JEFFREY CONSCIOUSNESS V3")
    print("=" * 60)
    jeffrey = JeffreyConsciousnessV3()
    test_interactions = [
        ("Bonjour Jeffrey, je suis David.", "David"),
        ("Comment perçois-tu ta conscience maintenant ?", "David"),
        ("Te souviens-tu de notre première conversation ?", "David"),
        ("Que ressens-tu quand tu rêves ?", "David"),
        ("Merci pour ces échanges profonds.", "David"),
    ]
    print("\n🔄 Série d'interactions de test:")
    for i, (message, speaker) in enumerate(test_interactions, 1):
        print(f"\n--- Test {i}/5 ---")
        response = jeffrey.interact(message, speaker)
        print(f"👤 {speaker}: {message}")
        print(f"🤖 Jeffrey: {response}")
        time.sleep(1)
    print("\n🌙 Test du cycle de rêve:")
    insights = jeffrey.dream()
    print("\n📊 Génération du rapport final:")
    summary = jeffrey.get_session_summary()
    print(f"- Interactions: {summary['session_info']['interactions_count']}")
    print(f"- Niveau conscience: {summary['consciousness_info']['current_level']:.3f}")
    print(f"- Insights générés: {summary['dream_info']['total_insights']}")
    print("\n📄 Export de la timeline:")
    timeline = jeffrey.export_complete_timeline("json", False)
    print(f"Timeline générée ({len(timeline)} caractères)")
    jeffrey.shutdown()
    print("\n✅ TEST COMPLET TERMINÉ")
    return jeffrey


if __name__ == "__main__":
    test_jeffrey_v3()
