import logging
import numpy as np
from typing import List, Dict
from datetime import datetime, timedelta
from jeffrey.memory.hybrid_store import HybridMemoryStore
from jeffrey.ml.embeddings_service import embeddings_service
import os

logger = logging.getLogger(__name__)

class ProactiveCuriositySafe:
    """Version sécurisée de ProactiveCuriosity - read-only par défaut"""

    def __init__(self):
        self.memory_store = HybridMemoryStore()
        self.write_enabled = os.getenv("ENABLE_CONSCIOUSNESS_WRITE", "false").lower() == "true"
        self.max_questions = int(os.getenv("CONSCIOUSNESS_MAX_QUESTIONS", "3"))
        self.enabled = os.getenv("ENABLE_CONSCIOUSNESS", "false").lower() == "true"

    async def analyze_gaps(self) -> Dict:
        """Analyse les patterns sans écrire"""
        if not self.enabled:
            return {"status": "disabled", "reason": "ENABLE_CONSCIOUSNESS=false"}

        try:
            # Récupérer mémoires récentes
            since = datetime.utcnow() - timedelta(hours=24)
            recent = await self.memory_store.get_recent(since, limit=30)

            if len(recent) < 5:
                return {"status": "insufficient_data", "memories_analyzed": len(recent)}

            # Analyse simple des émotions
            emotion_counts = {}
            for mem in recent:
                emotion = mem.get('emotion', 'neutral')
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

            # Identifier le pattern dominant
            dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else 'neutral'

            # Analyse temporelle
            time_distribution = self._analyze_time_patterns(recent)

            return {
                "status": "analyzed",
                "memories_analyzed": len(recent),
                "emotion_distribution": emotion_counts,
                "dominant_emotion": dominant_emotion,
                "time_patterns": time_distribution,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"Gap analysis failed: {e}")
            return {"status": "error", "error": str(e)}

    def _analyze_time_patterns(self, memories: List[Dict]) -> Dict:
        """Analyse temporelle des mémoires"""
        try:
            hours = []
            for mem in memories:
                timestamp = mem.get('timestamp')
                if isinstance(timestamp, str):
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    hours.append(dt.hour)
                elif hasattr(timestamp, 'hour'):
                    hours.append(timestamp.hour)

            if not hours:
                return {"pattern": "no_data"}

            # Identifier les créneaux d'activité
            hour_counts = {}
            for hour in hours:
                hour_counts[hour] = hour_counts.get(hour, 0) + 1

            peak_hour = max(hour_counts.items(), key=lambda x: x[1])[0]

            return {
                "pattern": "analyzed",
                "peak_hour": peak_hour,
                "total_hours_active": len(set(hours)),
                "distribution": hour_counts
            }
        except Exception as e:
            logger.warning(f"Time pattern analysis failed: {e}")
            return {"pattern": "error", "error": str(e)}

    async def generate_questions(self) -> List[str]:
        """Génère des questions sans les stocker"""
        if not self.enabled:
            return []

        analysis = await self.analyze_gaps()
        questions = []

        if analysis.get("status") != "analyzed":
            return questions

        dominant = analysis.get("dominant_emotion", "neutral")
        memories_count = analysis.get("memories_analyzed", 0)

        # Questions basées sur l'émotion dominante
        if dominant == "curiosity":
            questions.append("What patterns connect these curious thoughts?")
            questions.append("What new areas of knowledge are most intriguing?")
        elif dominant == "joy":
            questions.append("What elements contribute most to positive experiences?")
            questions.append("How can these joyful patterns be expanded?")
        elif dominant == "surprise":
            questions.append("What unexpected connections emerged recently?")
        elif dominant == "neutral":
            questions.append("What new perspectives could enrich understanding?")
        else:
            questions.append(f"How does the pattern of {dominant} emotions relate to learning?")

        # Questions basées sur l'activité
        if memories_count > 20:
            questions.append("What themes emerge from this period of high activity?")
        elif memories_count < 10:
            questions.append("What topics deserve deeper exploration?")

        # Limiter le nombre de questions
        return questions[:self.max_questions]

    async def get_status(self) -> Dict:
        """Statut complet du système de curiosité"""
        return {
            "enabled": self.enabled,
            "write_enabled": self.write_enabled,
            "max_questions": self.max_questions,
            "embeddings_available": embeddings_service.enabled,
            "last_check": datetime.utcnow().isoformat()
        }