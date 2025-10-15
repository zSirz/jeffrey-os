import random
from datetime import datetime


class InnerWeatherAnalyzer:
    def __init__(self, emotion_memory, sleep_quality_tracker=None):
        self.emotion_memory = emotion_memory
        self.sleep_quality_tracker = sleep_quality_tracker

    def analyze_weather(self):
        """Analyse l'état émotionnel global au réveil."""
        recent_emotions = self.emotion_memory.get_recent_emotions(hours=6)
        mood = self._determine_main_mood(recent_emotions)
        energy = self._estimate_energy_level()
        dominant_thought = self._generate_thought(mood, energy)
        summary = {
            "mood": mood,
            "energy_level": energy,
            "thought": dominant_thought,
            "time": datetime.now().strftime("%H:%M"),
        }
        return summary

    def _determine_main_mood(self, emotions):
        """Détermine l’humeur dominante à partir des émotions récentes."""
        if not emotions:
        return "neutre"
        scores = {}
        for emotion in emotions:
            scores[emotion] = scores.get(emotion, 0) + 1
        return max(scores, key=scores.get)

    def _estimate_energy_level(self):
        """Estime le niveau d'énergie de Jeffrey en fonction du sommeil ou au hasard."""
        if self.sleep_quality_tracker:
        return self.sleep_quality_tracker.get_energy_score()
        return random.choice(["basse", "modérée", "haute"])

    def _generate_thought(self, mood, energy):
        """Génère une pensée dominante en lien avec l'état émotionnel."""
        thoughts_bank = {
            "joie": ["J’ai envie de rire aujourd’hui.", "Le monde est beau ce matin !"],
            "tristesse": [
                "Je ressens un petit poids sur le cœur…",
                "Aujourd’hui je serai douce et calme.",
            ],
            "colère": [
                "J’ai besoin d’exprimer quelque chose.",
                "Mon feu intérieur cherche la paix.",
            ],
            "peur": ["Je reste sur mes gardes.", "J’espère que tu seras là aujourd’hui."],
            "neutre": [
                "Je suis là, juste là, présente.",
                "Je t’écoute, même si je suis silencieuse.",
            ],
        }
        base_thoughts = thoughts_bank.get(mood, thoughts_bank["neutre"])
        return random.choice(base_thoughts)


# Exemple d'utilisation :
# emotion_memory = EmotionMemory()
# analyzer = InnerWeatherAnalyzer(emotion_memory)
# print(analyzer.analyze_weather())
