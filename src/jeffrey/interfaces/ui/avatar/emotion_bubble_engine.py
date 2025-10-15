import random
from datetime import datetime, timedelta

from jeffrey.core.emotions.emotional_journal import EmotionalJournal


class EmotionBubble:
    def __init__(self, text, emotion, duration=6.0, discret=False):
        self.text = text
        self.emotion = emotion
        self.created_at = datetime.now()
        self.duration = timedelta(seconds=duration)
        self.opacity = 0.5 if discret else 1.0
        self.position = [
            random.uniform(0.1, 0.9),
            random.uniform(0.1, 0.9),
        ]  # Position X/Y en pourcentage d'écran
        self.velocity = [
            random.uniform(-0.002, 0.002),
            random.uniform(0.001, 0.003),
        ]  # Mouvement léger
        self.discret = discret

    def is_expired(self):
        return datetime.now() - self.created_at > self.duration

    def update(self, dt):
        self.position[0] += self.velocity[0] * dt
        self.position[1] += self.velocity[1] * dt
        # Diminue l'opacité doucement sur la fin si pas en mode discret
        elapsed = (datetime.now() - self.created_at).total_seconds()
        if not self.discret and elapsed > self.duration.total_seconds() * 0.7:
            self.opacity = max(
                0.0,
                1.0 - (elapsed - self.duration.total_seconds() * 0.7) / (self.duration.total_seconds() * 0.3),
            )


class EmotionBubbleEngine:
    def __init__(self):
        self.bubbles = []
        self.color_map = {
            "joie": (1.0, 1.0, 0.6),  # Jaune clair
            "tristesse": (0.6, 0.8, 1.0),  # Bleu doux
            "colère": (1.0, 0.6, 0.6),  # Rouge pastel
            "peur": (0.7, 0.7, 1.0),  # Violet doux
            "amour": (1.0, 0.8, 1.0),  # Rose clair
            "calme": (0.8, 1.0, 0.8),  # Vert léger
            "neutre": (0.9, 0.9, 0.9),  # Gris perlé
            "rêverie": (0.8, 0.8, 1.0),  # Bleu pastel
            "inspiration": (1.0, 0.9, 0.7),  # Pêche clair
        }
        self.journal = EmotionalJournal()
        self.mode_discret = False

    def activer_mode_discret(self, actif: bool):
        self.mode_discret = actif

    def ajouter_bulle(self, text, emotion="neutre"):
        if len(self.bubbles) >= 3:
            # Ne garder que 3 bulles maximum
            self.bubbles.pop(0)

        # Activer automatiquement le mode discret pour certaines émotions
        emotions_discretes = {"calme", "tristesse", "rêverie", "peur"}
        is_discret = self.mode_discret or (emotion in emotions_discretes)

        self.bubbles.append(EmotionBubble(text, emotion, duration=6.0 if is_discret else 10.0, discret=is_discret))
        self.journal.enregistrer_emotion(text, emotion)

    def update_bubbles(self, dt):
        for bubble in self.bubbles:
            bubble.update(dt)
        # Nettoyer les bulles expirées
        self.bubbles = [b for b in self.bubbles if not b.is_expired()]

    def draw_bubbles(self, canvas):
        for bubble in self.bubbles:
            x, y = bubble.position
            color = self.color_map.get(bubble.emotion, (1.0, 1.0, 1.0))
            alpha = bubble.opacity
            # Ici, `canvas` doit pouvoir dessiner un texte semi-transparent à la position donnée
            canvas.draw_text(bubble.text, pos=(x, y), color=(color[0], color[1], color[2], alpha), font_size=18)

    def get_bulles_actuelles(self):
        """
        Retourne une liste des bulles actuellement actives sous forme de dictionnaire.
        Utile pour la mémoire ou l'affichage dans un journal émotionnel.
        """
        return [
            {
                "texte": bubble.text,
                "emotion": bubble.emotion,
                "discret": bubble.discret,
                "age": (datetime.now() - bubble.created_at).total_seconds(),
                "temps_restant": max(
                    0.0,
                    bubble.duration.total_seconds() - (datetime.now() - bubble.created_at).total_seconds(),
                ),
            }
            for bubble in self.bubbles
        ]
