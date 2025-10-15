#!/usr/bin/env python
"""
Contrôleur du visage émotionnel de Jeffrey - Version refactorisée (PACK 18)
Ce fichier a été restructuré pour améliorer la maintenabilité et la lisibilité.
"""

import logging

# Import des modules refactorisés
# Import des sous-modules refactorisés
from widgets.energy_face_core import EnergyFaceCoreWidget


class EnergyFaceWidget(EnergyFaceCoreWidget):
    """
    Contrôle l'affichage et l'animation du visage émotionnel de Jeffrey.

    Cette classe a été refactorisée pour déléguer les fonctionnalités spécifiques
    à des modules spécialisés tout en gardant la compatibilité avec le code existant.
    """

    def __init__(self, **kwargs):
        """
        Initialise le widget du visage en chargeant les sous-systèmes spécialisés.
        """
        # Initialisation du widget de base
        super(EnergyFaceWidget, self).__init__(**kwargs)

        # Journal d'événements
        self.logger = logging.getLogger("jeffrey.face")
        self.logger.info("Initialisation du visage émotionnel de Jeffrey")

    def on_emotion_change(self, instance, value):
        """
        Override de la méthode de base pour garantir la compatibilité.

        Args:
            instance: Instance qui a changé
            value: Nouvelle valeur de l'émotion
        """
        # Appeler d'abord la méthode parent
        super(EnergyFaceWidget, self).on_emotion_change(instance, value)

        # Journaliser le changement d'émotion
        self.logger.info(f"Changement d'émotion: {value} (intensité: {self.intensity})")

    def set_emotion(self, emotion_name, intensity=0.5, secondary_emotion=None):
        """
        Définit l'émotion du visage.

        Args:
            emotion_name: Nom de l'émotion
            intensity: Intensité de l'émotion (0.0 à 1.0)
            secondary_emotion: Émotion secondaire optionnelle
        """
        # Mettre à jour les propriétés
        self.emotion = emotion_name
        self.intensity = intensity

        # Mettre à jour l'émotion secondaire si fournie
        if secondary_emotion:
            self.emotion_secondary = secondary_emotion

        # Utiliser directement le visual_renderer pour garantir que tous les effets sont appliqués
        self.visual_renderer.render_emotion(
            emotion=emotion_name,
            intensity=intensity,
            secondary_emotion=self.emotion_secondary,
            blend=self.emotion_blend,
        )

    def clear_emotion(self):
        """Efface l'émotion actuelle et revient à l'état neutre."""
        self.set_emotion("neutre", 0.5)
        self.emotion_secondary = None
        self.emotion_blend = 0.0

        # Désactiver les effets actifs
        for effect_name in list(self.active_effects.keys()):
            # Obtenir la méthode d'arrêt
            stop_method_name = f"stop_{effect_name}"
            if hasattr(self.effects, stop_method_name):
                stop_method = getattr(self.effects, stop_method_name)
                stop_method()

    def process_touch(self, x, y, touch_type="caresse", intensity=0.5):
        """
        Traite un toucher à une position spécifique.

        Args:
            x: Coordonnée X du toucher
            y: Coordonnée Y du toucher
            touch_type: Type de toucher (caresse, tapotement, etc.)
            intensity: Intensité du toucher (0.0 à 1.0)
        """
        # Déterminer la zone touchée en fonction des coordonnées
        zone = self._identify_touch_zone(x, y)

        # PACK 20: Déclencher les réactions visuelles appropriées
        if self.effects:
            # Catégoriser le type de toucher
            affectueux = ["caresse", "bisou", "effleurement", "massage"]
            neutre = ["toucher", "frôlement"]
            intense = ["appui", "tape", "pincement", "grattement"]

            # Déclencher l'effet approprié
            if touch_type in affectueux:
                self.effects.react_to_touch_affectueux(zone, intensity)
            elif touch_type in neutre:
                self.effects.react_to_touch_neutre(zone, intensity)
            elif touch_type in intense:
                self.effects.react_to_touch_intense(zone, intensity, touch_type)

            # Zones sensibles
            sensibles = ["lèvres", "joue_gauche", "joue_droite", "nez"]
            if zone in sensibles:
                self.effects.react_to_touch_sensible(zone, intensity)

        # Déléguer au gestionnaire de mémoire
        self.memory_handler.process_touch(x, y, touch_type, intensity)

        # Enregistrer l'activité
        self.utils.register_activity()

        # Marquer comme récemment touché pour l'effet visuel de base
        self.touched_recently = True

        # Programmer la fin de l'effet de toucher de base
        def end_touch_effect(dt):
            self.touched_recently = False

        self.utils.schedule_once(end_touch_effect, 0.3, "touch_effect")

    def _identify_touch_zone(self, x: float, y: float) -> str:
        """
        PACK 20: Identifie la zone du visage touchée en fonction des coordonnées.

        Args:
            x: Coordonnée X du toucher
            y: Coordonnée Y du toucher

        Returns:
            Nom de la zone touchée
        """
        # Calculer les distances au centre
        center_x = self.center_x
        center_y = self.center_y
        dx = x - center_x
        dy = y - center_y
        distance = math.sqrt(dx * dx + dy * dy)

        # Angles (en radians, origine à droite, sens anti-horaire)
        angle = math.atan2(dy, dx)

        # Si toucher près du centre du visage
        if distance < 30:
            return "nez"

        # Si toucher dans le cercle du visage
        if distance < 70:
            # Joue gauche
            if -3 * math.pi / 4 < angle < -math.pi / 4:
                return "joue_gauche"
            # Joue droite
            elif -math.pi / 4 < angle < math.pi / 4:
                return "joue_droite"
            # Front
            elif math.pi / 4 < angle < 3 * math.pi / 4:
                return "front"
            # Menton/bouche
            else:
                if distance < 55:
                    return "lèvres"
                else:
                    return "menton"

        # Au-delà du cercle du visage
        return "tête"

    def on_touch_down(self, touch):
        """
        Gère les interactions tactiles avec le visage.

        Args:
            touch: Objet Touch contenant les informations du toucher
        """
        # Vérifier si le toucher est dans les limites du widget
        if self.collide_point(*touch.pos):
            # Enregistrer l'activité
            self.utils.register_activity()

            # Traiter le toucher au niveau sensoriel
            self.process_touch(touch.pos[0], touch.pos[1])

            return True

        return super(EnergyFaceWidget, self).on_touch_down(touch)

    def trigger_emotional_response(self, trigger_type, intensity=0.7):
        """
        Déclenche une réponse émotionnelle à un événement externe.

        Args:
            trigger_type: Type de déclencheur émotionnel
            intensity: Intensité de la réponse (0.0 à 1.0)
        """
        # Déléguer au gestionnaire d'émotions
        self.emotion_handler.trigger_emotional_response(trigger_type, intensity)

    def trigger_immersive_scene(self, trigger_type, intensity=1.0):
        """
        Déclenche une scène immersive avec effets spéciaux.

        Args:
            trigger_type: Type de déclencheur ("attention", "pause", "réflexion", etc.)
            intensity: Intensité de l'effet (0.0 à 1.0)
        """
        # Utiliser le renderer visuel pour la scène immersive
        self.visual_renderer.trigger_immersive_scene(trigger_type, intensity)

    def animate_mouth_speak(self, text, duration=None):
        """
        Anime la bouche pour simuler la parole d'un texte.

        Args:
            text: Texte à prononcer
            duration: Durée totale de la parole (si None, calculé à partir du texte)
        """
        # Déléguer au gestionnaire de mouvements
        self.movement_handler.animate_mouth_speak(text, duration)

    # Propriétés pour compatibilité avec l'ancien code
    @property
    def active_effects(self):
        """Liste des effets actifs (pour compatibilité)."""
        return getattr(self.effects, "_active_effects", set())
