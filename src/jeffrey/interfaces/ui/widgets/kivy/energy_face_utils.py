#!/usr/bin/env python
"""
energy_face_utils.py - Utilitaires pour le visage de Jeffrey
Partie de la refactorisation du fichier energy_face.py d'origine (PACK 18)

Ce module contient les fonctions utilitaires pour le visage :
- Fonctions d'animation génériques
- Gestion des états d'inactivité
- Fonctions de mapping des coordonnées
- Utilitaires pour les conversions de formats
"""

import logging
import math
import random
import time
from collections.abc import Callable
from typing import Any

from kivy.clock import Clock


class UtilityFunctions:
    """
    Classe d'utilitaires pour le visage de Jeffrey.
    Fournit des fonctions communes utilisées par les autres modules.
    """

    def __init__(self, face_widget):
        """
        Initialise les fonctions utilitaires.

        Args:
            face_widget: Widget du visage (EnergyFaceCoreWidget)
        """
        self.face = face_widget

        # Variables pour le suivi de l'activité
        self.last_activity_time = time.time()
        self.idle_duration = 0
        self.is_idle = False
        self.idle_phase = 0.0
        self.idle_animation_active = False

        # Initialiser les timers et états
        self.timers = {}

        # Planifier la vérification d'inactivité
        Clock.schedule_interval(self.check_inactivity, 5.0)
        Clock.schedule_interval(self.animate_idle_state, 1 / 15.0)

    def check_inactivity(self, dt):
        """
        Vérifie si l'interaction est inactive depuis longtemps.

        Args:
            dt: Delta temps depuis la dernière vérification
        """
        # Calculer le temps d'inactivité
        current_time = time.time()
        inactivity_time = current_time - self.last_activity_time

        # Après 30 secondes d'inactivité, entrer en mode présence silencieuse
        if inactivity_time > 30.0 and not self.is_idle:
            self.is_idle = True
            self.activate_idle_mode()
        # Si l'activité reprend, sortir du mode inactivité
        elif inactivity_time < 5.0 and self.is_idle:
            self.is_idle = False
            self.deactivate_idle_mode()

    def animate_idle_state(self, dt):
        """
        Gère l'animation des états d'inactivité.

        Args:
            dt: Delta temps depuis la dernière mise à jour
        """
        # Si l'animation d'inactivité est active
        if self.idle_animation_active:
            # Mise à jour de la phase d'animation
            self.idle_phase += dt * 0.3  # Fréquence lente pour une animation douce

            # Subtile oscillation de l'ouverture des paupières
            paupiere_variation = math.sin(self.idle_phase) * 0.1
            self.face.eyelid_openness = 0.8 + paupiere_variation
        else:
            # Vérifier si l'inactivité dure depuis longtemps
            current_time = time.time()
            if current_time - self.last_activity_time > 60.0:  # Plus d'une minute
                # Augmenter lentement la fatigue
                if hasattr(self.face, "fatigue_level"):
                    self.face.fatigue_level = min(1.0, getattr(self.face, "fatigue_level", 0) + dt * 0.005)
            else:
                # Réduire progressivement la fatigue
                if hasattr(self.face, "fatigue_level"):
                    self.face.fatigue_level = max(0.0, getattr(self.face, "fatigue_level", 0) - dt * 0.01)

    def activate_idle_mode(self):
        """Active le mode d'inactivité avec effets visuels subtils."""
        self.idle_animation_active = True
        self.idle_phase = 0.0

        # Appliquer un effet de présence subtile
        if hasattr(self.face.effects, "pulse_light"):
            self.face.effects.pulse_light(intensity=0.3)

    def deactivate_idle_mode(self):
        """Désactive le mode d'inactivité."""
        self.idle_animation_active = False

        # Restaurer l'état normal
        self.face.eyelid_openness = 1.0

    def register_activity(self):
        """Enregistre une activité utilisateur, réinitialisant le timer d'inactivité."""
        self.last_activity_time = time.time()

        # Si en mode inactif, désactiver
        if self.is_idle:
            self.is_idle = False
            self.deactivate_idle_mode()

    def schedule_once(self, callback: Callable, delay: float, timer_id: str = None):
        """
        Programme l'exécution d'une fonction après un délai.

        Args:
            callback: Fonction à exécuter
            delay: Délai en secondes
            timer_id: Identifiant du timer (optionnel)
        """
        # Générer un ID aléatoire si non spécifié
        if timer_id is None:
            timer_id = f"timer_{random.randint(1000, 9999)}"

        # Annuler le timer existant si présent
        if timer_id in self.timers:
            Clock.unschedule(self.timers[timer_id])

        # Fonction wrapper pour le nettoyage
        def wrapped_callback(dt):
            # Exécuter le callback
            callback(dt)
            # Supprimer la référence au timer
            if timer_id in self.timers:
                del self.timers[timer_id]

        # Créer et enregistrer le timer
        timer = Clock.schedule_once(wrapped_callback, delay)
        self.timers[timer_id] = timer

        return timer_id

    def cancel_timer(self, timer_id: str):
        """
        Annule un timer programmé.

        Args:
            timer_id: Identifiant du timer à annuler
        """
        if timer_id in self.timers:
            Clock.unschedule(self.timers[timer_id])
            del self.timers[timer_id]

    def get_coordinates_for_zone(self, zone: str) -> tuple[float, float, float]:
        """
        Récupère les coordonnées d'une zone du visage.

        Args:
            zone: Nom de la zone

        Returns:
            Tuple (x, y, rayon) des coordonnées et taille de la zone
        """
        # Coordonnées du centre du visage
        center_x = self.face.center_x
        center_y = self.face.center_y

        # Mapping des zones avec leurs coordonnées
        zone_coordinates = {
            "joue_gauche": (center_x - 40, center_y - 10, 20),
            "joue_droite": (center_x + 40, center_y - 10, 20),
            "front": (center_x, center_y + 40, 25),
            "menton": (center_x, center_y - 45, 15),
            "nez": (center_x, center_y + 5, 12),
            "lèvres": (center_x, center_y - 25, 15),
            "visage": (center_x, center_y, 60),
            "tête": (center_x, center_y + 20, 70),
            "œil_gauche": (center_x - 40, center_y + 20, 15),
            "œil_droit": (center_x + 40, center_y + 20, 15),
        }

        return zone_coordinates.get(zone, (center_x, center_y, 30))

    def is_inside_zone(self, x: float, y: float, zone: str) -> bool:
        """
        Vérifie si un point est à l'intérieur d'une zone du visage.

        Args:
            x: Coordonnée X
            y: Coordonnée Y
            zone: Nom de la zone à vérifier

        Returns:
            True si le point est dans la zone, False sinon
        """
        # Récupérer les coordonnées de la zone
        zone_x, zone_y, zone_radius = self.get_coordinates_for_zone(zone)

        # Calculer la distance au centre de la zone
        distance = math.sqrt((x - zone_x) ** 2 + (y - zone_y) ** 2)

        # Le point est dans la zone si la distance est inférieure au rayon
        return distance <= zone_radius

    def get_emotion_color(self, emotion: str, alpha: float = 0.5) -> tuple[float, float, float, float]:
        """
        Retourne une couleur RGBA correspondant à une émotion.

        Args:
            emotion: Nom de l'émotion
            alpha: Opacité de la couleur (0.0 à 1.0)

        Returns:
            Tuple RGBA (r, g, b, a)
        """
        # Palette de couleurs par émotion
        emotion_colors = {
            "joie": (1.0, 0.9, 0.4, alpha),
            "tristesse": (0.4, 0.6, 0.9, alpha),
            "colère": (0.9, 0.3, 0.3, alpha),
            "peur": (0.6, 0.4, 0.8, alpha),
            "surprise": (0.8, 0.8, 0.4, alpha),
            "dégoût": (0.5, 0.8, 0.5, alpha),
            "neutre": (0.7, 0.7, 0.7, alpha * 0.8),
            "émerveillement": (0.9, 0.6, 1.0, alpha),
            "sérénité": (0.6, 0.9, 1.0, alpha),
            "stress": (0.8, 0.5, 0.3, alpha),
            "curiosité": (0.6, 0.8, 0.9, alpha),
            "affection": (1.0, 0.6, 0.8, alpha),
            "amour": (0.9, 0.5, 0.7, alpha * 1.2),
        }

        # Couleur par défaut si émotion inconnue
        return emotion_colors.get(emotion.lower(), (0.8, 0.8, 0.8, alpha * 0.8))

    def format_duration(self, seconds: float) -> str:
        """
        Formate une durée en secondes en texte lisible.

        Args:
            seconds: Durée en secondes

        Returns:
            Chaîne formatée
        """
        # Arrondir à l'entier le plus proche
        seconds = int(round(seconds))

        # Calculer minutes et heures
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        days, hours = divmod(hours, 24)

        # Générer le texte selon les unités
        if days > 0:
            return f"{days}j {hours}h"
        elif hours > 0:
            return f"{hours}h {minutes}min"
        elif minutes > 0:
            return f"{minutes}min {seconds}s"
        else:
            return f"{seconds}s"

    def interpolate(self, start: float, end: float, progress: float) -> float:
        """
        Effectue une interpolation linéaire entre deux valeurs.

        Args:
            start: Valeur de départ
            end: Valeur d'arrivée
            progress: Progression (0.0 à 1.0)

        Returns:
            Valeur interpolée
        """
        # Limiter progress entre 0 et 1
        progress = max(0.0, min(1.0, progress))

        # Interpolation linéaire
        return start + (end - start) * progress

    def ease_in_out(self, progress: float) -> float:
        """
        Fonction d'accélération/décélération pour les animations.

        Args:
            progress: Progression linéaire (0.0 à 1.0)

        Returns:
            Progression avec effet d'accélération/décélération
        """
        # Limiter progress entre 0 et 1
        progress = max(0.0, min(1.0, progress))

        # Courbe sinusoïdale pour un effet doux
        return 0.5 - 0.5 * math.cos(math.pi * progress)

    def create_property_animation(
        self,
        property_name: str,
        start_value: Any,
        end_value: Any,
        duration: float,
        easing: bool = True,
    ) -> None:
        """
        Crée une animation de propriété fluide.

        Args:
            property_name: Nom de la propriété à animer
            start_value: Valeur de départ
            end_value: Valeur d'arrivée
            duration: Durée de l'animation en secondes
            easing: Si True, utilise une fonction d'accélération/décélération
        """
        # Vérifier que la propriété existe
        if not hasattr(self.face, property_name):
            logging.warning(f"Propriété {property_name} non trouvée")
            return

        # Temps de départ de l'animation
        start_time = time.time()

        # Fonction d'animation
        def animate_property(dt):
            # Calculer la progression
            current_time = time.time()
            elapsed = current_time - start_time

            # Progression normalisée (0 à 1)
            progress = min(1.0, elapsed / duration)

            # Appliquer l'easing si demandé
            if easing:
                progress = self.ease_in_out(progress)

            # Calculer la valeur intermédiaire
            try:
                # Pour les nombres
                if isinstance(start_value, (int, float)) and isinstance(end_value, (int, float)):
                    current_value = self.interpolate(start_value, end_value, progress)
                # Pour les tuples/listes de nombres (ex: couleurs)
                elif (
                    isinstance(start_value, (list, tuple))
                    and isinstance(end_value, (list, tuple))
                    and len(start_value) == len(end_value)
                ):
                    current_value = tuple(
                        self.interpolate(start, end, progress) for start, end in zip(start_value, end_value)
                    )
                else:
                    # Pour les autres types, pas d'interpolation
                    current_value = end_value if progress > 0.5 else start_value
            except Exception:
                # En cas d'erreur, utiliser directement la valeur finale
                current_value = end_value

            # Appliquer la valeur
            setattr(self.face, property_name, current_value)

            # Continuer l'animation tant que la durée n'est pas écoulée
            return progress < 1.0

        # Démarrer l'animation
        Clock.schedule_interval(animate_property, 1 / 30.0)
