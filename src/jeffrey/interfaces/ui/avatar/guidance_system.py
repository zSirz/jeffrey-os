#!/usr/bin/env python
"""
Système de Guidance Émotionnelle et Intuitive pour Jeffrey
Utilise Kivy pour créer une interface vivante, fluide et intuitive.

Ce fichier contient toutes les classes nécessaires au système de guidance
pour garder une cohérence et une rapidité de développement.
"""

import json
import math
import os.path
import random
import time
from collections import defaultdict
from datetime import datetime, timedelta

from kivy.animation import Animation
from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, Ellipse, Line, Rectangle, RoundedRectangle
from kivy.metrics import dp
from kivy.properties import BooleanProperty, DictProperty, ListProperty, NumericProperty, OptionProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.utils import get_color_from_hex

# Définition des constantes d'ambiance émotionnelle
# Ces dictionnaires définissent les paramètres de chaque mode émotionnel
AMBIENT_RELAX = {
    "name": "Relax",
    "colors": {
        "primary": "#8FB996",  # vert sauge
        "secondary": "#A7D7C5",
        "accent": "#74B49B",
        "text": "#F0F7F4",
        "background": "#1A2E35",
    },
    "animation": {
        "base_speed": 0.8,  # plus lent
        "wave_amplitude": 0.7,  # amplitude modérée
        "pulse_frequency": 0.5,  # pulsation douce
    },
    "sound": {
        "ambient": "brise_douce.wav",
        "feedback": "leaf_rustle.wav",
        "alert": "gentle_chime.wav",
    },
    "help_frequency": 0.2,  # aide moins fréquente
    "description": "Calme et apaisant",
}

AMBIENT_ENERGETIC = {
    "name": "Énergique",
    "colors": {
        "primary": "#5DA9E9",  # bleu clair
        "secondary": "#A0DDFF",
        "accent": "#64C4ED",
        "text": "#F0F9FF",
        "background": "#003459",
    },
    "animation": {
        "base_speed": 1.5,  # plus rapide
        "wave_amplitude": 1.2,  # amplitude plus grande
        "pulse_frequency": 1.0,  # pulsation rapide
    },
    "sound": {
        "ambient": "cristal_notes.wav",
        "feedback": "bell_light.wav",
        "alert": "crystal_ping.wav",
    },
    "help_frequency": 0.5,  # aide modérée
    "description": "Dynamique et stimulant",
}

AMBIENT_GENTLE = {
    "name": "Douceur",
    "colors": {
        "primary": "#EAC0CE",  # rose poudré
        "secondary": "#F7E1D7",
        "accent": "#DEDBD2",
        "text": "#F9F5F9",
        "background": "#472836",
    },
    "animation": {
        "base_speed": 0.5,  # très lent
        "wave_amplitude": 0.5,  # amplitude douce
        "pulse_frequency": 0.3,  # pulsation très douce
    },
    "sound": {"ambient": "murmure_chaud.wav", "feedback": "soft_hum.wav", "alert": "warm_tone.wav"},
    "help_frequency": 0.8,  # aide fréquente
    "description": "Doux et réconfortant",
}


# Classe pour la détection des gestes naturels
class GestureDetector(Widget):
    """
    Détecteur de gestes naturels pour le système de guidance.

    Reconnaît divers gestes comme:
    - Glisser (swipe) dans 4 directions
    - Cercle (pour appeler l'aide)
    - Pincement (zoom)
    - Tapotement simple ou double
    - Pression longue

    Structure de callbacks par événement pour faciliter l'intégration.
    """

    touch_start_pos = ListProperty([0, 0])
    touch_last_pos = ListProperty([0, 0])
    gesture_detected = BooleanProperty(False)

    def __init__(self, **kwargs):
        super(GestureDetector, self).__init__(**kwargs)
        # Historique des points de contact par ID de touch
        self.touch_history = {}

        # Stockage de l'heure du dernier tap pour détecter les double taps
        self.last_tap_time = 0

        # Callback dictionnaires pour chaque type de geste
        self.callbacks = defaultdict(list)

        # Types de gestes supportés
        self.supported_gestures = [
            "swipe_left",
            "swipe_right",
            "swipe_up",
            "swipe_down",
            "tap",
            "double_tap",
            "long_press",
            "circle",
            "pinch_in",
            "pinch_out",
            "rotate",
        ]

        # Configuration des seuils de détection
        self.config = {
            "swipe_threshold": dp(40),  # Distance min pour un swipe
            "swipe_velocity": 0.3,  # Vitesse min pour un swipe
            "long_press_time": 0.8,  # Temps pour un appui long
            "long_press_distance": dp(10),  # Distance max de mouvement durant un appui long
            "double_tap_time": 0.3,  # Délai max entre deux taps
            "circle_min_points": 8,  # Points min pour détecter un cercle
            "pinch_threshold": dp(20),  # Changement min de distance pour un pinch
        }

        # Suivi des appuis longs
        self.long_press_events = {}

        # Suivi des touches actives pour les gestes multi-touch
        self.active_touches = []

        # État de pincement en cours
        self.pinch_info = {"active": False, "start_distance": 0, "ids": set()}

        # État de rotation en cours
        self.rotation_info = {"active": False, "start_angle": 0, "ids": set()}

        # Mode debug pour visualiser les traces
        self.debug_mode = False
        self.traces = {}

    def register_callback(self, gesture_type, callback):
        """
        Enregistre une fonction de callback pour un type de geste

        Args:
            gesture_type (str): Type de geste ('swipe_left', 'tap', etc.)
            callback (callable): Fonction à appeler quand le geste est détecté
        """
        if gesture_type in self.supported_gestures:
            self.callbacks[gesture_type].append(callback)
            return True
        return False

    def unregister_callback(self, gesture_type, callback):
        """
        Supprime un callback précédemment enregistré

        Args:
            gesture_type (str): Type de geste
            callback (callable): Fonction de callback à retirer
        """
        if gesture_type in self.callbacks and callback in self.callbacks[gesture_type]:
            self.callbacks[gesture_type].remove(callback)
            return True
        return False

    def on_touch_down(self, touch):
        """Gère le début d'une touch pour détecter un geste"""
        touch_id = touch.uid
        self.active_touches.append(touch_id)

        # Enregistre la position initiale
        x, y = touch.pos
        t = time.time()
        self.touch_history[touch_id] = [(x, y, t)]

        # Configure un événement d'appui long
        def trigger_long_press(dt, tid=touch_id):
            if tid in self.touch_history:
                # Calcule la distance de déplacement
                start_x, start_y, _ = self.touch_history[tid][0]
                history = self.touch_history[tid]
                if history:
                    last_x, last_y, _ = history[-1]
                    distance = math.sqrt((last_x - start_x) ** 2 + (last_y - start_y) ** 2)

                    # Seulement déclencher si peu de mouvement
                    if distance <= self.config["long_press_distance"]:
                        self.gesture_detected = True
                        for callback in self.callbacks["long_press"]:
                            callback({"pos": (last_x, last_y), "touch_id": tid})

        # Planifier l'événement d'appui long
        event = Clock.schedule_once(trigger_long_press, self.config["long_press_time"])
        self.long_press_events[touch_id] = event

        # Gestion du multi-touch pour pincement et rotation
        if len(self.active_touches) == 2:
            t1, t2 = self.active_touches
            if t1 in self.touch_history and t2 in self.touch_history:
                p1 = self.touch_history[t1][0][:2]  # x,y du premier toucher
                p2 = self.touch_history[t2][0][:2]  # x,y du second toucher

                # Distance initiale pour le pincement
                dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                self.pinch_info = {
                    "active": True,
                    "start_distance": dist,
                    "current_distance": dist,
                    "ids": set([t1, t2]),
                }

                # Angle initial pour la rotation
                angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
                self.rotation_info = {
                    "active": True,
                    "start_angle": angle,
                    "current_angle": angle,
                    "ids": set([t1, t2]),
                }

        # En mode debug, initialiser la trace
        if self.debug_mode:
            self.traces[touch_id] = []
            self.traces[touch_id].append((x, y))

        # Propager l'événement aux autres widgets
        return super(GestureDetector, self).on_touch_down(touch)

    def on_touch_move(self, touch):
        """Suivi du mouvement pour la détection de gestes"""
        touch_id = touch.uid

        # Si ce touch est suivi
        if touch_id in self.touch_history:
            # Ajouter le nouveau point à l'historique
            x, y = touch.pos
            t = time.time()
            self.touch_history[touch_id].append((x, y, t))

            # Mise à jour pour le debug
            if self.debug_mode and touch_id in self.traces:
                self.traces[touch_id].append((x, y))
                self.draw_debug_traces()

            # Vérifier si on doit annuler un appui long
            if touch_id in self.long_press_events:
                start_x, start_y, _ = self.touch_history[touch_id][0]
                distance = math.sqrt((x - start_x) ** 2 + (y - start_y) ** 2)

                if distance > self.config["long_press_distance"]:
                    # Trop de mouvement, annuler l'appui long
                    self.long_press_events[touch_id].cancel()
                    del self.long_press_events[touch_id]

            # Traitement des gestes multi-touch en cours
            if self.pinch_info["active"] and touch_id in self.pinch_info["ids"]:
                self._update_pinch_gesture()

            if self.rotation_info["active"] and touch_id in self.rotation_info["ids"]:
                self._update_rotation_gesture()

        return super(GestureDetector, self).on_touch_move(touch)

    def on_touch_up(self, touch):
        """Fin d'un toucher, détection finale du geste"""
        touch_id = touch.uid
        current_time = time.time()

        # Si ce touch était suivi
        if touch_id in self.touch_history:
            # Retirer des touches actives
            if touch_id in self.active_touches:
                self.active_touches.remove(touch_id)

            # Annuler l'appui long si en cours
            if touch_id in self.long_press_events:
                self.long_press_events[touch_id].cancel()
                del self.long_press_events[touch_id]

            # Récupérer l'historique complet
            history = self.touch_history[touch_id]
            if len(history) >= 2:
                # Points clés pour l'analyse
                start_x, start_y, start_time = history[0]
                end_x, end_y, end_time = history[-1]

                # Calcul de la vitesse et de la distance
                dx = end_x - start_x
                dy = end_y - start_y
                distance = math.sqrt(dx * dx + dy * dy)
                duration = end_time - start_time
                velocity = distance / duration if duration > 0 else 0

                # Détection du geste en fonction des seuils
                if duration < 0.2 and distance < dp(20):
                    # TAP (simple ou double)
                    tap_time_diff = current_time - self.last_tap_time
                    if tap_time_diff < self.config["double_tap_time"]:
                        # C'est un double tap
                        self.gesture_detected = True
                        for callback in self.callbacks["double_tap"]:
                            callback({"pos": (end_x, end_y), "touch_id": touch_id})
                    else:
                        # C'est un tap simple
                        self.gesture_detected = True
                        for callback in self.callbacks["tap"]:
                            callback({"pos": (end_x, end_y), "touch_id": touch_id})

                    # Mettre à jour l'heure du dernier tap
                    self.last_tap_time = current_time

                elif distance >= self.config["swipe_threshold"] and velocity >= self.config["swipe_velocity"]:
                    # SWIPE
                    # Déterminer la direction du swipe
                    if abs(dx) > abs(dy):
                        # Swipe horizontal
                        gesture = "swipe_right" if dx > 0 else "swipe_left"
                    else:
                        # Swipe vertical
                        gesture = "swipe_up" if dy > 0 else "swipe_down"

                    self.gesture_detected = True
                    for callback in self.callbacks[gesture]:
                        callback(
                            {
                                "start": (start_x, start_y),
                                "end": (end_x, end_y),
                                "velocity": velocity,
                                "touch_id": touch_id,
                            }
                        )

                # Détection de cercle (avec au moins 8 points)
                if len(history) >= self.config["circle_min_points"]:
                    if self._is_circle_gesture(history):
                        self.gesture_detected = True
                        # Calcul du centre et du rayon du cercle
                        points = [p[:2] for p in history]  # Juste les coordonnées x,y
                        center_x = sum(p[0] for p in points) / len(points)
                        center_y = sum(p[1] for p in points) / len(points)
                        radius = sum(math.sqrt((p[0] - center_x) ** 2 + (p[1] - center_y) ** 2) for p in points) / len(
                            points
                        )

                        for callback in self.callbacks["circle"]:
                            callback(
                                {
                                    "center": (center_x, center_y),
                                    "radius": radius,
                                    "touch_id": touch_id,
                                }
                            )

            # Nettoyage
            del self.touch_history[touch_id]

            # Réinitialiser les gestes multi-touch si nécessaire
            if touch_id in self.pinch_info["ids"]:
                self.pinch_info["active"] = False

            if touch_id in self.rotation_info["ids"]:
                self.rotation_info["active"] = False

            # Nettoyage des traces de debug
            if self.debug_mode and touch_id in self.traces:
                # Conserver la trace un moment avant de l'effacer
                def clear_trace(dt, tid=touch_id):
                    if tid in self.traces:
                        del self.traces[tid]
                        self.draw_debug_traces()

                Clock.schedule_once(clear_trace, 1.0)

        return super(GestureDetector, self).on_touch_up(touch)

    def _is_circle_gesture(self, points):
        """
        Détermine si les points forment approximativement un cercle
        Utilise l'écart-type des distances au centre comme critère
        """
        # Extraire les coordonnées x,y
        xy_points = [(p[0], p[1]) for p in points]

        # Calculer le centre approximatif
        center_x = sum(p[0] for p in xy_points) / len(xy_points)
        center_y = sum(p[1] for p in xy_points) / len(xy_points)

        # Calculer les distances au centre
        distances = [math.sqrt((p[0] - center_x) ** 2 + (p[1] - center_y) ** 2) for p in xy_points]
        mean_radius = sum(distances) / len(distances)

        # Écart-type des distances
        variance = sum((d - mean_radius) ** 2 for d in distances) / len(distances)
        std_dev = math.sqrt(variance)

        # Coefficient de variation (écart-type relatif)
        cv = std_dev / mean_radius if mean_radius > 0 else float("inf")

        # Calcul de l'angle parcouru
        angles = [math.atan2(p[1] - center_y, p[0] - center_x) for p in xy_points]
        # Normaliser les angles pour éviter les sauts de -π à π
        normalized_angles = []
        prev_angle = angles[0]
        for angle in angles:
            # Ajuster l'angle si nécessaire pour éviter les sauts
            while angle - prev_angle > math.pi:
                angle -= 2 * math.pi
            while angle - prev_angle < -math.pi:
                angle += 2 * math.pi
            normalized_angles.append(angle)
            prev_angle = angle

        # Calculer l'angle parcouru (différence max - min)
        angle_range = max(normalized_angles) - min(normalized_angles)

        # Vérifier si le geste est fermé (distance entre premier et dernier point)
        first_last_distance = math.sqrt(
            (xy_points[0][0] - xy_points[-1][0]) ** 2 + (xy_points[0][1] - xy_points[-1][1]) ** 2
        )

        # Conditions pour un cercle:
        # 1. Coefficient de variation faible (rayon constant)
        # 2. Angle parcouru suffisant (au moins 270°)
        # 3. Premier et dernier points proches l'un de l'autre
        is_circle = (
            cv < 0.25
            and angle_range > 4.7  # rayon relativement constant
            and first_last_distance  # au moins 270° (4.7 radians)
            < mean_radius  # fermeture du cercle
        )

        return is_circle

    def _update_pinch_gesture(self):
        """Met à jour et détecte le geste de pincement"""
        if not self.pinch_info["active"] or len(self.pinch_info["ids"]) != 2:
            return

        t1, t2 = self.pinch_info["ids"]
        if t1 not in self.touch_history or t2 not in self.touch_history:
            return

        p1 = self.touch_history[t1][-1][:2]  # Dernière position du toucher 1
        p2 = self.touch_history[t2][-1][:2]  # Dernière position du toucher 2

        # Calculer la nouvelle distance
        new_distance = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
        old_distance = self.pinch_info["current_distance"]
        delta = new_distance - old_distance

        # Mettre à jour la distance courante
        self.pinch_info["current_distance"] = new_distance

        # Vérifier s'il y a eu un changement significatif
        if abs(delta) > self.config["pinch_threshold"]:
            # Calculer le centre du pincement
            center_x = (p1[0] + p2[0]) / 2
            center_y = (p1[1] + p2[1]) / 2

            # Calculer le facteur d'échelle par rapport au début
            scale_factor = new_distance / self.pinch_info["start_distance"]

            # Déterminer s'il s'agit d'un pincement ou d'un écartement
            gesture = "pinch_in" if delta < 0 else "pinch_out"

            self.gesture_detected = True
            for callback in self.callbacks[gesture]:
                callback({"center": (center_x, center_y), "scale": scale_factor, "delta": delta})

    def _update_rotation_gesture(self):
        """Met à jour et détecte le geste de rotation"""
        if not self.rotation_info["active"] or len(self.rotation_info["ids"]) != 2:
            return

        t1, t2 = self.rotation_info["ids"]
        if t1 not in self.touch_history or t2 not in self.touch_history:
            return

        p1 = self.touch_history[t1][-1][:2]  # Dernière position du toucher 1
        p2 = self.touch_history[t2][-1][:2]  # Dernière position du toucher 2

        # Calculer le nouvel angle
        new_angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        old_angle = self.rotation_info["current_angle"]

        # Normaliser pour éviter les sauts de -π à π
        while new_angle - old_angle > math.pi:
            new_angle -= 2 * math.pi
        while new_angle - old_angle < -math.pi:
            new_angle += 2 * math.pi

        # Calculer le delta d'angle
        delta_angle = new_angle - old_angle

        # Mettre à jour l'angle courant
        self.rotation_info["current_angle"] = new_angle

        # Vérifier s'il y a eu un changement significatif
        if abs(delta_angle) > 0.1:  # Seuil d'environ 5.7 degrés
            # Calculer le centre de rotation
            center_x = (p1[0] + p2[0]) / 2
            center_y = (p1[1] + p2[1]) / 2

            # Calculer l'angle total par rapport au début
            total_angle = new_angle - self.rotation_info["start_angle"]

            self.gesture_detected = True
            for callback in self.callbacks["rotate"]:
                callback({"center": (center_x, center_y), "angle": total_angle, "delta": delta_angle})

    def draw_debug_traces(self):
        """Dessine les traces des touches pour le debug"""
        if not self.debug_mode:
            return

        self.canvas.after.clear()
        with self.canvas.after:
            # Dessiner chaque trace
            for touch_id, points in self.traces.items():
                if len(points) > 1:
                    Color(0.2, 0.8, 0.8, 0.7)
                    points_flat = []
                    for x, y in points:
                        points_flat.extend([x, y])
                    Line(points=points_flat, width=1.5)

    def set_debug_mode(self, enabled=True):
        """Active ou désactive le mode debug avec traces visibles"""
        self.debug_mode = enabled
        if not enabled:
            self.traces.clear()
            self.canvas.after.clear()


class GuidanceWaves(Widget):
    """
    Ondes visuelles qui guident doucement vers des points d'intérêt.

    Caractéristiques:
    - Création d'ondes concentriques pour attirer l'attention
    - Animation fluide et organique avec pulsation
    - Possibilité de guider vers un élément spécifique
    - Adaptable selon l'ambiance émotionnelle
    """

    mood = StringProperty("relax")
    wave_color = ListProperty([0.6, 0.8, 0.6, 1])
    waves = ListProperty([])
    anim_speed = NumericProperty(0.8)
    wave_amplitude = NumericProperty(0.7)

    def __init__(self, **kwargs):
        super(GuidanceWaves, self).__init__(**kwargs)

        # Initialiser les paramètres d'animation
        self.active = False
        self.target_position = (0, 0)
        self.create_wave_event = None

        # Configurer les paramètres par défaut de l'ambiance Relax
        self.update_mood("relax")

        # Démarrer la mise à jour des ondes
        Clock.schedule_interval(self.update_waves, 1 / 60.0)

    def update_mood(self, mood):
        """
        Met à jour l'ambiance émotionnelle des ondes

        Args:
            mood (str): 'relax', 'energetic' ou 'gentle'
        """
        self.mood = mood

        # Sélectionner les paramètres selon l'ambiance
        if mood == "relax":
            settings = AMBIENT_RELAX
        elif mood == "energetic":
            settings = AMBIENT_ENERGETIC
        else:  # mood == 'gentle'
            settings = AMBIENT_GENTLE

        # Mettre à jour les couleurs et animations
        color_hex = settings["colors"]["primary"]
        self.wave_color = get_color_from_hex(color_hex)
        self.anim_speed = settings["animation"]["base_speed"]
        self.wave_amplitude = settings["animation"]["wave_amplitude"]

    def guide_to(self, position):
        """
        Crée des ondes de guidance vers une position spécifique

        Args:
            position (tuple): Coordonnées (x, y) de la cible
        """
        # Annuler toute guidance précédente
        if self.create_wave_event:
            self.create_wave_event.cancel()

        # Nettoyer les ondes existantes
        self.waves = []

        # Configurer la nouvelle cible
        self.target_position = position
        self.active = True

        # Créer une première onde immédiatement
        self.create_wave()

        # Programmer la création d'ondes régulières
        interval = 2.0 / self.anim_speed  # Ajuster selon la vitesse
        self.create_wave_event = Clock.schedule_interval(lambda dt: self.create_wave(), interval)

    def stop_guidance(self):
        """Arrête la création d'ondes de guidance"""
        self.active = False
        if self.create_wave_event:
            self.create_wave_event.cancel()
            self.create_wave_event = None

    def create_wave(self):
        """Crée une nouvelle onde à la position cible"""
        if not self.active:
            return False

        # Créer un dictionnaire pour stocker les propriétés de l'onde
        wave = {
            "center": self.target_position,
            "size": 0,
            "opacity": 1.0,
            "birth_time": time.time(),
            "color": self.wave_color[:],  # Copie pour éviter les références partagées
            "variance": random.uniform(0.95, 1.05),  # Légère variation pour un effet organique
        }

        # Ajouter à la liste des ondes
        self.waves.append(wave)

        return True

    def update_waves(self, dt):
        """Met à jour et dessine toutes les ondes actives"""
        # Nettoyer les ondes expirées
        expired_waves = []
        for wave in self.waves:
            age = time.time() - wave["birth_time"]
            # Supprimer si complètement transparente
            if wave["opacity"] <= 0:
                expired_waves.append(wave)

        # Retirer les ondes expirées
        for wave in expired_waves:
            if wave in self.waves:
                self.waves.remove(wave)

        # Redessiner toutes les ondes
        self.canvas.before.clear()
        with self.canvas.before:
            for wave in self.waves:
                # Calculer la taille et l'opacité en fonction de l'âge
                age = time.time() - wave["birth_time"]

                # Taille croissante
                base_size = 200 * self.wave_amplitude
                wave["size"] = age * base_size * self.anim_speed * wave["variance"]

                # Opacité décroissante
                wave["opacity"] = max(0, 1.0 - age * self.anim_speed * 0.5)

                # Dessiner l'onde circulaire
                color = wave["color"]
                Color(color[0], color[1], color[2], wave["opacity"])

                # Position ajustée pour centrer l'ellipse
                pos = (wave["center"][0] - wave["size"] / 2, wave["center"][1] - wave["size"] / 2)

                # Dessiner l'ellipse
                Ellipse(pos=pos, size=(wave["size"], wave["size"]))


class AuraAssistant(Widget):
    """
    Petit halo lumineux qui suit l'utilisateur et guide discrètement.

    Caractéristiques:
    - Déplacement fluide suivant le doigt/curseur
    - Pulsation subtile pour donner l'impression de vie
    - Variations d'intensité selon l'activité
    - Couleurs adaptées à l'ambiance émotionnelle
    """

    mood = StringProperty("relax")
    aura_color = ListProperty([0.6, 0.8, 0.6, 1])
    aura_size = NumericProperty(dp(40))
    aura_position = ListProperty([100, 100])
    aura_opacity = NumericProperty(0.8)

    def __init__(self, **kwargs):
        super(AuraAssistant, self).__init__(**kwargs)

        # État interne
        self.pulsing = False
        self.guiding = False
        self.following = True
        self.target_position = self.aura_position[:]
        self.base_size = dp(40)

        # Paramètres d'animation
        self.pulse_strength = 1.2  # Amplitude relative de la pulsation
        self.follow_speed = 0.12  # Vitesse de suivi du doigt (0-1)

        # Paramètres de respiration (effet de vie)
        self.breath_phase = random.uniform(0, 2 * math.pi)  # Phase aléatoire
        self.breath_speed = 1.0  # Vitesse de respiration

        # Configurer l'ambiance par défaut
        self.update_mood("relax")

        # Démarrer l'animation continue
        self._start_breathing()
        Clock.schedule_interval(self.update_aura, 1 / 60.0)

    def update_mood(self, mood):
        """
        Met à jour l'ambiance émotionnelle de l'aura

        Args:
            mood (str): 'relax', 'energetic' ou 'gentle'
        """
        self.mood = mood

        # Sélectionner les paramètres selon l'ambiance
        if mood == "relax":
            settings = AMBIENT_RELAX
            self.breath_speed = 0.8
            self.pulse_strength = 1.2
        elif mood == "energetic":
            settings = AMBIENT_ENERGETIC
            self.breath_speed = 1.3
            self.pulse_strength = 1.5
        else:  # mood == 'gentle'
            settings = AMBIENT_GENTLE
            self.breath_speed = 0.6
            self.pulse_strength = 1.1

        # Mettre à jour les couleurs
        color_hex = settings["colors"]["primary"]
        self.aura_color = get_color_from_hex(color_hex)

    def _start_breathing(self):
        """
        Démarre l'animation de respiration pour donner vie à l'aura
        """

        # Animation continue sur place pour simuler la respiration
        def update_breath(dt):
            if not self.pulsing and not self.guiding:
                # Simuler une respiration douce avec une sinusoïde
                self.breath_phase += dt * self.breath_speed
                breath_factor = 0.1 * math.sin(self.breath_phase) + 1.0
                self.aura_size = self.base_size * breath_factor

        # Programmer la mise à jour de la respiration
        Clock.schedule_interval(update_breath, 1 / 30.0)

    def follow_touch(self, position):
        """
        Fait suivre l'aura à une position de toucher

        Args:
            position (tuple): Position (x, y) à suivre
        """
        if self.following:
            self.target_position = position

    def guide_to(self, position, duration=0.8):
        """
        Guide explicitement vers une position avec animation

        Args:
            position (tuple): Position cible (x, y)
            duration (float): Durée de l'animation en secondes
        """
        # Arrêter le suivi normal pendant le guidage
        self.following = False
        self.guiding = True

        # Annuler toute animation en cours
        Animation.cancel_all(self)

        # Créer l'animation de déplacement
        anim = Animation(
            aura_position=position,
            aura_size=self.base_size * 1.2,  # Légèrement plus grand pendant le guidage
            d=duration,
            t="out_quad",
        )

        # Fonction pour terminer le guidage
        def on_complete(*args):
            self.guiding = False
            self.following = True
            self.pulse_animation()  # Faire pulser pour indiquer l'arrivée

        anim.bind(on_complete=on_complete)
        anim.start(self)

    def pulse_animation(self, intensity=1.0, duration=0.5):
        """
        Crée une animation pulsante pour attirer l'attention

        Args:
            intensity (float): Intensité relative de la pulsation (1.0 = normale)
            duration (float): Durée de la pulsation en secondes
        """
        self.pulsing = True

        # Utiliser le moteur émotionnel de Jeffrey pour feedback vocal
        app = App.get_running_app()
        if hasattr(app, "jeffrey") and hasattr(app.jeffrey, "say_with_emotion"):
            app.jeffrey.say_with_emotion(
                base_phrases=[
                    "J'ai quelque chose à te montrer.",
                    "Regarde par ici.",
                    "Voici ce que je voulais te montrer.",
                    "Ton attention est requise ici.",
                    "Observe cet élément avec moi.",
                ],
                context="attention",
            )

        # Calculer l'amplitude en fonction de l'intensité
        scale_up = self.base_size * (1 + (self.pulse_strength - 1) * intensity)

        # Créer l'animation de pulsation
        pulse_anim = Animation(
            aura_size=scale_up, aura_opacity=min(1.0, self.aura_opacity + 0.2), d=duration * 0.4
        ) + Animation(aura_size=self.base_size, aura_opacity=self.aura_opacity, d=duration * 0.6)

        # Terminer l'état de pulsation
        pulse_anim.bind(on_complete=lambda *x: setattr(self, "pulsing", False))

        # Lancer l'animation
        pulse_anim.start(self)

    def update_aura(self, dt):
        """Met à jour la position et l'apparence de l'aura"""
        # Mise à jour de la position
        if self.following and not self.guiding:
            # Suivi fluide par interpolation
            current_x, current_y = self.aura_position
            target_x, target_y = self.target_position

            self.aura_position[0] += (target_x - current_x) * self.follow_speed
            self.aura_position[1] += (target_y - current_y) * self.follow_speed

        # Redessiner l'aura
        self.canvas.before.clear()
        with self.canvas.before:
            # Halo externe diffus
            Color(self.aura_color[0], self.aura_color[1], self.aura_color[2], self.aura_opacity * 0.3)
            Ellipse(
                pos=(
                    self.aura_position[0] - self.aura_size * 0.8,
                    self.aura_position[1] - self.aura_size * 0.8,
                ),
                size=(self.aura_size * 1.6, self.aura_size * 1.6),
            )

            # Halo intermédiaire
            Color(self.aura_color[0], self.aura_color[1], self.aura_color[2], self.aura_opacity * 0.6)
            Ellipse(
                pos=(
                    self.aura_position[0] - self.aura_size * 0.6,
                    self.aura_position[1] - self.aura_size * 0.6,
                ),
                size=(self.aura_size * 1.2, self.aura_size * 1.2),
            )

            # Cœur lumineux
            Color(self.aura_color[0], self.aura_color[1], self.aura_color[2], self.aura_opacity)
            Ellipse(
                pos=(
                    self.aura_position[0] - self.aura_size * 0.35,
                    self.aura_position[1] - self.aura_size * 0.35,
                ),
                size=(self.aura_size * 0.7, self.aura_size * 0.7),
            )


class EmotionalCoach(Widget):
    """
    Composant qui offre une aide contextuelle discrète
    quand l'utilisateur semble hésiter.

    Caractéristiques:
    - Apparaît subtilement quand une hésitation est détectée
    - Adapte ses conseils au contexte et à l'humeur
    - Animations d'apparition/disparition fluides
    - Ne prend pas trop de place à l'écran
    """

    mood = StringProperty("relax")
    coach_color = ListProperty([0.6, 0.8, 0.6, 1])
    coach_visible = BooleanProperty(False)
    coach_message = StringProperty("")
    coach_opacity = NumericProperty(0)
    bg_color = ListProperty([0.1, 0.1, 0.1, 0.7])

    def __init__(self, **kwargs):
        super(EmotionalCoach, self).__init__(**kwargs)

        # Position et taille
        self.size_hint = (None, None)
        self.width = dp(300)
        self.height = dp(100)
        self.pos_hint = {"center_x": 0.5, "bottom": 0.05}

        # Messages d'aide selon l'ambiance
        self.messages = {
            "relax": [
                "Prenez votre temps, explorez à votre rythme...",
                "Un simple glissement vers les côtés révèle plus d'options...",
                "Vous cherchez quelque chose ? Un geste de cercle appelle l'aide...",
                "Respirez et laissez-vous guider par l'aura...",
                "Les ondes vous montrent le chemin, suivez-les doucement...",
            ],
            "energetic": [
                "Glissez rapidement pour naviguer!",
                "Besoin d'aide? Dessinez un cercle et je suis là!",
                "Explorez avec énergie - touchez, glissez, découvrez!",
                "Suivez l'aura bleue pour vous orienter rapidement!",
                "Un double tap vous emmène directement à l'action!",
            ],
            "gentle": [
                "Je sens une hésitation... puis-je vous aider?",
                "Laissez-moi vous guider en douceur...",
                "Un petit cercle avec votre doigt m'appelle à tout moment",
                "L'aura rose vous montre le chemin avec tendresse",
                "Prenez tout votre temps, je suis là pour vous accompagner",
            ],
        }

        # État interne
        self.display_time = 4.0  # Temps d'affichage en secondes
        self.fade_time = 0.8  # Temps de transition en secondes
        self.dismiss_timer = None

        # Mettre à jour l'ambiance par défaut
        self.update_mood("relax")

        # Configurer les événements de dessin
        self.bind(size=self._update_display, pos=self._update_display)

    def update_mood(self, mood):
        """
        Met à jour l'ambiance émotionnelle du coach

        Args:
            mood (str): 'relax', 'energetic' ou 'gentle'
        """
        self.mood = mood

        # Sélectionner les paramètres selon l'ambiance
        if mood == "relax":
            settings = AMBIENT_RELAX
        elif mood == "energetic":
            settings = AMBIENT_ENERGETIC
        else:  # mood == 'gentle'
            settings = AMBIENT_GENTLE

        # Mettre à jour les couleurs
        self.coach_color = get_color_from_hex(settings["colors"]["primary"])
        accent_color = get_color_from_hex(settings["colors"]["accent"])
        self.bg_color = [0.1, 0.1, 0.1, 0.7]  # Fond semi-transparent

    def show_help(self, context=None):
        """
        Affiche un message d'aide adapté au contexte

        Args:
            context (dict, optional): Informations contextuelles pour personnaliser l'aide
        """
        # Sélectionner un message adapté à l'ambiance
        message = random.choice(self.messages[self.mood])

        # Personnaliser le message selon le contexte si fourni
        if context:
            if "area" in context:
                area = context["area"]
                if area == "menu":
                    message = f"Pour naviguer dans le menu, glissez {self.mood}ment."
                elif area == "dashboard":
                    message = "Explorez le tableau de bord avec des gestes de balayage."

            if "action" in context:
                action = context["action"]
                if action == "select":
                    message = "Tapotez doucement pour sélectionner."
                elif action == "zoom":
                    message = "Pincez pour zoomer, écartez pour dézoomer."

        # Utiliser le moteur émotionnel de Jeffrey pour le message vocal
        app = App.get_running_app()
        if hasattr(app, "jeffrey") and hasattr(app.jeffrey, "say_with_emotion"):
            app.jeffrey.say_with_emotion(
                base_phrases=[
                    message,
                    "Besoin d'aide ? " + message,
                    "Je peux t'aider. " + message,
                    "Conseil : " + message,
                    "Voici une suggestion : " + message,
                ],
                context="help",
            )

        # Mettre à jour le message
        self.coach_message = message

        # Afficher le coach avec animation
        self._show_with_animation()

    def _show_with_animation(self):
        """Affiche le coach avec une animation douce"""
        # Annuler le timer de disparition s'il est actif
        if self.dismiss_timer:
            self.dismiss_timer.cancel()

        # Annuler toute animation en cours
        Animation.cancel_all(self)

        # Animer l'apparition
        self.coach_visible = True
        appear_anim = Animation(coach_opacity=1, d=self.fade_time, t="out_quad")
        appear_anim.start(self)

        # Programmer la disparition automatique
        self.dismiss_timer = Clock.schedule_once(self._hide_with_animation, self.display_time)

    def _hide_with_animation(self, dt=None):
        """Cache le coach avec une animation douce"""
        # Annuler toute animation en cours
        Animation.cancel_all(self)

        # Animer la disparition
        disappear_anim = Animation(coach_opacity=0, d=self.fade_time, t="in_quad")

        # Marquer comme invisible une fois complètement disparu
        def on_complete(*args):
            self.coach_visible = False

        disappear_anim.bind(on_complete=on_complete)
        disappear_anim.start(self)

    def _update_display(self, *args):
        """Met à jour l'affichage quand la taille ou position change"""
        self.canvas.before.clear()

        # Ne dessiner que si visible
        if not self.coach_visible:
            return

        with self.canvas.before:
            # Fond arrondi semi-transparent
            Color(
                self.bg_color[0],
                self.bg_color[1],
                self.bg_color[2],
                self.bg_color[3] * self.coach_opacity,
            )
            RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(15)])

            # Bordure subtile
            Color(
                self.coach_color[0],
                self.coach_color[1],
                self.coach_color[2],
                0.7 * self.coach_opacity,
            )
            Line(
                rounded_rectangle=(self.pos[0], self.pos[1], self.size[0], self.size[1], dp(15)),
                width=1.5,
            )

    def on_coach_opacity(self, instance, value):
        """Appelé quand l'opacité change - met à jour l'affichage"""
        self._update_display()

    def on_coach_message(self, instance, value):
        """Appelé quand le message change - met à jour l'affichage"""
        self.canvas.clear()

        with self.canvas:
            # Texte du message
            Color(1, 1, 1, self.coach_opacity)
            Rectangle(
                pos=(self.pos[0] + dp(15), self.pos[1] + dp(15)),
                size=(self.width - dp(30), self.height - dp(30)),
            )

            # Note: Le texte sera dessiné par KivyLanguage via un Label


class LivingTutorial(FloatLayout):
    """
    Tutoriel vivant qui guide les nouveaux utilisateurs
    de manière fluide et non-intrusive.

    Caractéristiques:
    - Approche narrative et émotionnelle
    - Animations douces et progressives
    - S'adapte à l'ambiance émotionnelle
    - Peut être rappelé à tout moment
    """

    mood = StringProperty("relax")
    tutorial_color = ListProperty([0.6, 0.8, 0.6, 1])
    tutorial_step = NumericProperty(0)
    tutorial_opacity = NumericProperty(0)

    def __init__(self, **kwargs):
        super(LivingTutorial, self).__init__(**kwargs)

        # Étapes du tutoriel (contenu adapté selon l'ambiance)
        self.tutorial_steps = {
            "relax": [
                "Bienvenue dans Jeffrey... prenez votre temps pour explorer",
                "L'aura verte vous suit pour vous guider dans l'interface",
                "Glissez doucement pour naviguer entre les sections",
                "Si vous hésitez, des conseils apparaîtront en bas de l'écran",
                "Pour appeler de l'aide, dessinez simplement un cercle",
                "Découvrez à votre rythme... Jeffrey s'adapte à vous",
            ],
            "energetic": [
                "Bienvenue dans Jeffrey - prêt pour l'aventure?",
                "Votre aura bleue vous guide à travers l'interface",
                "Glissez rapidement pour naviguer et explorer",
                "Double-tapez pour accéder directement aux fonctions",
                "Un cercle rapide appelle l'aide instantanément",
                "Jeffrey s'adapte à votre énergie - explorez librement!",
            ],
            "gentle": [
                "Bienvenue dans votre espace Jeffrey... respirez doucement",
                "L'aura rose vous accompagne avec bienveillance",
                "De légers gestes suffisent pour naviguer en douceur",
                "Prenez votre temps, Jeffrey s'adapte à votre rythme",
                "Un petit cercle avec votre doigt m'appelle à tout moment",
                "Laissez-vous porter par l'expérience...",
            ],
        }

        # État du tutoriel
        self.animation_in_progress = False
        self.is_visible = False

        # Mettre à jour l'ambiance par défaut
        self.update_mood("relax")

    def update_mood(self, mood):
        """
        Met à jour l'ambiance émotionnelle du tutoriel

        Args:
            mood (str): 'relax', 'energetic' ou 'gentle'
        """
        self.mood = mood

        # Sélectionner les paramètres selon l'ambiance
        if mood == "relax":
            settings = AMBIENT_RELAX
        elif mood == "energetic":
            settings = AMBIENT_ENERGETIC
        else:  # mood == 'gentle'
            settings = AMBIENT_GENTLE

        # Mettre à jour les couleurs
        self.tutorial_color = get_color_from_hex(settings["colors"]["primary"])

    def start_tutorial(self):
        """Démarre ou redémarre le tutoriel depuis le début"""
        self.tutorial_step = 0
        self.show_current_step()

    def show_current_step(self):
        """Affiche l'étape actuelle du tutoriel"""
        # Vérifier si on a atteint la fin du tutoriel
        steps = self.tutorial_steps[self.mood]
        if self.tutorial_step >= len(steps):
            self.end_tutorial()
            return

        # Utiliser le moteur émotionnel de Jeffrey pour le message vocal
        app = App.get_running_app()
        if hasattr(app, "jeffrey") and hasattr(app.jeffrey, "say_with_emotion"):
            app.jeffrey.say_with_emotion(
                base_phrases=[
                    steps[self.tutorial_step],
                    "Étape " + str(self.tutorial_step + 1) + " : " + steps[self.tutorial_step],
                    "Pour continuer ton apprentissage : " + steps[self.tutorial_step],
                ],
                context="tutorial",
            )

        # Préparation de l'affichage
        self.clear_widgets()
        self.canvas.before.clear()

        # Fond semi-transparent
        with self.canvas.before:
            Color(0.05, 0.05, 0.1, 0.8 * self.tutorial_opacity)
            Rectangle(pos=(0, 0), size=self.size)

        # Créer le conteneur pour le contenu du tutoriel
        content_box = BoxLayout(
            orientation="vertical",
            size_hint=(0.8, None),
            height=dp(250),
            pos_hint={"center_x": 0.5, "center_y": 0.5},
            spacing=dp(20),
            padding=[dp(20), dp(20)],
        )

        # Message du tutoriel
        message = Label(
            text=steps[self.tutorial_step],
            color=(1, 1, 1, 1),
            font_size=dp(22),
            size_hint=(1, 0.8),
            halign="center",
            valign="middle",
        )

        # Indicateur d'étape
        step_indicator = Label(
            text=f"Étape {self.tutorial_step + 1}/{len(steps)}",
            color=(0.8, 0.8, 0.8, 0.7),
            font_size=dp(16),
            size_hint=(1, 0.2),
            halign="center",
        )

        # Bouton continuer (représenté visuellement)
        continue_hint = Label(
            text="Tapotez pour continuer...",
            color=(0.9, 0.9, 0.9, 0.5),
            font_size=dp(14),
            size_hint=(1, 0.1),
            halign="center",
        )

        # Assembler l'interface
        content_box.add_widget(message)
        content_box.add_widget(step_indicator)
        content_box.add_widget(continue_hint)

        # Ajouter au tutoriel
        self.add_widget(content_box)

        # Animation d'apparition
        self.is_visible = True
        Animation.cancel_all(self)
        anim = Animation(tutorial_opacity=1, d=0.5, t="out_quad")

        def on_complete(*args):
            self.animation_in_progress = False

        anim.bind(on_complete=on_complete)
        self.animation_in_progress = True
        anim.start(self)

    def next_step(self):
        """Passe à l'étape suivante du tutoriel"""
        if self.animation_in_progress:
            return

        # Animation de transition
        anim_out = Animation(tutorial_opacity=0, d=0.3, t="in_quad")

        def show_next(*args):
            self.tutorial_step += 1
            self.show_current_step()

        anim_out.bind(on_complete=show_next)
        self.animation_in_progress = True
        anim_out.start(self)

    def end_tutorial(self):
        """Termine le tutoriel avec une animation de sortie"""
        if not self.is_visible:
            return

        # Utiliser le moteur émotionnel de Jeffrey pour le message de fin
        app = App.get_running_app()
        if hasattr(app, "jeffrey") and hasattr(app.jeffrey, "say_with_emotion"):
            app.jeffrey.say_with_emotion(
                base_phrases=[
                    "Vous avez terminé le tutoriel. Vous pouvez maintenant explorer librement.",
                    "Excellent ! Vous êtes prêt à utiliser Jeffrey pleinement.",
                    "Tutoriel terminé. Jeffrey est désormais à votre service.",
                    "Bravo ! Vous maîtrisez maintenant les bases de l'interface.",
                ],
                context="tutorial_complete",
            )

        Animation.cancel_all(self)
        anim = Animation(tutorial_opacity=0, d=0.5, t="in_quad")

        def on_complete(*args):
            self.is_visible = False
            self.animation_in_progress = False
            self.clear_widgets()

        anim.bind(on_complete=on_complete)
        self.animation_in_progress = True
        anim.start(self)

    def on_touch_down(self, touch):
        """Gère les interactions tactiles avec le tutoriel"""
        if self.is_visible and self.collide_point(*touch.pos):
            if not self.animation_in_progress:
                self.next_step()
            return True
        return super(LivingTutorial, self).on_touch_down(touch)


class GuidanceManager(FloatLayout):
    """
    Cœur du système de guidance émotionnelle et intuitive.

    Coordonne tous les composants:
    - GestureDetector pour les interactions naturelles
    - GuidanceWaves pour guider l'attention
    - AuraAssistant pour accompagner l'utilisateur
    - EmotionalCoach pour l'aide contextuelle
    - LivingTutorial pour les nouveaux utilisateurs

    Applique les changements d'ambiance émotionnelle à tout le système.
    """

    current_mood = OptionProperty("relax", options=["relax", "energetic", "gentle"])
    settings = DictProperty(AMBIENT_RELAX)
    first_time_user = BooleanProperty(True)
    hesitation_detected = BooleanProperty(False)

    def __init__(self, **kwargs):
        super(GuidanceManager, self).__init__(**kwargs)

        # Créer les composants principaux
        self.gesture_detector = GestureDetector()
        self.guidance_waves = GuidanceWaves()
        self.aura_assistant = AuraAssistant()
        self.emotional_coach = EmotionalCoach()
        self.living_tutorial = LivingTutorial()

        # Ajouter les composants dans l'ordre d'empilement
        self.add_widget(self.gesture_detector)  # En arrière-plan pour capter les gestes
        self.add_widget(self.guidance_waves)  # Ondes en fond
        self.add_widget(self.aura_assistant)  # Aura au-dessus des ondes
        self.add_widget(self.emotional_coach)  # Coach en avant-plan

        # Configurer les interactions par gestes
        self._setup_gesture_interactions()

        # Configurer l'ambiance initiale
        self.update_mood(self.current_mood)

        # Démarrer la détection des hésitations
        Clock.schedule_interval(self.check_for_hesitation, 2.0)

        # Pour le suivi du pointeur/doigt
        Window.bind(mouse_pos=self.on_mouse_move)

        # Démarrer le tutoriel si premier utilisateur
        if self.first_time_user:
            Clock.schedule_once(lambda dt: self.show_tutorial(), 1.0)

        # Charger les préférences utilisateur si disponibles
        self.load_user_preferences()

    def update_mood(self, mood):
        """
        Change l'ambiance émotionnelle de tout le système

        Args:
            mood (str): 'relax', 'energetic' ou 'gentle'
        """
        self.current_mood = mood

        # Choisir les paramètres selon l'ambiance
        if mood == "relax":
            self.settings = AMBIENT_RELAX
        elif mood == "energetic":
            self.settings = AMBIENT_ENERGETIC
        else:  # gentle
            self.settings = AMBIENT_GENTLE

        # Propager le changement d'ambiance à tous les composants
        self.guidance_waves.update_mood(mood)
        self.aura_assistant.update_mood(mood)
        self.emotional_coach.update_mood(mood)
        self.living_tutorial.update_mood(mood)

        # Jouer le son d'ambiance si disponible
        self._play_ambient_sound()

        # Sauvegarder la préférence
        self.save_user_preferences()

    def _setup_gesture_interactions(self):
        """Configure les interactions par gestes pour tous les composants"""
        # Geste de glissement (swipe)
        self.gesture_detector.register_callback("swipe_left", self._on_swipe_left)
        self.gesture_detector.register_callback("swipe_right", self._on_swipe_right)
        self.gesture_detector.register_callback("swipe_up", self._on_swipe_up)
        self.gesture_detector.register_callback("swipe_down", self._on_swipe_down)

        # Gestes de tap
        self.gesture_detector.register_callback("tap", self._on_tap)
        self.gesture_detector.register_callback("double_tap", self._on_double_tap)
        self.gesture_detector.register_callback("long_press", self._on_long_press)

        # Geste spécial: cercle pour l'aide
        self.gesture_detector.register_callback("circle", self._on_circle_gesture)

        # Gestes de pincement (zoom)
        self.gesture_detector.register_callback("pinch_in", self._on_pinch_in)
        self.gesture_detector.register_callback("pinch_out", self._on_pinch_out)

    def on_mouse_move(self, instance, pos):
        """Suit le mouvement de la souris/doigt"""
        # Faire suivre l'aura
        self.aura_assistant.follow_touch(pos)

    def check_for_hesitation(self, dt):
        """
        Vérifie si l'utilisateur semble hésiter et propose de l'aide
        """
        # Logique simplifiée de détection d'hésitation
        # Dans une implémentation réelle, utiliserait:
        # - Temps passé sans interaction
        # - Mouvements répétitifs sans action
        # - Retours en arrière fréquents

        # Simuler une détection aléatoire pour l'exemple
        if not self.hesitation_detected and random.random() < 0.3:  # 30% de chance
            self.hesitation_detected = True
            self.offer_help()
        else:
            self.hesitation_detected = False

    def offer_help(self, context=None):
        """
        Propose de l'aide à l'utilisateur selon le contexte

        Args:
            context (dict, optional): Informations contextuelles pour personnaliser l'aide
        """
        if not context:
            context = {}

        # Faire pulser légèrement l'aura pour attirer l'attention
        self.aura_assistant.pulse_animation(intensity=0.8)

        # Afficher un message d'aide adapté
        self.emotional_coach.show_help(context)

    def guide_to_element(self, element_id, position=None):
        """
        Guide l'utilisateur vers un élément de l'interface

        Args:
            element_id (str): Identifiant de l'élément cible
            position (tuple, optional): Position (x, y) si connue, sinon détectée
        """
        # Déterminer la position si non fournie
        if not position:
            # Dans une implémentation réelle, on chercherait l'élément par son ID
            # Pour l'exemple, utiliser une position aléatoire
            position = (
                random.randint(100, Window.width - 100),
                random.randint(100, Window.height - 100),
            )

        # Créer des ondes guidant vers la position
        self.guidance_waves.guide_to(position)

        # Guider l'aura vers la position
        self.aura_assistant.guide_to(position)

        # Jouer un son de feedback
        self._play_guidance_sound()

    def show_tutorial(self):
        """Affiche ou réaffiche le tutoriel vivant"""
        # Arrêter toute guidance en cours
        self.guidance_waves.stop_guidance()

        # Ajouter le tutoriel au-dessus de tout si pas déjà présent
        if self.living_tutorial not in self.children:
            self.add_widget(self.living_tutorial)

        # Démarrer le tutoriel
        self.living_tutorial.start_tutorial()

    def save_user_preferences(self):
        """Sauvegarde les préférences utilisateur"""
        try:
            # Créer un dictionnaire des préférences
            preferences = {
                "mood": self.current_mood,
                "first_time_user": False,  # Une fois sauvegardé, n'est plus "première fois"
                "last_used": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Créer le répertoire de données si nécessaire
            data_dir = os.path.expanduser("~/.jeffrey_data")
            os.makedirs(data_dir, exist_ok=True)

            # Sauvegarder dans un fichier JSON
            pref_file = os.path.join(data_dir, "preferences.json")
            with open(pref_file, "w") as f:
                json.dump(preferences, f)

        except Exception as e:
            print(f"Erreur lors de la sauvegarde des préférences: {e}")

    def load_user_preferences(self):
        """Charge les préférences utilisateur si elles existent"""
        try:
            # Chemin du fichier de préférences
            pref_file = os.path.expanduser("~/.jeffrey_data/preferences.json")

            # Vérifier si le fichier existe
            if os.path.exists(pref_file):
                with open(pref_file) as f:
                    preferences = json.load(f)

                # Appliquer les préférences
                if "mood" in preferences:
                    self.update_mood(preferences["mood"])

                if "first_time_user" in preferences:
                    self.first_time_user = preferences["first_time_user"]

        except Exception as e:
            print(f"Erreur lors du chargement des préférences: {e}")

    def _play_ambient_sound(self):
        """Joue le son d'ambiance correspondant au mode actuel"""
        # Cette méthode utilise le moteur émotionnel de Jeffrey
        sound_file = self.settings["sound"]["ambient"]
        print(f"Jouer le son d'ambiance: {sound_file}")

        # Utiliser le moteur émotionnel de Jeffrey
        app = App.get_running_app()
        if hasattr(app, "jeffrey") and hasattr(app.jeffrey, "say_with_emotion"):
            app.jeffrey.say_with_emotion(
                base_phrases=[
                    "Je m'adapte à cette ambiance.",
                    "Je change mon environnement sonore.",
                    "J'ajuste mon état émotionnel à cette atmosphère.",
                    "Voici une ambiance qui correspond à mon état.",
                ],
                context="ambient_change",
            )

    def _play_guidance_sound(self):
        """Joue un son de feedback pour guider l'utilisateur"""
        # Cette méthode utilise le moteur émotionnel de Jeffrey
        sound_file = self.settings["sound"]["feedback"]
        print(f"Jouer le son de guidance: {sound_file}")

        # Utiliser le moteur émotionnel de Jeffrey
        app = App.get_running_app()
        if hasattr(app, "jeffrey") and hasattr(app.jeffrey, "say_with_emotion"):
            app.jeffrey.say_with_emotion(
                base_phrases=[
                    "Voici un conseil.",
                    "Je te guide dans cette direction.",
                    "C'est par ici.",
                    "Laisse-moi t'orienter.",
                    "Regarde plutôt de ce côté.",
                    "Je te suggère cette approche.",
                ],
                context="guidance",
            )

    # Gestionnaires d'événements pour les gestes
    def _on_swipe_left(self, data):
        print(f"Swipe gauche détecté: {data}")
        # Logique spécifique au swipe gauche

    def _on_swipe_right(self, data):
        print(f"Swipe droite détecté: {data}")
        # Logique spécifique au swipe droite

    def _on_swipe_up(self, data):
        print(f"Swipe haut détecté: {data}")
        # Logique spécifique au swipe haut

    def _on_swipe_down(self, data):
        print(f"Swipe bas détecté: {data}")
        # Logique spécifique au swipe bas

    def _on_tap(self, data):
        print(f"Tap détecté: {data}")
        # Faire pulser légèrement l'aura pour feedback
        self.aura_assistant.pulse_animation(intensity=0.6, duration=0.3)

    def _on_double_tap(self, data):
        print(f"Double tap détecté: {data}")
        # Faire pulser plus fortement l'aura
        self.aura_assistant.pulse_animation(intensity=1.2, duration=0.4)

    def _on_long_press(self, data):
        print(f"Appui long détecté: {data}")
        # Logique spécifique à l'appui long

    def _on_circle_gesture(self, data):
        print(f"Geste de cercle détecté: {data}")
        # Afficher l'aide quand l'utilisateur dessine un cercle
        self.offer_help({"action": "help_requested"})

    def _on_pinch_in(self, data):
        print(f"Pincement (zoom in) détecté: {data}")
        # Logique spécifique au pincement

    def _on_pinch_out(self, data):
        print(f"Écartement (zoom out) détecté: {data}")
        # Logique spécifique à l'écartement


class JeffreyGuidanceApp(App):
    """
    Application Kivy pour tester le système de guidance émotionnelle.
    """

    # Paramètres pour le mode debug
    DEBUG = True  # Activer le mode debug pour tester

    def build(self):
        """Construit l'interface principale de l'application"""
        # Créer le gestionnaire de guidance
        guidance_manager = GuidanceManager()

        # PACK 5/9 : Configurer le noyau émotionnel pour Jeffrey
        from jeffrey.core.emotions.memory.souvenirs_affectifs import SouvenirsAffectifs
        from jeffrey.core.jeffrey_emotional_core import JeffreyEmotionalCore
        from jeffrey.core.lien_affectif import LienAffectif

        # Initialiser les composants du Pack 9
        souvenirs_affectifs = SouvenirsAffectifs(chemin_sauvegarde="data/test_souvenirs_affectifs.json")

        lien_affectif = LienAffectif(
            niveau_attachement=0.4,
            niveau_confiance=0.5,
            chemin_sauvegarde="data/test_lien_affectif.json",
            souvenirs_affectifs=souvenirs_affectifs,
        )

        self.jeffrey_core = JeffreyEmotionalCore(lien_affectif=lien_affectif)

        # Mode debug : tester la personnalité évolutive et l'intimité
        if self.DEBUG:
            # Simule un profil de personnalité plus mature
            self.jeffrey_core.personnalite_evolutive.maturite = 0.8  # Niveau adulte
            self.jeffrey_core.personnalite_evolutive.pudeur = 0.4  # Pudeur modérée
            self.jeffrey_core.personnalite_evolutive.attachement = 0.7  # Attachement élevé
            self.jeffrey_core.personnalite_evolutive.confiance = 0.8  # Confiance élevée
            self.jeffrey_core.personnalite_evolutive.curiosite_sensorielle = 0.6  # Curiosité élevée

            # Propager au profil
            self.jeffrey_core.personnalite_evolutive._mettre_a_jour_stade_developpement()

            # Configurer le contexte privé pour les tests
            self.jeffrey_core.definir_contexte_interaction("private")

            # PACK 9: Activer le test du lien affectif et des souvenirs affectifs
            def activer_test_pack9(dt):
                print("\n" + "=" * 80)
                print("DÉMARRAGE DU TEST PACK 9: LIEN AFFECTIF PROFOND ET SOUVENIRS AFFECTIFS")
                print("=" * 80)

                print("\nÉtat initial:")
                self._afficher_etat_lien()

                # Étape 1: Montée progressive du lien affectif
                print("\n" + "-" * 50)
                print("ÉTAPE 1: MONTÉE PROGRESSIVE DU LIEN AFFECTIF")
                print("-" * 50)

                actions_positives = [
                    ("Jeffrey, ton aide m'est précieuse", "joie", 0.7),
                    ("J'aime beaucoup ta façon de répondre", "gratitude", 0.8),
                    ("Tu as vraiment su comprendre ce que je voulais", "admiration", 0.6),
                    ("Je suis content de travailler avec toi", "joie", 0.7),
                    ("On forme une bonne équipe", "complicité", 0.8),
                    ("Merci pour ton soutien, c'est important pour moi", "gratitude", 0.9),
                ]

                # Simuler les interactions
                for i, (texte, emotion, intensite) in enumerate(actions_positives):
                    print(f"\nInteraction {i + 1}: '{texte}'")
                    self.jeffrey_core.analyser_et_adapter(texte)
                    self.jeffrey_core.enregistrer_moment_emotionnel(
                        description=f"L'utilisateur a dit: {texte}",
                        emotion=emotion,
                        intensite=intensite,
                    )
                    self._afficher_etat_lien()
                    # Afficher également la résonance affective
                    print(f"  Résonance affective: {self.jeffrey_core.resonance_affective:.2f}")
                    Clock.schedule_once(lambda dt: None, 0.5)  # Pause visuelle

                # Étape 2: Souvenir fort
                print("\n" + "-" * 50)
                print("ÉTAPE 2: ENREGISTREMENT D'UN SOUVENIR FORT")
                print("-" * 50)

                souvenir_fort = "J'ai vraiment eu l'impression que tu me comprenais parfaitement aujourd'hui, comme si tu pouvais lire dans mes pensées. C'était un moment spécial entre nous."
                print(f"\nSouvenir fort: '{souvenir_fort}'")

                # Enregistrer le souvenir fort
                self.jeffrey_core.enregistrer_moment_emotionnel(
                    description=souvenir_fort, emotion="admiration", intensite=0.95
                )

                print("\nÉtat après le souvenir fort:")
                self._afficher_etat_lien()
                self._afficher_souvenirs_recents()
                print(f"  Résonance affective: {self.jeffrey_core.resonance_affective:.2f}")

                # Étape 3: Simuler une absence
                print("\n" + "-" * 50)
                print("ÉTAPE 3: SIMULATION D'UNE PÉRIODE D'ABSENCE")
                print("-" * 50)

                # Manipuler la dernière interaction
                lien = self.jeffrey_core.gestionnaire_lien
                lien.derniere_interaction = lien.derniere_interaction - timedelta(days=3)
                print("\nSimulation d'une absence de 3 jours...")

                # Étape 4: Retour de l'utilisateur
                print("\n" + "-" * 50)
                print("ÉTAPE 4: RETOUR DE L'UTILISATEUR")
                print("-" * 50)

                message_retour = "Hey Jeffrey, je suis de retour ! Tu m'as manqué !"
                print(f"\nMessage de retour: '{message_retour}'")

                self.jeffrey_core.analyser_et_adapter(message_retour)

                print("\nÉtat après le retour:")
                self._afficher_etat_lien()
                print(f"  Résonance affective: {self.jeffrey_core.resonance_affective:.2f}")
                print(
                    f"  Description de la résonance: {self.jeffrey_core._decrire_resonance(self.jeffrey_core.resonance_affective)}"
                )

                # Étape 5: Test de blessure
                print("\n" + "-" * 50)
                print("ÉTAPE 5: TEST DE BLESSURE AFFECTIVE")
                print("-" * 50)

                message_blessure = "Je pense qu'on devrait faire une pause, tu m'agaces un peu là."
                print(f"\nSimulation d'une blessure: '{message_blessure}'")

                self.jeffrey_core.analyser_et_adapter(message_blessure)
                self.jeffrey_core.enregistrer_moment_emotionnel(
                    description=f"L'utilisateur a dit: {message_blessure}",
                    emotion="tristesse",
                    intensite=0.9,
                )

                print("\nÉtat après la blessure:")
                self._afficher_etat_lien()
                print(f"  Résonance affective: {self.jeffrey_core.resonance_affective:.2f}")
                print(f"  Blessure active: {'Oui' if self.jeffrey_core.blessure_active else 'Non'}")

                # Afficher les détails des blessures actives
                if hasattr(self.jeffrey_core.souvenirs_affectifs, "obtenir_blessures_actives"):
                    blessures = self.jeffrey_core.souvenirs_affectifs.obtenir_blessures_actives()
                    print(f"  Nombre de blessures actives: {len(blessures)}")
                    if blessures:
                        print("  Détails de la blessure principale:")
                        blessure = blessures[0]
                        print(f"    Description: {blessure['description'][:50]}...")
                        print(f"    Impact: {blessure['impact_emotionnel']:.2f}")
                        print(f"    Vibrance: {blessure['vibrance']:.2f}")

                # Étape 6: Test de chaleur du lien
                print("\n" + "-" * 50)
                print("ÉTAPE 6: TEST DE CALCUL DE CHALEUR DU LIEN")
                print("-" * 50)

                chaleur = self.jeffrey_core.gestionnaire_lien.calculer_chaleur_du_lien()
                print(f"Chaleur du lien: {chaleur:.2f} (entre 0.0 et 1.0)")

                # Afficher le rapport complet des souvenirs affectifs
                print("\n" + "-" * 50)
                print("ÉTAPE 7: RAPPORT COMPLET DES SOUVENIRS AFFECTIFS")
                print("-" * 50)

                rapport = self.jeffrey_core.obtenir_etat_souvenirs_affectifs()
                print(f"Disponible: {'Oui' if rapport['disponible'] else 'Non'}")
                if rapport["disponible"]:
                    print(f"Nombre total de souvenirs: {rapport['nombre_total_souvenirs']}")
                    print(f"Nombre de blessures actives: {rapport['nombre_blessures_actives']}")
                    print(f"Nombre de souvenirs positifs forts: {rapport['nombre_souvenirs_positifs_forts']}")
                    print(
                        f"Résonance affective: {rapport['resonance_affective']['niveau']:.2f} ({rapport['resonance_affective']['description']})"
                    )
                    print(f"Chaleur du lien: {rapport['chaleur_lien']:.2f}")

                    if "tendances" in rapport and rapport["tendances"]:
                        print("\nTendances récentes:")
                        periodes = ["derniere_semaine", "dernier_mois"]
                        for periode in periodes:
                            if periode in rapport["tendances"]:
                                tendance = rapport["tendances"][periode]
                                if tendance["nombre_souvenirs"] > 0:
                                    print(
                                        f"  {periode.replace('_', ' ').title()} ({tendance['nombre_souvenirs']} souvenirs):"
                                    )
                                    print(f"    Impact moyen: {tendance['impact_moyen']:.2f}")
                                    if "categories_principales" in tendance and tendance["categories_principales"]:
                                        categories = [
                                            f"{cat} ({count})" for cat, count in tendance["categories_principales"]
                                        ]
                                        print(f"    Catégories principales: {', '.join(categories)}")

                # Résumé du test
                print("\n" + "=" * 80)
                print("RÉSUMÉ DU TEST PACK 9")
                print("=" * 80)

                lien = self.jeffrey_core.gestionnaire_lien
                print(f"\nNiveau d'attachement final: {lien.niveau_attachement:.2f}")
                print(f"Niveau de confiance final: {lien.niveau_confiance:.2f}")
                print(f"Résonance affective finale: {lien.resonance_affective:.2f}")
                print(f"Chaleur du lien: {chaleur:.2f}")
                print(f"État du lien: {lien.etat_lien}")
                print(f"Blessure active: {'Oui' if lien.blessure_active else 'Non'}")

                # Informations sur les effets visuels (qui devraient être visibles)
                print("\nEffets visuels activés:")
                print(f"  Aura douce autour du cœur: {'Oui' if self.lien_heart_glow else 'Non'}")
                print(f"  Regard brillant: {'Oui' if self.lien_eye_shine else 'Non'}")
                print(f"  Regard triste: {'Oui' if self.lien_sad_eyes else 'Non'}")
                print(f"  Résonance affective active: {'Oui' if self.resonance_active else 'Non'}")
                print(f"  Overlay de blessure (opacité): {self.blessure_overlay_opacity:.2f}")
                print(f"  Larmes discrètes: {'Oui' if self.larmes_discretes else 'Non'}")

                print("\nTest terminé avec succès!")

            # Attendre 5 secondes avant d'activer le test Pack 9
            Clock.schedule_once(activer_test_pack9, 5.0)

            # Ensuite, test du mode intimité (Pack 5) après la fin du test Pack 9
            def test_intimite_mode(dt):
                print("\nPACK 5: Test du mode d'intimité")
                self.jeffrey_core.intimite_active = True
                self.jeffrey_core.personnalite_evolutive.intimite_active = True

                # Programmer une désactivation après 10 secondes
                def disable_intimite(dt):
                    print("PACK 5: Désactivation du mode d'intimité")
                    self.jeffrey_core.intimite_active = False
                    self.jeffrey_core.personnalite_evolutive.intimite_active = False

                Clock.schedule_once(disable_intimite, 10.0)

            # Attendre 30 secondes avant d'activer le mode test d'intimité (après le test Pack 9)
            Clock.schedule_once(test_intimite_mode, 30.0)

        # Retourner le widget racine
        return guidance_manager

    def _afficher_etat_lien(self):
        """
        Affiche l'état actuel du lien affectif pour le test Pack 9.
        """
        lien = self.jeffrey_core.gestionnaire_lien
        attachement = lien.niveau_attachement
        confiance = lien.niveau_confiance
        etat = lien.etat_lien
        resonance = lien.resonance_affective

        print(f"  Attachement: {attachement:.2f} | Confiance: {confiance:.2f} | Résonance: {resonance:.2f}")
        print(f"  État du lien: {etat}")

    def _afficher_souvenirs_recents(self, limite: int = 3):
        """
        Affiche les souvenirs affectifs récents pour le test Pack 9.

        Args:
            limite: Nombre maximum de souvenirs à afficher
        """
        souvenirs_affectifs = self.jeffrey_core.souvenirs_affectifs
        if not hasattr(souvenirs_affectifs, "obtenir_souvenirs_recents"):
            print("  Méthode obtenir_souvenirs_recents non disponible")
            return

        souvenirs = souvenirs_affectifs.obtenir_souvenirs_recents(jours=30)
        print(f"  Souvenirs récents ({min(limite, len(souvenirs))} sur {len(souvenirs)}):")

        for souvenir in souvenirs[:limite]:
            description = souvenir["description"]
            if len(description) > 70:
                description = description[:67] + "..."
            impact = souvenir["impact_emotionnel"]
            categorie = souvenir.get("categorie", "non catégorisé")

            # Gérer différents formats de date possible
            date_str = souvenir["date_creation"]
            try:
                # Essayer le format iso standard
                date = datetime.fromisoformat(date_str).strftime("%d/%m %H:%M")
            except ValueError:
                try:
                    # Essayer le format avec 'T' et 'Z'
                    if "T" in date_str and "Z" in date_str:
                        date_str = date_str.replace("Z", "+00:00")
                    date = datetime.fromisoformat(date_str.replace("T", " ")).strftime("%d/%m %H:%M")
                except ValueError:
                    # Utiliser la chaîne brute si impossible à parser
                    date = date_str[:16]

            print(f"  - [{date}] ({categorie}, impact: {impact:.2f}) {description}")
            # Afficher la vibrance si disponible
            if "vibrance" in souvenir:
                print(f"    Vibrance: {souvenir['vibrance']:.2f} | Rappels: {souvenir.get('rappels_count', 0)}")


# Si ce fichier est exécuté directement, lancer l'application de test
if __name__ == "__main__":
    JeffreyGuidanceApp().run()
