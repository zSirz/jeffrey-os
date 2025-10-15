#!/usr/bin/env python
"""
Module de détection de gestes intuitifs pour le système de guidance.
Permet de reconnaître des gestes naturels (glisser, pincer, cercle, etc.)
"""

import math
from collections import deque

import numpy as np
from kivy.clock import Clock
from kivy.graphics import Color, Line
from kivy.metrics import dp
from kivy.uix.widget import Widget


class GestureDetector(Widget):
    """
    Détecteur de gestes pour le système de guidance émotionnelle.
    Reconnaît des gestes naturels et fluides.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.touch_history = {}  # Historique des points de contact par touch_id
        self.gesture_callbacks = {
            "swipe_left": [],
            "swipe_right": [],
            "swipe_up": [],
            "swipe_down": [],
            "circle": [],
            "pinch_in": [],
            "pinch_out": [],
            "tap": [],
            "double_tap": [],
            "long_press": [],
        }
        self.debug_mode = False
        self.long_press_time = 0.5  # secondes
        self.long_press_distance = dp(10)  # distance max de mouvement pour un long press
        self.active_long_press = {}  # touch_id -> (clock_event, start_pos)

        # Pour dessiner les traces en debug
        self.trace_lines = {}  # touch_id -> liste de points
        self.draw_traces = False

    def on_touch_down(self, touch):
        """Démarre le suivi d'un nouveau toucher"""
        touch_id = touch.uid
        self.touch_history[touch_id] = deque(maxlen=20)  # Limite l'historique
        self.touch_history[touch_id].append((touch.x, touch.y, touch.time_start))

        # Gérer le long press
        def trigger_long_press(dt, tid=touch_id):
            if tid in self.touch_history:
                for callback in self.gesture_callbacks["long_press"]:
                    callback(self.touch_history[tid][-1])

        self.active_long_press[touch_id] = (
            Clock.schedule_once(trigger_long_press, self.long_press_time),
            (touch.x, touch.y),
        )

        # Initialiser traçage en mode debug
        if self.draw_traces:
            self.trace_lines[touch_id] = []
            self.trace_lines[touch_id].append((touch.x, touch.y))

        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        """Met à jour l'historique de mouvement"""
        touch_id = touch.uid
        if touch_id in self.touch_history:
            self.touch_history[touch_id].append((touch.x, touch.y, touch.time_start))

            # Vérifier si on doit annuler le long press (trop de mouvement)
            if touch_id in self.active_long_press:
                event, start_pos = self.active_long_press[touch_id]
                distance = math.sqrt((touch.x - start_pos[0]) ** 2 + (touch.y - start_pos[1]) ** 2)
                if distance > self.long_press_distance:
                    event.cancel()  # Annuler l'événement de long press
                    self.active_long_press.pop(touch_id)

            # Ajouter le point à la trace en mode debug
            if self.draw_traces and touch_id in self.trace_lines:
                self.trace_lines[touch_id].append((touch.x, touch.y))
                # Redessiner la trace
                self.canvas.clear()
                with self.canvas:
                    for tid, points in self.trace_lines.items():
                        Color(0.2, 0.8, 0.8, 0.7)
                        if len(points) > 1:
                            Line(points=sum([(x, y) for x, y in points], ()), width=1.5)

        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        """Analyse le geste à la fin du toucher"""
        touch_id = touch.uid
        if touch_id in self.touch_history and len(self.touch_history[touch_id]) > 1:
            # Annuler l'événement de long press s'il existe
            if touch_id in self.active_long_press:
                self.active_long_press[touch_id][0].cancel()
                self.active_long_press.pop(touch_id)

            # Analyser le geste
            history = list(self.touch_history[touch_id])

            # Vérifier tap vs swipe
            start_point = history[0]
            end_point = history[-1]

            # Calcul de déplacement total
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]
            distance = math.sqrt(dx * dx + dy * dy)

            # Calculer le temps écoulé
            duration = touch.time_end - start_point[2]

            # Tap simple (courte pression, peu de mouvement)
            if duration < 0.2 and distance < dp(20):
                for callback in self.gesture_callbacks["tap"]:
                    callback(end_point)

            # Détection de direction de swipe (si distance suffisante)
            elif distance > dp(40):
                # Déterminer la direction principale
                if abs(dx) > abs(dy):  # Horizontal
                    gesture = "swipe_right" if dx > 0 else "swipe_left"
                else:  # Vertical
                    gesture = "swipe_up" if dy > 0 else "swipe_down"

                # Appeler les callbacks
                for callback in self.gesture_callbacks[gesture]:
                    callback((start_point, end_point))

            # Détection de cercle
            if len(history) > 5:
                if self._is_circle_gesture(history):
                    for callback in self.gesture_callbacks["circle"]:
                        callback(history)

            # Nettoyage des traces en mode debug
            if self.draw_traces and touch_id in self.trace_lines:
                # Garder la trace pendant 1 seconde puis effacer
                def clear_trace(dt, tid=touch_id):
                    if tid in self.trace_lines:
                        self.trace_lines.pop(tid)
                        self.canvas.clear()

                Clock.schedule_once(clear_trace, 1.0)

        # Nettoyage
        if touch_id in self.touch_history:
            del self.touch_history[touch_id]

        return super().on_touch_up(touch)

    def _is_circle_gesture(self, points):
        """
        Détecte si les points forment approximativement un cercle.
        Utilise l'algorithme de RANSAC pour la détection de cercle.
        """
        # Minimum 8 points pour un cercle fiable
        if len(points) < 8:
            return False

        # Extraire juste les coordonnées x,y
        points_array = np.array([(p[0], p[1]) for p in points])

        # Calculer le centre approximatif
        center_x = np.mean(points_array[:, 0])
        center_y = np.mean(points_array[:, 1])

        # Calculer le rayon moyen et la variance
        distances = np.sqrt((points_array[:, 0] - center_x) ** 2 + (points_array[:, 1] - center_y) ** 2)

        radius = np.mean(distances)
        radius_std = np.std(distances)

        # Vérifier si la forme est fermée (le dernier point est proche du premier)
        first_last_distance = np.sqrt(
            (points_array[0, 0] - points_array[-1, 0]) ** 2 + (points_array[0, 1] - points_array[-1, 1]) ** 2
        )

        # Calcul d'angles pour vérifier que la trajectoire couvre un cercle complet
        angles = np.arctan2(points_array[:, 1] - center_y, points_array[:, 0] - center_x)
        angle_range = np.ptp(angles)  # peak to peak, différence max-min

        # Critères pour un cercle:
        # 1. Rayon relativement constant (faible écart-type)
        # 2. Points couvrant au moins 270 degrés
        # 3. Premier et dernier points relativement proches
        return (
            radius_std / radius < 0.3
            and angle_range  # Écart-type du rayon < 30% du rayon
            > 4.71  # Au moins 270 degrés (en radians)
            and first_last_distance < radius * 0.8  # Premier et dernier points proches
        )

    def register_gesture_callback(self, gesture_type, callback):
        """
        Enregistre une fonction de rappel pour un type de geste spécifique.

        Args:
            gesture_type: Type de geste ('swipe_left', 'circle', etc.)
            callback: Fonction à appeler quand le geste est détecté
        """
        if gesture_type in self.gesture_callbacks:
            self.gesture_callbacks[gesture_type].append(callback)

    def unregister_gesture_callback(self, gesture_type, callback):
        """Supprime une fonction de rappel pour un type de geste"""
        if gesture_type in self.gesture_callbacks and callback in self.gesture_callbacks[gesture_type]:
            self.gesture_callbacks[gesture_type].remove(callback)

    def enable_debug_traces(self, enable=True):
        """Active ou désactive l'affichage des traces de gestes"""
        self.draw_traces = enable
        if not enable:
            self.trace_lines.clear()
            self.canvas.clear()


class MultiTouchGestureDetector(GestureDetector):
    """
    Version avancée du détecteur de gestes qui gère également les gestes
    multi-touch comme le pincement et la rotation.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.multitouch_state = {}  # Pour suivre l'état des gestes multi-touch

    def on_touch_down(self, touch):
        """Gère le début d'un touch en mode multi-touch"""
        result = super().on_touch_down(touch)

        # Mettre à jour l'état multi-touch
        active_touches = list(self.touch_history.keys())
        if len(active_touches) == 2:
            # Initialiser le suivi de pincement/rotation
            touch1_id, touch2_id = active_touches
            if touch1_id in self.touch_history and touch2_id in self.touch_history:
                touch1_pos = self.touch_history[touch1_id][-1][:2]  # x,y
                touch2_pos = self.touch_history[touch2_id][-1][:2]  # x,y

                # Calculer distance initiale pour pincement
                distance = math.sqrt((touch1_pos[0] - touch2_pos[0]) ** 2 + (touch1_pos[1] - touch2_pos[1]) ** 2)

                # Calculer angle initial pour rotation
                angle = math.atan2(touch2_pos[1] - touch1_pos[1], touch2_pos[0] - touch1_pos[0])

                self.multitouch_state = {
                    "touch_ids": (touch1_id, touch2_id),
                    "start_distance": distance,
                    "current_distance": distance,
                    "start_angle": angle,
                    "current_angle": angle,
                }

        return result

    def on_touch_move(self, touch):
        """Met à jour l'état multi-touch lors du mouvement"""
        result = super().on_touch_move(touch)

        # Mettre à jour gestes multi-touch
        if "touch_ids" in self.multitouch_state:
            touch1_id, touch2_id = self.multitouch_state["touch_ids"]

            if (
                touch1_id in self.touch_history
                and touch2_id in self.touch_history
                and len(self.touch_history[touch1_id]) > 0
                and len(self.touch_history[touch2_id]) > 0
            ):
                touch1_pos = self.touch_history[touch1_id][-1][:2]  # x,y
                touch2_pos = self.touch_history[touch2_id][-1][:2]  # x,y

                # Mettre à jour distance pour pincement
                new_distance = math.sqrt((touch1_pos[0] - touch2_pos[0]) ** 2 + (touch1_pos[1] - touch2_pos[1]) ** 2)

                old_distance = self.multitouch_state["current_distance"]
                self.multitouch_state["current_distance"] = new_distance

                # Détecter pincement
                distance_change = new_distance - old_distance
                if abs(distance_change) > dp(10):
                    gesture = "pinch_out" if distance_change > 0 else "pinch_in"
                    event_data = {
                        "center": (
                            (touch1_pos[0] + touch2_pos[0]) / 2,
                            (touch1_pos[1] + touch2_pos[1]) / 2,
                        ),
                        "scale_factor": new_distance / self.multitouch_state["start_distance"],
                    }
                    for callback in self.gesture_callbacks[gesture]:
                        callback(event_data)

        return result

    def on_touch_up(self, touch):
        """Finalise les gestes multi-touch quand un touch se termine"""
        touch_id = touch.uid

        # Vérifier si ce touch faisait partie d'un geste multi-touch
        if "touch_ids" in self.multitouch_state and touch_id in self.multitouch_state["touch_ids"]:
            # Réinitialiser l'état multi-touch
            self.multitouch_state = {}

        return super().on_touch_up(touch)
