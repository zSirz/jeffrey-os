#!/usr/bin/env python

"""
Module des effets visuels pour le visage de Jeffrey.
Contient les effets liés aux émotions, sensations et interactions.

Ce module permet de décharger le fichier energy_face.py en regroupant
tous les effets visuels et leurs méthodes de contrôle dans un fichier dédié.
"""

import math
import random

from kivy.clock import Clock
from kivy.graphics import Color, Ellipse, Line


class FaceEffects:
    """
    Classe regroupant tous les effets visuels et sensoriels pour le visage.
    Fonctionne comme une extension du widget EnergyFaceWidget.
    """

    def __init__(self, parent_widget):
        """
        Initialise les effets visuels.

        Args:
            parent_widget: Référence au widget EnergyFaceWidget parent
        """
        self.widget = parent_widget

        # Variables d'état pour les effets
        self._effect_states = {}
        self._active_effects = set()
        self._effect_timers = {}

        # Mémoire des effets sensoriels
        self._memory_timers = {}

        # PACK 5 : Variables spécifiques à la mémoire sensorielle
        self._touch_memory = {}  # Zone -> dernier contact
        self._active_sensory_responses = {}  # Zone -> réponse active

        # PACK 7 : Variables spécifiques aux zones corporelles
        self._active_zone_effects = {}  # Zone -> effet actif
        self._zone_memory = {}  # Mémoire des zones touchées

        # PACK 11 : Variables pour les effets liés aux relations
        self._eye_sparkle_active = False
        self._eye_sparkle_intensity = 0.0
        self._eye_sparkle_timer = None

    #
    # ===== PACK 4 : EFFETS DE BASE =====
    #

    # EFFET PACK 4 : Vibration émotionnelle
    def start_emotional_vibration(self):
        """
        Démarre l'effet de vibration émotionnelle subtile pour
        colère, tristesse et stress.
        """
        self._set_effect_state("vibration_active", True)
        self._set_effect_state("vibration_phase", 0.0)
        self._set_effect_state("vibration_offset_x", 0.0)

        # Arrêter automatiquement après 10 secondes
        self._schedule_effect_timer("vibration_timer", self.stop_emotional_vibration, 10.0)

    def stop_emotional_vibration(self):
        """Arrête la vibration émotionnelle et réinitialise les valeurs."""
        self._set_effect_state("vibration_active", False)
        self._set_effect_state("vibration_offset_x", 0.0)
        self._clear_effect_timer("vibration_timer")

    # EFFET PACK 4 : Micro-tensions musculaires
    def start_muscle_tensions(self):
        """
        Démarre l'effet de micro-tensions musculaires pour stress et peur.
        Crée des spasmes aléatoires sur les sourcils et la mâchoire.
        """
        self._set_effect_state("muscle_tensions_active", True)
        self._set_effect_state("eyebrow_spasm_offset", 0)
        self._set_effect_state("jaw_spasm_offset", 0)

        # Fonction pour les spasmes aléatoires
        def update_muscle_spasms(dt):
            if not self._get_effect_state("muscle_tensions_active"):
                return False

            # Valeurs aléatoires pour les spasmes
            self._set_effect_state("eyebrow_spasm_offset", random.uniform(-1.0, 1.0))
            self._set_effect_state("jaw_spasm_offset", random.uniform(-1.0, 1.0))

            # Reprogrammer avec intervalle aléatoire
            next_interval = random.uniform(0.05, 0.15)
            Clock.schedule_once(update_muscle_spasms, next_interval)
            return False

        # Démarrer les spasmes
        update_muscle_spasms(0)

        # Arrêter automatiquement après 8 secondes
        self._schedule_effect_timer("muscle_tension_timer", self.stop_muscle_tensions, 8.0)

    def stop_muscle_tensions(self):
        """Arrête les micro-tensions musculaires et réinitialise les valeurs."""
        self._set_effect_state("muscle_tensions_active", False)
        self._set_effect_state("eyebrow_spasm_offset", 0)
        self._set_effect_state("jaw_spasm_offset", 0)
        self._clear_effect_timer("muscle_tension_timer")

    # EFFET PACK 4 : Goutte de sueur ou larme rapide
    def trigger_fast_tear(self):
        """
        Crée une goutte de sueur ou larme qui descend rapidement
        du côté gauche du visage en cas de peur.
        """
        # Ajouter une nouvelle larme
        tear = {
            "x": self.widget.center_x - 45 + random.uniform(-3, 3),
            "y": self.widget.center_y + 20,
            "size": (5, 10),
            "speed": 4.0,
            "alpha": 0.15,
            "age": 0.0,
            "max_age": 1.0,
        }

        # Ajouter à la liste des larmes actives
        tears = self._get_effect_state("fast_tears", [])
        tears.append(tear)
        self._set_effect_state("fast_tears", tears)

    def update_fast_tears(self, dt):
        """Met à jour les larmes rapides, les fait descendre et disparaître."""
        tears = self._get_effect_state("fast_tears", [])
        if not tears:
            return

        new_tears = []

        for tear in tears:
            # Mettre à jour l'âge
            tear["age"] += dt

            # Faire descendre la larme
            tear["y"] -= tear["speed"]

            # Conserver la larme si elle n'a pas disparu
            if tear["age"] < tear["max_age"] and tear["y"] > self.widget.center_y - 100:
                new_tears.append(tear)

        # Mettre à jour la liste
        self._set_effect_state("fast_tears", new_tears)

    # EFFET PACK 4 : Nébuleuse mentale
    def start_mental_nebula(self):
        """
        Démarre l'effet de nébuleuse mentale pour les émotions fatigue et vide.
        Crée une aura flottante gris-lavande diffuse autour de la tête.
        """
        self._set_effect_state("mental_nebula_active", True)
        self._set_effect_state("nebula_phase", 0.0)

        # Arrêter automatiquement après 15 secondes
        self._schedule_effect_timer("nebula_timer", self.stop_mental_nebula, 15.0)

    def stop_mental_nebula(self):
        """Arrête l'effet de nébuleuse mentale."""
        self._set_effect_state("mental_nebula_active", False)
        self._clear_effect_timer("nebula_timer")

    #
    # ===== PACK 5A : PLAISIR AFFECTIF =====
    #

    # EFFET PACK 5A : Halo de plaisir affectif
    def start_pleasure_halo(self):
        """
        Démarre l'effet de halo rose pulsant autour du visage,
        indiquant un état de plaisir affectif élevé.
        """
        self._set_effect_state("pleasure_halo_active", True)
        self._set_effect_state("pleasure_halo_phase", 0.0)

    def stop_pleasure_halo(self):
        """Arrête l'effet de halo rose pulsant."""
        self._set_effect_state("pleasure_halo_active", False)

    #
    # ===== PACK 5B : RÉSONANCE SENSORIELLE =====
    #

    # EFFET PACK 5B : Résonance sensorielle intime
    def start_sensual_resonance(self):
        """
        Démarre l'effet de résonance sensorielle subtile,
        caractérisé par des pulsations dans les joues,
        dilatation des pupilles, etc.

        Uniquement en mode adulte, contexte privé.
        """
        if self.widget.developmental_mode != "adulte" or self.widget.context_mode != "private":
            return

        self._set_effect_state("sensual_resonance_active", True)
        self._set_effect_state("sensual_phase", 0.0)

    def stop_sensual_resonance(self):
        """Arrête l'effet de résonance sensorielle."""
        self._set_effect_state("sensual_resonance_active", False)

    # EFFET PACK 5B : Déclenchement d'une mini montée de chaleur
    def trigger_warmth_surge(self):
        """
        Déclenche une mini montée de chaleur (rougissement, paupières mi-closes)
        quand plaisir > 0.9 et in_love = True en mode adulte.
        """
        if self.widget.developmental_mode != "adulte" or self.widget.context_mode != "private":
            return

        # Augmenter l'intensité du rougissement
        self.widget.blushing_intensity = min(1.0, self.widget.blushing_intensity + 0.3)

        # Effet temporaire de paupières mi-closes
        def temp_eyelid_effect(dt):
            # Revenir progressivement à l'état normal
            self.widget.eyelid_openness = min(1.0, self.widget.eyelid_openness + 0.1)

            if self.widget.eyelid_openness >= 1.0:
                return False
            return True

        # Réduire l'ouverture des paupières
        self.widget.eyelid_openness = 0.6

        # Revenir progressivement à la normale
        Clock.schedule_interval(temp_eyelid_effect, 0.2)

    # EFFET PACK 5 : État d'intimité
    def start_intimite_effect(self):
        """
        Démarre les effets visuels liés à l'intimité.
        Adapte les effets selon le stade de développement.
        """
        self._set_effect_state("intimite_effect_active", True)
        self._set_effect_state("intimite_phase", 0.0)

        # Effets différents selon le stade de développement
        if self.widget.stade_developpement == "enfant":
            # Pour enfant: pas d'effet d'intimité, juste innocence
            self._set_effect_state("intimite_effect_active", False)

        elif self.widget.stade_developpement == "adolescent":
            # Pour adolescent: timidité, rougissement léger
            self.widget.blushing_intensity = min(0.5, self.widget.blushing_intensity + 0.3)

            # Timer pour l'animation des effets de timidité
            self._schedule_effect_timer("intimite_timer", self.stop_intimite_effect, 5.0)

        elif self.widget.stade_developpement == "adulte" and self.widget.context_mode == "private":
            # Pour adulte: effets plus prononcés, aura subtile
            # Le niveau dépend de l'inverse de la pudeur (1 - pudeur)
            intensity = max(0.0, 1.0 - self.widget.pudeur)

            # Fonction d'animation pour les effets d'intimité adulte
            def animate_intimite(dt):
                if not self._get_effect_state("intimite_effect_active"):
                    return False

                phase = self._get_effect_state("intimite_phase", 0.0)
                self._set_effect_state("intimite_phase", phase + dt * 0.2)

                # Variation du rougissement
                if random.random() < 0.05:
                    self.widget.blushing_intensity = min(0.7, self.widget.blushing_intensity + 0.05)
                else:
                    self.widget.blushing_intensity = max(0.0, self.widget.blushing_intensity - 0.01)

                return True

            # Démarrer l'animation
            Clock.schedule_interval(animate_intimite, 1 / 15.0)

            # Timer pour l'arrêt automatique (entre 8 et 15 secondes, aléatoire)
            self._schedule_effect_timer("intimite_timer", self.stop_intimite_effect, 8.0 + 7.0 * random.random())

    def stop_intimite_effect(self):
        """Arrête les effets visuels liés à l'intimité."""
        if not self._get_effect_state("intimite_effect_active"):
            return

        self._set_effect_state("intimite_effect_active", False)
        self._clear_effect_timer("intimite_timer")

        # Réinitialiser progressivement les effets
        def fade_out_effects(dt):
            # Réduire l'intensité du rougissement
            self.widget.blushing_intensity = max(0.0, self.widget.blushing_intensity - 0.05)

            # Continuer tant que les effets ne sont pas estompés
            if self.widget.blushing_intensity > 0.01:
                return True
            return False

        # Démarrer l'estompage
        Clock.schedule_interval(fade_out_effects, 1 / 10.0)

    # EFFET PACK 5A/5B : État de vide émotionnel doux
    def trigger_emotional_emptiness(self):
        """
        Déclenche un état de vide émotionnel doux après une longue période
        sans affection, pour simuler un manque affectif naturel.
        """
        # Transition vers l'émotion "vide" avec intensité modérée
        # Affecte les aspects visuels
        self.widget.pleasure_level = max(0, self.widget.pleasure_level - 0.2)

        # Si le halo est actif, le désactiver doucement
        if self._get_effect_state("pleasure_halo_active"):
            self.stop_pleasure_halo()

    #
    # ===== PACK 7 : MÉMOIRE SENSORIELLE =====
    #

    # EFFET PACK 7 : Frisson localisé
    def trigger_localized_shiver(self, zone, intensity=0.5):
        """
        Déclenche un frisson localisé sur une zone spécifique.

        Args:
            zone: Zone où le frisson se produit
            intensity: Intensité du frisson (0.0 à 1.0)
        """
        # Déjà un frisson actif sur cette zone?
        if zone in self._active_zone_effects and "shiver" in self._active_zone_effects[zone]:
            return

        # Déterminer les paramètres du frisson selon la zone
        zone_params = self._get_zone_parameters(zone)
        if not zone_params:
            return

        center_x, center_y = zone_params["center"]
        radius = zone_params["radius"]

        # Créer l'effet de frisson
        shiver_effect = {
            "type": "shiver",
            "center": (center_x, center_y),
            "radius": radius,
            "intensity": intensity,
            "phase": 0.0,
            "duration": 1.5 + intensity * 2.0,  # 1.5-3.5 secondes selon intensité
            "start_time": Clock.get_time(),
        }

        # Enregistrer l'effet
        if zone not in self._active_zone_effects:
            self._active_zone_effects[zone] = {}
        self._active_zone_effects[zone]["shiver"] = shiver_effect

        # Programmer l'arrêt automatique
        def stop_shiver(dt):
            if zone in self._active_zone_effects and "shiver" in self._active_zone_effects[zone]:
                del self._active_zone_effects[zone]["shiver"]

        Clock.schedule_once(stop_shiver, shiver_effect["duration"])

    # EFFET PACK 7 : Aura sensorielle localisée
    def trigger_sensory_aura(self, zone, emotion="neutral", intensity=0.5):
        """
        Déclenche une aura sensorielle localisée sur une zone spécifique.

        Args:
            zone: Zone où l'aura se produit
            emotion: Émotion associée à l'aura
            intensity: Intensité de l'aura (0.0 à 1.0)
        """
        # Déjà une aura active sur cette zone?
        if zone in self._active_zone_effects and "aura" in self._active_zone_effects[zone]:
            return

        # Déterminer les paramètres de l'aura selon la zone
        zone_params = self._get_zone_parameters(zone)
        if not zone_params:
            return

        center_x, center_y = zone_params["center"]
        radius = zone_params["radius"] * 1.5  # Aura plus large

        # Déterminer la couleur selon l'émotion
        color = self._get_emotion_color(emotion, intensity)

        # Créer l'effet d'aura
        aura_effect = {
            "type": "aura",
            "center": (center_x, center_y),
            "radius": radius,
            "color": color,
            "intensity": intensity,
            "phase": 0.0,
            "duration": 3.0 + intensity * 4.0,  # 3-7 secondes selon intensité
            "start_time": Clock.get_time(),
        }

        # Enregistrer l'effet
        if zone not in self._active_zone_effects:
            self._active_zone_effects[zone] = {}
        self._active_zone_effects[zone]["aura"] = aura_effect

        # Programmer l'arrêt automatique
        def stop_aura(dt):
            if zone in self._active_zone_effects and "aura" in self._active_zone_effects[zone]:
                del self._active_zone_effects[zone]["aura"]

        Clock.schedule_once(stop_aura, aura_effect["duration"])

    # EFFET PACK 7 : Gêne subtile
    def trigger_subtle_discomfort(self, zone, intensity=0.3):
        """
        Déclenche une gêne subtile sur une zone spécifique.

        Args:
            zone: Zone où la gêne se produit
            intensity: Intensité de la gêne (0.0 à 1.0)
        """
        # Déjà une gêne active sur cette zone?
        if zone in self._active_zone_effects and "discomfort" in self._active_zone_effects[zone]:
            return

        # Déterminer les paramètres de la gêne selon la zone
        zone_params = self._get_zone_parameters(zone)
        if not zone_params:
            return

        center_x, center_y = zone_params["center"]
        radius = zone_params["radius"] * 0.8  # Zone plus petite

        # Créer l'effet de gêne
        discomfort_effect = {
            "type": "discomfort",
            "center": (center_x, center_y),
            "radius": radius,
            "intensity": intensity,
            "phase": 0.0,
            "duration": 2.0 + intensity * 3.0,  # 2-5 secondes selon intensité
            "start_time": Clock.get_time(),
        }

        # Enregistrer l'effet
        if zone not in self._active_zone_effects:
            self._active_zone_effects[zone] = {}
        self._active_zone_effects[zone]["discomfort"] = discomfort_effect

        # Programmer l'arrêt automatique
        def stop_discomfort(dt):
            if zone in self._active_zone_effects and "discomfort" in self._active_zone_effects[zone]:
                del self._active_zone_effects[zone]["discomfort"]

        Clock.schedule_once(stop_discomfort, discomfort_effect["duration"])

    # EFFET PACK 7 : Mise à jour des effets de zone
    def update_zone_effects(self, dt):
        """Met à jour les effets actifs sur les différentes zones corporelles."""
        if not self._active_zone_effects:
            return

        # Mettre à jour chaque zone
        for zone, effects in list(self._active_zone_effects.items()):
            if not effects:
                del self._active_zone_effects[zone]
                continue

            # Mettre à jour chaque effet dans la zone
            for effect_type, effect in list(effects.items()):
                current_time = Clock.get_time()
                effect["phase"] += dt * 2.0  # Fréquence commune pour tous les effets

                # Vérifier si l'effet doit se terminer
                elapsed = current_time - effect["start_time"]
                if elapsed > effect["duration"]:
                    del effects[effect_type]

        # Forcer le rafraîchissement de l'affichage
        self.widget.update_canvas()

    # EFFET PACK 7 : Traitement du contact sur une zone
    def process_zone_touch(self, zone, touch_type="caresse", intensity=0.5):
        """
        Traite un contact sur une zone spécifique et déclenche
        les effets associés en fonction de la mémoire sensorielle.

        Args:
            zone: Zone touchée
            touch_type: Type de contact
            intensity: Intensité du contact (0.0 à 1.0)
        """
        # Enregistrer le contact dans la mémoire
        self._record_zone_touch(zone, touch_type, intensity)

        # Récupérer la sensibilité actuelle de la zone
        sensitivity = self._get_zone_sensitivity(zone)

        # Déterminer l'effet à déclencher en fonction du type de contact et de la sensibilité
        if touch_type in ["caresse", "effleurement", "frôlement"] and sensitivity > 0.5:
            # Contact doux sur zone sensible
            self.trigger_localized_shiver(zone, intensity * sensitivity)

            # Potentiellement ajouter une aura pour les zones très sensibles
            if sensitivity > 0.7 and intensity > 0.6:
                self.trigger_sensory_aura(zone, "plaisir", intensity * 0.8)

        elif touch_type in ["appui", "tape", "pincement"]:
            # Contact plus intense
            if sensitivity > 0.7:
                # Zone très sensible: discomfort potentiel
                self.trigger_subtle_discomfort(zone, intensity * sensitivity)
            else:
                # Zone moins sensible: juste un petit frisson
                self.trigger_localized_shiver(zone, intensity * 0.5)

        elif touch_type == "bisou" and sensitivity > 0.3:
            # Bisou: toujours aura, intensité variable selon sensibilité
            self.trigger_sensory_aura(zone, "affection", intensity * sensitivity)

        # En plus, pour toute zone suffisamment touchée, potentiel de rougissement
        touch_count = self._get_zone_touch_count(zone)
        if touch_count > 3 and zone in ["joue_gauche", "joue_droite", "visage"]:
            self.widget.blushing_intensity = min(0.6, self.widget.blushing_intensity + 0.05 * intensity)

    #
    # ===== MÉTHODES UTILITAIRES =====
    #

    def _get_effect_state(self, key, default=None):
        """Récupère l'état d'un effet."""
        # D'abord essayer de lire depuis le widget parent
        # (pour compatibilité avec le code existant)
        attr_name = f"_{key}"
        if hasattr(self.widget, attr_name):
            return getattr(self.widget, attr_name)

        # Sinon, lire depuis l'état interne
        return self._effect_states.get(key, default)

    def _set_effect_state(self, key, value):
        """Définit l'état d'un effet."""
        # D'abord essayer de définir dans le widget parent
        # (pour compatibilité avec le code existant)
        attr_name = f"_{key}"
        if hasattr(self.widget, attr_name):
            setattr(self.widget, attr_name, value)
            return

        # Sinon, définir dans l'état interne
        self._effect_states[key] = value

        # Marquer l'effet comme actif si nécessaire
        if key.endswith("_active") and value:
            base_name = key[:-7]  # Enlever "_active"
            self._active_effects.add(base_name)
        elif key.endswith("_active") and not value:
            base_name = key[:-7]  # Enlever "_active"
            if base_name in self._active_effects:
                self._active_effects.remove(base_name)

    def _schedule_effect_timer(self, timer_name, callback, timeout):
        """Programme un timer pour un effet."""
        # Annuler le timer existant si nécessaire
        self._clear_effect_timer(timer_name)

        # Créer le nouveau timer
        timer = Clock.schedule_once(lambda dt: callback(), timeout)
        self._effect_timers[timer_name] = timer

    def _clear_effect_timer(self, timer_name):
        """Annule un timer d'effet."""
        if timer_name in self._effect_timers:
            Clock.unschedule(self._effect_timers[timer_name])
            del self._effect_timers[timer_name]

    def _record_zone_touch(self, zone, touch_type, intensity):
        """
        Enregistre un contact sur une zone dans la mémoire locale.
        """
        if zone not in self._zone_memory:
            self._zone_memory[zone] = {
                "last_touch": None,
                "touch_count": 0,
                "touch_types": {},
                "average_intensity": 0.0,
            }

        memory = self._zone_memory[zone]
        memory["last_touch"] = Clock.get_time()
        memory["touch_count"] += 1

        # Mettre à jour le compteur par type
        memory["touch_types"][touch_type] = memory["touch_types"].get(touch_type, 0) + 1

        # Mettre à jour l'intensité moyenne (moyenne mobile)
        memory["average_intensity"] = (memory["average_intensity"] * (memory["touch_count"] - 1) + intensity) / memory[
            "touch_count"
        ]

    def _get_zone_touch_count(self, zone):
        """Récupère le nombre de contacts sur une zone."""
        if zone not in self._zone_memory:
            return 0
        return self._zone_memory[zone]["touch_count"]

    def _get_zone_sensitivity(self, zone):
        """
        Calcule la sensibilité d'une zone en fonction de l'historique des contacts.

        C'est une version simplifiée pour les tests, à remplacer par l'intégration
        avec la classe MémoireSensorielle.
        """
        # Sensibilité de base par zone
        base_sensitivity = {
            "joue_gauche": 0.7,
            "joue_droite": 0.7,
            "front": 0.5,
            "menton": 0.4,
            "nez": 0.6,
            "lèvres": 0.9,
            "visage": 0.6,
            "tête": 0.5,
        }.get(zone, 0.5)

        # Ajuster selon l'historique des contacts
        if zone in self._zone_memory:
            memory = self._zone_memory[zone]

            # La sensibilité augmente avec le nombre de contacts (jusqu'à un certain point)
            touch_factor = min(1.0, memory["touch_count"] / 10.0)

            # L'intensité moyenne des contacts précédents influence aussi
            intensity_factor = memory["average_intensity"]

            # Calculer la sensibilité ajustée
            adjusted = base_sensitivity * (1.0 + touch_factor * 0.3) * (1.0 + intensity_factor * 0.2)
            return min(1.0, adjusted)

        return base_sensitivity

    def _get_zone_parameters(self, zone):
        """
        Récupère les paramètres de position et taille pour une zone.

        Returns:
            Dict avec center (x, y) et radius, ou None si zone inconnue
        """
        # Récupérer les coordonnées centrales du visage
        center_x = self.widget.center_x
        center_y = self.widget.center_y

        # Paramètres par zone
        zone_map = {
            "joue_gauche": {"center": (center_x - 40, center_y - 10), "radius": 20},
            "joue_droite": {"center": (center_x + 40, center_y - 10), "radius": 20},
            "front": {"center": (center_x, center_y + 40), "radius": 25},
            "menton": {"center": (center_x, center_y - 45), "radius": 15},
            "nez": {"center": (center_x, center_y + 5), "radius": 12},
            "lèvres": {"center": (center_x, center_y - 25), "radius": 15},
            "visage": {"center": (center_x, center_y), "radius": 60},
            "tête": {"center": (center_x, center_y + 20), "radius": 70},
        }

        return zone_map.get(zone)

    def _get_emotion_color(self, emotion, intensity):
        """
        Retourne une couleur RGBA correspondant à une émotion.

        Args:
            emotion: Nom de l'émotion
            intensity: Intensité (0.0 à 1.0)

        Returns:
            Tuple RGBA (r, g, b, a)
        """
        # Palette de couleurs par émotion
        palette = {
            "joie": (1.0, 0.9, 0.4, 0.2),
            "tristesse": (0.4, 0.6, 0.9, 0.2),
            "colère": (0.9, 0.3, 0.3, 0.2),
            "peur": (0.6, 0.4, 0.8, 0.2),
            "surprise": (0.8, 0.8, 0.4, 0.2),
            "dégoût": (0.5, 0.8, 0.5, 0.2),
            "neutre": (0.7, 0.7, 0.7, 0.15),
            "plaisir": (0.9, 0.5, 0.7, 0.25),
            "curiosité": (0.6, 0.8, 0.9, 0.2),
            "affection": (1.0, 0.6, 0.8, 0.3),
            "sérénité": (0.7, 0.9, 1.0, 0.2),
        }

        # Couleur par défaut si émotion inconnue
        base_color = palette.get(emotion, (0.8, 0.8, 0.8, 0.2))

        # Ajuster l'alpha selon l'intensité
        r, g, b, a = base_color
        adjusted_alpha = a * intensity

        return (r, g, b, adjusted_alpha)

    def trigger_eye_sparkle(self, duration: float = 3.0, intensity: float = 0.8):
        """
        PACK 11: Déclenche un effet d'étincelles dans les yeux,
        généralement utilisé pour exprimer l'affection ou la joie.

        Args:
            duration: Durée de l'effet en secondes
            intensity: Intensité de l'effet (0.0 à 1.0)
        """
        # Activer l'effet
        self._eye_sparkle_active = True
        self._eye_sparkle_intensity = min(1.0, max(0.0, intensity))

        # Annuler tout timer existant
        if self._eye_sparkle_timer:
            self._eye_sparkle_timer.cancel()

        # Programmer la fin de l'effet
        def end_eye_sparkle(dt):
            # Diminution progressive
            def fade_out(dt):
                self._eye_sparkle_intensity = max(0.0, self._eye_sparkle_intensity - 0.1)
                if self._eye_sparkle_intensity <= 0.0:
                    self._eye_sparkle_active = False
                    return False
                return True

            Clock.schedule_interval(fade_out, 0.1)

        # Programmer la fin de l'effet
        self._eye_sparkle_timer = Clock.schedule_once(end_eye_sparkle, duration)

    def _trigger_tactile_warmth(self, x: float, y: float, warmth_factor: float = 1.0):
        """
        Déclenche un effet de chaleur tactile à l'endroit touché.

        Args:
            x: Coordonnée X du toucher
            y: Coordonnée Y du toucher
            warmth_factor: Facteur de chaleur (PACK 11)
        """
        # Créer un effet de chaleur
        # Ce sera simplement un cercle radial dégradé
        warmth = {
            "x": x,
            "y": y,
            "radius": 40.0 * warmth_factor,  # PACK 11: Moduler le rayon
            "color": (1.0, 0.9, 0.7, 0.2 * warmth_factor),  # PACK 11: Moduler l'opacité
            "duration": 0.8 * warmth_factor,  # PACK 11: Moduler la durée
            "age": 0.0,
            "max_age": 1.0,
            "expansion_rate": 1.2 * warmth_factor,  # PACK 11: Moduler la vitesse d'expansion
        }

        # Ajouter à la liste des effets de chaleur actifs
        warmth_effects = self._get_effect_state("tactile_warmth", [])
        warmth_effects.append(warmth)
        self._set_effect_state("tactile_warmth", warmth_effects)

    def add_eye_reflection(self, intensity: float = 0.7):
        """
        PACK 18: Ajoute un effet de brillance/humidité dans les yeux.
        Utile pour les émotions intenses, tristesse, émotion forte, etc.

        Args:
            intensity: Intensité de l'effet (0.0 à 1.0)
        """
        # Normaliser l'intensité
        intensity = min(1.0, max(0.0, intensity))

        # Activer l'effet
        self._set_effect_state("eye_reflection_active", True)
        self._set_effect_state("eye_reflection_intensity", intensity)

        # Programmer l'arrêt automatique après un délai proportionnel à l'intensité
        duration = 2.0 + intensity * 5.0  # Entre 2 et 7 secondes

        def fade_out_reflection(dt):
            # Diminution progressive
            def fade_step(dt):
                current = self._get_effect_state("eye_reflection_intensity", 0.0)
                current = max(0.0, current - 0.05)
                self._set_effect_state("eye_reflection_intensity", current)

                # Désactiver si complètement estompé
                if current <= 0.01:
                    self._set_effect_state("eye_reflection_active", False)
                    return False
                return True

            # Lancer la diminution progressive
            Clock.schedule_interval(fade_step, 0.1)

        # Planifier la fin de l'effet
        self._schedule_effect_timer("eye_reflection_timer", fade_out_reflection, duration)

    def add_blush(self, intensity: float = 0.5, left: bool = True, right: bool = True):
        """
        PACK 18: Ajoute un effet de rougissement sur les joues.
        Utile pour la gêne, l'excitation, la timidité, etc.

        Args:
            intensity: Intensité du rougissement (0.0 à 1.0)
            left: Si True, ajoute le rougissement sur la joue gauche
            right: Si True, ajoute le rougissement sur la joue droite
        """
        # Normaliser l'intensité
        intensity = min(1.0, max(0.0, intensity))

        # Définir l'intensité des joues
        if left:
            self._set_effect_state("left_cheek_blush", intensity)
        if right:
            self._set_effect_state("right_cheek_blush", intensity)

        # Marquer l'effet comme actif
        self._set_effect_state("blush_active", True)

        # Programmer l'estompage progressif
        duration = 2.0 + intensity * 3.0  # Entre 2 et 5 secondes

        def fade_out_blush(dt):
            # Diminution progressive
            def fade_step(dt):
                # Récupérer l'intensité actuelle
                left_intensity = self._get_effect_state("left_cheek_blush", 0.0)
                right_intensity = self._get_effect_state("right_cheek_blush", 0.0)

                # Diminuer progressivement
                if left:
                    left_intensity = max(0.0, left_intensity - 0.05)
                    self._set_effect_state("left_cheek_blush", left_intensity)

                if right:
                    right_intensity = max(0.0, right_intensity - 0.05)
                    self._set_effect_state("right_cheek_blush", right_intensity)

                # Désactiver si complètement estompé
                if (not left or left_intensity <= 0.01) and (not right or right_intensity <= 0.01):
                    self._set_effect_state("blush_active", False)
                    return False
                return True

            # Lancer la diminution progressive
            Clock.schedule_interval(fade_step, 0.1)

        # Planifier la fin de l'effet
        self._schedule_effect_timer("blush_timer", fade_out_blush, duration)

    def pulse_light(self, intensity: float = 0.6, color: tuple[float, float, float, float] = None):
        """
        PACK 18: Crée une lumière douce pulsée autour du visage.
        Utile pour l'émerveillement, l'attention, le focus, etc.

        Args:
            intensity: Intensité de l'effet (0.0 à 1.0)
            color: Couleur RGBA optionnelle, par défaut blanche douce
        """
        # Normaliser l'intensité
        intensity = min(1.0, max(0.0, intensity))

        # Couleur par défaut si non spécifiée (blanc doux)
        if not color:
            color = (0.95, 0.95, 1.0, 0.15 * intensity)

        # Activer l'effet
        self._set_effect_state("light_pulse_active", True)
        self._set_effect_state("light_pulse_intensity", intensity)
        self._set_effect_state("light_pulse_color", color)
        self._set_effect_state("light_pulse_phase", 0.0)

        # Durée proportionnelle à l'intensité
        duration = 4.0 + intensity * 4.0  # Entre 4 et 8 secondes

        # Arrêt progressif
        def fade_out_light(dt):
            # Diminution progressive
            def fade_step(dt):
                current = self._get_effect_state("light_pulse_intensity", 0.0)
                current = max(0.0, current - 0.05)
                self._set_effect_state("light_pulse_intensity", current)

                # Désactiver si complètement estompé
                if current <= 0.01:
                    self._set_effect_state("light_pulse_active", False)
                    return False
                return True

            # Lancer la diminution progressive
            Clock.schedule_interval(fade_step, 0.1)

        # Planifier la fin de l'effet
        self._schedule_effect_timer("light_pulse_timer", fade_out_light, duration)

    def apply_frisson(self, intensity: float = 0.7, duration: float = 2.0):
        """
        PACK 18: Applique un effet de frisson visuel (tremblement + vibration + halo).
        Utile pour la surprise, la peur, le froid, l'émotion intense.

        Args:
            intensity: Intensité du frisson (0.0 à 1.0)
            duration: Durée en secondes
        """
        # Normaliser l'intensité
        intensity = min(1.0, max(0.0, intensity))

        # Borner la durée (entre 0.5 et 5 secondes)
        duration = min(5.0, max(0.5, duration))

        # Activer l'effet
        self._set_effect_state("frisson_active", True)
        self._set_effect_state("frisson_intensity", intensity)
        self._set_effect_state("frisson_phase", 0.0)

        # Effet de tremblement
        if hasattr(self, "start_emotional_vibration"):
            self.start_emotional_vibration()

        # Effet de halo rapide qui pulse
        color = (0.9, 0.9, 1.0, 0.1 * intensity)
        self._set_effect_state("frisson_halo_color", color)

        # Programmer la fin de l'effet
        def stop_frisson(dt):
            self._set_effect_state("frisson_active", False)

            # Arrêter la vibration si active
            if hasattr(self, "stop_emotional_vibration"):
                self.stop_emotional_vibration()

        # Arrêter automatiquement après la durée spécifiée
        self._schedule_effect_timer("frisson_timer", stop_frisson, duration)

    #
    # ===== PACK 20 : RÉACTIONS SENSORIELLES AU TOUCHER =====
    #

    def react_to_touch_affectueux(self, zone: str, intensite: float = 0.7):
        """
        PACK 20: Réaction visuelle à un toucher affectueux (caresse, bisou).
        Crée une cascade d'effets visuels coordonnés qui reflètent le plaisir.

        Args:
            zone: Zone touchée (joue, front, etc.)
            intensite: Intensité du toucher (0.0 à 1.0)
        """
        # Normaliser l'intensité
        intensite = min(1.0, max(0.0, intensite))

        # Séquence d'effets visuels coordonnés

        # 1. Léger rougissement sur les joues (progressif)
        self.add_blush(intensite * 0.6, left=True, right=True)

        # 2. Trigger de brillance dans les yeux avec léger délai
        def trigger_eyes(dt):
            self.trigger_eye_sparkle(3.0 + intensite * 2.0, intensite)

        Clock.schedule_once(trigger_eyes, 0.2)

        # 3. Aura de plaisir si intensité suffisante
        if intensite > 0.5:

            def trigger_aura(dt):
                self.trigger_sensory_aura(zone, "affection", intensite * 0.8)

            Clock.schedule_once(trigger_aura, 0.4)

        # 4. Effet de chaleur tactile localisé
        zone_params = self._get_zone_parameters(zone)
        if zone_params:
            center_x, center_y = zone_params["center"]
            self._trigger_tactile_warmth(center_x, center_y, intensite)

        # 5. Pour touches très intenses, déclencher un effet d'émotion plus profond
        if intensite > 0.8:

            def trigger_deep(dt):
                self.pulse_light(intensite * 0.5, color=(0.9, 0.7, 0.8, 0.15 * intensite))

            Clock.schedule_once(trigger_deep, 0.8)

    def react_to_touch_neutre(self, zone: str, intensite: float = 0.5):
        """
        PACK 20: Réaction visuelle à un toucher neutre (contact simple, appui léger).

        Args:
            zone: Zone touchée
            intensite: Intensité du toucher (0.0 à 1.0)
        """
        # Normaliser l'intensité
        intensite = min(1.0, max(0.0, intensite))

        # Effet de micro-mouvement subtil à l'endroit touché
        zone_params = self._get_zone_parameters(zone)
        if zone_params:
            center_x, center_y = zone_params["center"]
            radius = zone_params["radius"] * 0.7

            # Créer un effet de vague très léger et subtil
            ripple_effect = {
                "center": (center_x, center_y),
                "radius": radius,
                "intensity": intensite * 0.4,
                "color": (0.8, 0.8, 0.8, 0.1 * intensite),
                "duration": 1.0,
                "age": 0.0,
            }

            # Ajouter à la liste des effets actifs
            ripples = self._get_effect_state("touch_ripples", [])
            ripples.append(ripple_effect)
            self._set_effect_state("touch_ripples", ripples)

        # Si le toucher est plus prononcé, ajouter un petit flash d'attention
        if intensite > 0.7:

            def flash_attention(dt):
                # Petit flash de lumière très bref
                self.pulse_light(intensite * 0.3, color=(0.9, 0.9, 1.0, 0.07 * intensite))

            Clock.schedule_once(flash_attention, 0.1)

    def react_to_touch_intense(self, zone: str, intensite: float = 0.8, type_toucher: str = "appui"):
        """
        PACK 20: Réaction visuelle à un toucher intense (appui fort, pincement, tape).

        Args:
            zone: Zone touchée
            intensite: Intensité du toucher (0.0 à 1.0)
            type_toucher: Type de toucher ("appui", "pincement", "tape")
        """
        # Normaliser l'intensité
        intensite = min(1.0, max(0.0, intensite))

        # Effet de légère onde de choc depuis le point de contact
        zone_params = self._get_zone_parameters(zone)
        if zone_params:
            center_x, center_y = zone_params["center"]

            # Créer un effet d'onde de choc
            shockwave = {
                "center": (center_x, center_y),
                "radius": zone_params["radius"] * 0.8,
                "max_radius": zone_params["radius"] * 2.0,
                "intensity": intensite,
                "age": 0.0,
                "duration": 0.5 + intensite * 0.5,  # 0.5-1.0 seconde
                "color": (0.9, 0.7, 0.7, 0.15 * intensite)
                if type_toucher == "pincement"
                else (0.8, 0.8, 0.9, 0.12 * intensite),
            }

            # Ajouter à la liste des effets de choc
            waves = self._get_effect_state("shockwaves", [])
            waves.append(shockwave)
            self._set_effect_state("shockwaves", waves)

        # Pour les touches très intenses, déclencher un effet de réaction plus visible
        if intensite > 0.8 or type_toucher == "tape":
            # Léger frisson rapide
            self.apply_frisson(intensite * 0.5, 0.8)

            # Si c'est un pincement ou une tape, possibilité de léger inconfort
            if type_toucher in ["pincement", "tape"] and intensite > 0.6:
                self.trigger_subtle_discomfort(zone, intensite * 0.4)

    def react_to_touch_sensible(self, zone: str, intensite: float = 0.6):
        """
        PACK 20: Réaction visuelle à un toucher sur zone sensible (lèvres, nuque).

        Args:
            zone: Zone touchée
            intensite: Intensité du toucher (0.0 à 1.0)
        """
        # Déterminer si la zone est très sensible
        sensibilite = 0.0

        # Zones particulièrement sensibles
        if zone == "lèvres":
            sensibilite = 0.9
        elif zone in ["joue_gauche", "joue_droite"]:
            sensibilite = 0.7
        elif zone == "nez":
            sensibilite = 0.6
        else:
            sensibilite = 0.4

        # Ajuster l'intensité en fonction de la sensibilité de la zone
        intensite_effective = intensite * sensibilite

        # Créer un léger frisson localisé
        self.trigger_localized_shiver(zone, intensite_effective)

        # Pour les zones très sensibles ou touchers intenses
        if sensibilite > 0.7 or intensite > 0.7:
            # Léger rougissement
            self.add_blush(intensite_effective * 0.5, left=True, right=True)

            # Petite réaction de surprise ou de plaisir selon le contexte
            # Ici, simplement un petit effet lumineux
            def add_light(dt):
                self.pulse_light(intensite_effective * 0.4, color=(0.9, 0.8, 0.8, 0.1 * intensite_effective))

            Clock.schedule_once(add_light, 0.3)

    def update_touch_effects(self, dt):
        """
        PACK 20: Met à jour les effets de toucher comme les ondes, ripples, etc.
        """
        # Mettre à jour les ripples (ondulations de toucher neutre)
        ripples = self._get_effect_state("touch_ripples", [])
        if ripples:
            new_ripples = []
            for ripple in ripples:
                ripple["age"] += dt
                # Expansion progressive
                growth_factor = 1.0 + ripple["age"] * 1.5
                ripple["radius"] *= growth_factor

                # Réduction de l'opacité
                r, g, b, a = ripple["color"]
                fade_factor = max(0.0, 1.0 - (ripple["age"] / ripple["duration"]))
                ripple["color"] = (r, g, b, a * fade_factor)

                # Conserver si encore visible
                if ripple["age"] < ripple["duration"]:
                    new_ripples.append(ripple)

            self._set_effect_state("touch_ripples", new_ripples)

        # Mettre à jour les ondes de choc (toucher intense)
        waves = self._get_effect_state("shockwaves", [])
        if waves:
            new_waves = []
            for wave in waves:
                wave["age"] += dt
                progress = wave["age"] / wave["duration"]

                # Expansion plus rapide au début
                if progress < 0.5:
                    factor = progress * 2.0
                else:
                    factor = 1.0

                # Calculer le rayon actuel
                wave["radius"] = min(wave["max_radius"], wave["radius"] + dt * wave["max_radius"] * 2.0 * factor)

                # Réduire l'intensité avec le temps
                fade_factor = max(0.0, 1.0 - progress)
                r, g, b, a = wave["color"]
                wave["color"] = (r, g, b, a * fade_factor)

                # Conserver si encore visible
                if wave["age"] < wave["duration"]:
                    new_waves.append(wave)

            self._set_effect_state("shockwaves", new_waves)

    def draw_touch_effects(self, canvas):
        """
        PACK 20: Dessine les effets de toucher sur le canvas.

        Args:
            canvas: Canvas Kivy où dessiner
        """
        # Dessiner les ripples (ondulations de toucher neutre)
        ripples = self._get_effect_state("touch_ripples", [])
        for ripple in ripples:
            center_x, center_y = ripple["center"]
            radius = ripple["radius"]
            Color(*ripple["color"])

            # Dessiner plusieurs cercles concentriques
            for i in range(3):
                scale = 1.0 - i * 0.2
                Ellipse(
                    pos=(center_x - radius * scale, center_y - radius * scale),
                    size=(radius * 2 * scale, radius * 2 * scale),
                )

        # Dessiner les ondes de choc (toucher intense)
        waves = self._get_effect_state("shockwaves", [])
        for wave in waves:
            center_x, center_y = wave["center"]
            radius = wave["radius"]
            Color(*wave["color"])

            # Dessiner un cercle fin pour l'onde de choc
            Line(circle=(center_x, center_y, radius), width=1.5)

            # Ajouter un léger remplissage au centre
            inner_radius = radius * 0.7
            r, g, b, a = wave["color"]
            inner_color = (r, g, b, a * 0.3)
            Color(*inner_color)
            Ellipse(
                pos=(center_x - inner_radius, center_y - inner_radius),
                size=(inner_radius * 2, inner_radius * 2),
            )

    def react_to_absence(self, niveau_solitude: float = 0.0):
        """
        PACK 20/19: Réaction visuelle à l'absence prolongée d'interactions.

        Args:
            niveau_solitude: Niveau de solitude (0.0 à 1.0)
        """
        # Ne rien faire si le niveau de solitude est trop faible
        if niveau_solitude < 0.3:
            return

        # Léger assombrissement progressif
        darkness = niveau_solitude * 0.2
        self._set_effect_state("solitude_darkness", darkness)

        # Ralentissement subtil des animations
        slowdown = 1.0 - niveau_solitude * 0.3
        self._set_effect_state("animation_speed_factor", slowdown)

        # Afficher une nébuleuse mentale pour solitude élevée
        if niveau_solitude > 0.7 and not self._get_effect_state("mental_nebula_active", False):
            self.start_mental_nebula()

        # Effet de vide émotionnel doux pour solitude prolongée
        if niveau_solitude > 0.8:
            self.trigger_emotional_emptiness()


# Fonction utilitaire pour dessiner les effets de zone
def draw_zone_effects(canvas, active_zone_effects):
    """
    Dessine les effets actifs sur les différentes zones corporelles.

    Args:
        canvas: Le canvas Kivy où dessiner
        active_zone_effects: Dictionnaire des effets actifs par zone
    """
    if not active_zone_effects:
        return

    current_time = Clock.get_time()

    # Dessiner les effets pour chaque zone
    for zone, effects in active_zone_effects.items():
        for effect_type, effect in effects.items():
            # Récupérer les paramètres communs
            center_x, center_y = effect["center"]
            radius = effect["radius"]
            intensity = effect["intensity"]
            phase = effect["phase"]
            elapsed = current_time - effect["start_time"]
            duration = effect["duration"]

            # Facteur de fondu pour la fin de l'effet
            fade_factor = 1.0
            if elapsed > duration * 0.7:
                # Commencer à fondre à partir de 70% de la durée
                fade_factor = max(0.0, 1.0 - (elapsed - duration * 0.7) / (duration * 0.3))

            # Adapter le dessin selon le type d'effet
            if effect_type == "shiver":
                # Frisson: oscillation rapide
                shiver_x = math.sin(phase * 8.0) * 2.0 * intensity
                shiver_y = math.cos(phase * 7.0) * 1.5 * intensity

                # Dessiner un léger halo vibrant
                shiver_color = (0.9, 0.9, 1.0, 0.07 * intensity * fade_factor)
                Color(*shiver_color)

                for i in range(3):
                    offset_scale = 1.0 + i * 0.2
                    Ellipse(
                        pos=(
                            center_x - radius * offset_scale + shiver_x,
                            center_y - radius * offset_scale + shiver_y,
                        ),
                        size=(radius * 2 * offset_scale, radius * 2 * offset_scale),
                    )

            elif effect_type == "aura":
                # Aura: halo pulsant doux
                color = effect["color"]
                r, g, b, a = color

                # Pulsation douce
                pulse = 0.2 * math.sin(phase * 2.0) + 0.8
                pulse_radius = radius * pulse

                # Couleur avec fondu
                aura_color = (r, g, b, a * fade_factor)
                Color(*aura_color)

                # Plusieurs couches avec opacité décroissante
                for i in range(3):
                    layer_scale = 1.0 + i * 0.15
                    layer_alpha = a * fade_factor * (1.0 - i * 0.3)
                    Color(r, g, b, layer_alpha)

                    Ellipse(
                        pos=(
                            center_x - pulse_radius * layer_scale,
                            center_y - pulse_radius * layer_scale,
                        ),
                        size=(pulse_radius * 2 * layer_scale, pulse_radius * 2 * layer_scale),
                    )

            elif effect_type == "discomfort":
                # Gêne: petites contractions irrégulières
                squeeze_x = math.sin(phase * 5.0) * intensity * 0.3
                squeeze_y = math.cos(phase * 6.0) * intensity * 0.3

                # Couleur légèrement rougeâtre
                discomfort_color = (0.9, 0.7, 0.7, 0.15 * intensity * fade_factor)
                Color(*discomfort_color)

                # Ellipse déformée
                Ellipse(
                    pos=(
                        center_x - radius * (1.0 - squeeze_x),
                        center_y - radius * (1.0 + squeeze_y),
                    ),
                    size=(radius * 2 * (1.0 + squeeze_x), radius * 2 * (1.0 - squeeze_y)),
                )
