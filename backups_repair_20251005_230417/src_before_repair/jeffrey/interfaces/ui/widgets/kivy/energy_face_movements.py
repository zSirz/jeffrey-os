#!/usr/bin/env python
"""
energy_face_movements.py - Gestion des mouvements du visage de Jeffrey
Partie de la refactorisation du fichier energy_face.py d'origine (PACK 18)

Ce module gère les mouvements et animations du visage :
- Micro-mouvements corporels
- Mouvements expressifs
- Animation du regard et de la bouche
- Gestion des effets de mouvement dynamique
"""

import math
import random

from kivy.clock import Clock


class MovementHandler:
    """
    Gestionnaire des mouvements et animations pour le visage de Jeffrey.
    Contrôle toutes les animations et micro-mouvements.
    """

    def __init__(self, face_widget):
        """
        Initialise le gestionnaire de mouvements.

        Args:
            face_widget: Widget du visage (EnergyFaceCoreWidget)
        """
        self.face = face_widget

        # Initialiser les variables de mouvement
        self.movement_amount = 0.0  # Quantité de mouvement actuelle
        self.max_movement = 0.0  # Mouvement maximal autorisé
        self.movement_decay = 0.97  # Facteur de diminution du mouvement

        # Paramètres pour les micro-mouvements
        self.micro_movement_chance = 0.1  # Probabilité de micro-mouvement par seconde
        self.micro_movement_intensity = 0.3  # Intensité des micro-mouvements

        # Variables pour la vibration émotionnelle
        self.vibration_active = False
        self.vibration_phase = 0.0
        self.vibration_intensity = 0.0

        # Timers pour les effets de mouvement
        self._movement_timers = {}

        # Planifier le check des micro-mouvements corporels
        Clock.schedule_interval(self.update_movements, 1 / 30.0)
        Clock.schedule_interval(self.check_micro_movements, 1.0)

    def update_movements(self, dt):
        """
        Met à jour tous les mouvements et animations du visage.

        Args:
            dt: Delta temps depuis la dernière mise à jour
        """
        # Mise à jour des vibrations si actives
        if self.vibration_active:
            self._update_vibration(dt)

        # Diminution progressive de la quantité de mouvement
        if self.movement_amount > 0:
            self.movement_amount *= self.movement_decay
            if self.movement_amount < 0.01:
                self.movement_amount = 0

    def _update_vibration(self, dt):
        """
        Met à jour l'effet de vibration du visage.

        Args:
            dt: Delta temps depuis la dernière mise à jour
        """
        # Mise à jour de la phase de vibration
        self.vibration_phase += dt * 12.0  # Fréquence de vibration

        # Calculer le déplacement horizontal
        vibration_offset = math.sin(self.vibration_phase) * self.vibration_intensity * 1.5

        # Appliquer le déplacement si le widget face a la méthode nécessaire
        if hasattr(self.face, "_set_effect_state"):
            self.face._set_effect_state("vibration_offset_x", vibration_offset)

    def check_micro_movements(self, dt):
        """
        Vérifie s'il faut déclencher des micro-mouvements expressifs.

        Args:
            dt: Delta temps depuis la dernière vérification
        """
        # Probabilité de micro-mouvement basée sur l'émotion
        chance = self.micro_movement_chance

        # Augmenter la chance selon l'émotion
        emotion = getattr(self.face, "emotion", "neutre")
        if emotion in ["joie", "excitation", "surprise"]:
            chance *= 2.0
        elif emotion in ["nervosité", "anxiété", "stress"]:
            chance *= 1.5

        # Vérifier si un micro-mouvement doit être déclenché
        if random.random() < chance:
            self._trigger_random_micro_movement()

    def _trigger_random_micro_movement(self):
        """Déclenche un micro-mouvement aléatoire expressif."""
        # Liste des types de micro-mouvements possibles
        movement_types = [
            "eye_dart",  # Mouvement rapide des yeux
            "eyebrow_raise",  # Haussement de sourcil
            "head_tilt",  # Inclinaison légère de la tête
            "mouth_twitch",  # Petit mouvement de bouche
            "blink_sequence",  # Séquence de clignements
        ]

        # Choisir un mouvement aléatoire
        movement = random.choice(movement_types)

        # Exécuter le micro-mouvement choisi
        if movement == "eye_dart":
            self._eye_dart_movement()
        elif movement == "eyebrow_raise":
            self._eyebrow_raise_movement()
        elif movement == "head_tilt":
            self._head_tilt_movement()
        elif movement == "mouth_twitch":
            self._mouth_twitch_movement()
        elif movement == "blink_sequence":
            self._blink_sequence_movement()

    def _eye_dart_movement(self):
        """Mouvement rapide des yeux dans une direction aléatoire."""
        # Paramètres du mouvement
        max_offset = 4.0
        duration = 0.15

        # Calculer les directions aléatoires
        x_offset = random.uniform(-max_offset, max_offset)
        y_offset = random.uniform(-max_offset / 2, max_offset / 2)

        # Appliquer le décalage au regard
        if hasattr(self.face, "iris_offset_x"):
            original_x = self.face.iris_offset_x
            original_y = self.face.iris_offset_y

            # Fonction pour l'animation du mouvement
            def eye_dart_animation(dt):
                progress = min(1.0, dt / duration)

                # Mouvement aller
                if progress < 0.4:
                    # Accélération jusqu'à l'offset maximum
                    factor = progress / 0.4
                    self.face.iris_offset_x = original_x + x_offset * factor
                    self.face.iris_offset_y = original_y + y_offset * factor
                # Maintien
                elif progress < 0.6:
                    self.face.iris_offset_x = original_x + x_offset
                    self.face.iris_offset_y = original_y + y_offset
                # Retour
                else:
                    # Retour progressif à la position d'origine
                    factor = (1.0 - progress) / 0.4
                    self.face.iris_offset_x = original_x + x_offset * factor
                    self.face.iris_offset_y = original_y + y_offset * factor

                # Continuer l'animation jusqu'à la fin
                if progress < 1.0:
                    return True
                else:
                    # Restaurer la position exacte d'origine
                    self.face.iris_offset_x = original_x
                    self.face.iris_offset_y = original_y
                    return False

            # Lancer l'animation
            Clock.schedule_interval(eye_dart_animation, 1 / 60.0)

    def _eyebrow_raise_movement(self):
        """Haussement de sourcil expressif."""
        # Paramètres du mouvement
        max_offset = 5.0
        duration = 0.4

        # Choisir le sourcil à déplacer (gauche, droit ou les deux)
        side = random.choice(["left", "right", "both"])

        # Appliquer le haussement de sourcil
        if hasattr(self.face, "eyebrow_left_offset"):
            original_left = self.face.eyebrow_left_offset if hasattr(self.face, "eyebrow_left_offset") else 0
            original_right = self.face.eyebrow_right_offset if hasattr(self.face, "eyebrow_right_offset") else 0

            # Fonction pour l'animation du mouvement
            def eyebrow_animation(dt):
                progress = min(1.0, dt / duration)

                # Courbe d'animation douce (sinusoïdale)
                factor = math.sin(progress * math.pi)

                # Appliquer le mouvement selon le côté choisi
                if side in ["left", "both"]:
                    self.face.eyebrow_left_offset = original_left + max_offset * factor
                if side in ["right", "both"]:
                    self.face.eyebrow_right_offset = original_right + max_offset * factor

                # Continuer l'animation jusqu'à la fin
                if progress < 1.0:
                    return True
                else:
                    # Restaurer la position exacte d'origine
                    if side in ["left", "both"]:
                        self.face.eyebrow_left_offset = original_left
                    if side in ["right", "both"]:
                        self.face.eyebrow_right_offset = original_right
                    return False

            # Lancer l'animation
            Clock.schedule_interval(eyebrow_animation, 1 / 60.0)

    def _head_tilt_movement(self):
        """Légère inclinaison de la tête."""
        # Paramètres du mouvement
        max_angle = 3.0
        duration = 0.5

        # Choisir la direction (inclinaison horaire ou anti-horaire)
        angle = random.choice([-max_angle, max_angle])

        # Appliquer l'inclinaison
        if hasattr(self.face, "head_rotation"):
            original_rotation = self.face.head_rotation

            # Fonction pour l'animation du mouvement
            def head_tilt_animation(dt):
                progress = min(1.0, dt / duration)

                # Mouvement aller
                if progress < 0.3:
                    # Rotation progressive
                    factor = progress / 0.3
                    self.face.head_rotation = original_rotation + angle * factor
                # Maintien
                elif progress < 0.7:
                    self.face.head_rotation = original_rotation + angle
                # Retour
                else:
                    # Rotation inverse progressive
                    factor = (1.0 - progress) / 0.3
                    self.face.head_rotation = original_rotation + angle * factor

                # Continuer l'animation jusqu'à la fin
                if progress < 1.0:
                    return True
                else:
                    # Restaurer la rotation exacte d'origine
                    self.face.head_rotation = original_rotation
                    return False

            # Lancer l'animation
            Clock.schedule_interval(head_tilt_animation, 1 / 60.0)

    def _mouth_twitch_movement(self):
        """Petit mouvement de bouche expressif."""
        # Paramètres du mouvement
        duration = 0.3

        # Mémoriser la forme actuelle de la bouche
        original_shape = self.face.current_mouth_shape

        # Choisir une forme temporaire
        temp_shapes = ["I", "E", "OE", "IN"]
        temp_shape = random.choice(temp_shapes)

        # Fonction pour l'animation du mouvement
        def mouth_twitch_animation(dt):
            elapsed = min(dt, duration)

            # Première moitié : transition vers la forme temporaire
            if elapsed < duration / 2:
                self.face.current_mouth_shape = temp_shape
            # Seconde moitié : retour à la forme originale
            else:
                self.face.current_mouth_shape = original_shape
                return False

            return True

        # Lancer l'animation
        Clock.schedule_once(mouth_twitch_animation, duration / 2)

    def _blink_sequence_movement(self):
        """Séquence de clignements rapides des yeux."""
        # Paramètres de la séquence
        blink_count = random.randint(2, 3)
        blink_interval = 0.15

        # Mémoriser l'ouverture actuelle des paupières
        original_openness = self.face.eyelid_openness

        # Compteur pour suivre le nombre de clignements effectués
        blink_counter = [0]  # Utilisation d'une liste pour pouvoir modifier la valeur dans la closure

        # Fonction pour un clignement
        def do_blink(dt):
            # Fermer rapidement
            self.face.eyelid_openness = 0.1

            # Programmer la réouverture
            Clock.schedule_once(lambda dt: setattr(self.face, "eyelid_openness", original_openness), 0.05)

            # Incrémenter le compteur
            blink_counter[0] += 1

            # Programmer le prochain clignement si nécessaire
            if blink_counter[0] < blink_count:
                Clock.schedule_once(do_blink, blink_interval)

        # Lancer la séquence
        do_blink(0)

    def start_vibration(self, intensity: float = 0.5, duration: float = 2.0):
        """
        Démarre un effet de vibration du visage.

        Args:
            intensity: Intensité de la vibration (0.0 à 1.0)
            duration: Durée de l'effet en secondes
        """
        # Activer la vibration
        self.vibration_active = True
        self.vibration_intensity = intensity
        self.vibration_phase = 0.0

        # Arrêter automatiquement après la durée spécifiée
        Clock.schedule_once(lambda dt: self.stop_vibration(), duration)

    def stop_vibration(self):
        """Arrête l'effet de vibration."""
        self.vibration_active = False
        self.vibration_intensity = 0.0

        # Réinitialiser le décalage de vibration
        if hasattr(self.face, "_set_effect_state"):
            self.face._set_effect_state("vibration_offset_x", 0.0)

    def animate_mouth_speak(self, text: str, duration: float = None):
        """
        Anime la bouche pour simuler la parole d'un texte.

        Args:
            text: Texte à prononcer
            duration: Durée totale de la parole (si None, calculé à partir du texte)
        """
        # Calculer la durée si non fournie
        if duration is None:
            # Estimation très simple : ~5 caractères par seconde
            duration = max(1.0, len(text) / 5.0)

        # Analyser le texte pour créer les événements de synchronisation labiale
        events = self._generate_lip_sync_events(text, duration)

        # Commencer à parler
        self.face.is_speaking = True

        # Appliquer les événements générés
        self.face.lip_sync_events = events

        # Arrêter de parler après la durée
        Clock.schedule_once(lambda dt: setattr(self.face, "is_speaking", False), duration)

    def _generate_lip_sync_events(self, text: str, duration: float) -> list[dict]:
        """
        Génère des événements de synchronisation labiale à partir d'un texte.

        Args:
            text: Texte à analyser
            duration: Durée totale de la parole

        Returns:
            Liste d'événements de synchronisation labiale
        """
        events = []

        # Voyelles fréquentes et leurs formes de bouche correspondantes
        vowels = {
            "a": "A",
            "à": "A",
            "â": "A",
            "e": "E",
            "é": "E",
            "è": "E",
            "ê": "E",
            "i": "I",
            "î": "I",
            "o": "O",
            "ô": "O",
            "u": "OU",
            "ù": "OU",
            "û": "OU",
            "ou": "OU",
            "on": "ON",
            "an": "AN",
            "en": "AN",
            "in": "IN",
            "un": "IN",
            "eu": "EU",
        }

        # Nombre total de caractères à traiter
        total_chars = max(1, len(text))

        # Temps moyen par caractère
        time_per_char = duration / total_chars

        # Position temporelle courante
        current_time = 0.0

        # Parcourir le texte et générer des événements pour les voyelles
        i = 0
        while i < len(text):
            # Vérifier les digrammes d'abord
            if i < len(text) - 1:
                digram = text[i : i + 2].lower()
                if digram in vowels:
                    shape = vowels[digram]

                    # Créer un événement de synchronisation
                    event = {
                        "start_time": current_time,
                        "end_time": current_time + time_per_char * 2.0,
                        "shape": shape,
                    }
                    events.append(event)

                    # Avancer de 2 caractères et mettre à jour le temps
                    i += 2
                    current_time += time_per_char * 2.0
                    continue

            # Sinon, vérifier les caractères individuels
            char = text[i].lower()
            if char in vowels:
                shape = vowels[char]

                # Créer un événement de synchronisation
                event = {
                    "start_time": current_time,
                    "end_time": current_time + time_per_char * 1.5,
                    "shape": shape,
                }
                events.append(event)

            # Avancer d'un caractère et mettre à jour le temps
            i += 1
            current_time += time_per_char

        # Ajouter un événement neutre à la fin
        events.append({"start_time": duration - 0.2, "end_time": duration, "shape": "X"})

        return events
