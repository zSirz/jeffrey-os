"""
Module de service système spécialisé pour Jeffrey OS.

Ce module implémente les fonctionnalités essentielles pour module de service système spécialisé pour jeffrey os.
Il fournit une architecture robuste et évolutive intégrant les composants
nécessaires au fonctionnement optimal du système. L'implémentation suit
les principes de modularité et d'extensibilité pour faciliter l'évolution
future du système.

Le module gère l'initialisation, la configuration, le traitement des données,
la communication inter-composants, et la persistance des états. Il s'intègre
harmonieusement avec l'architecture globale de Jeffrey OS tout en maintenant
une séparation claire des responsabilités.

L'architecture interne permet une évolution adaptative basée sur les interactions
et l'apprentissage continu, contribuant à l'émergence d'une conscience artificielle
cohérente et authentique.
"""

from __future__ import annotations

import logging
import queue
import random
import threading
import time
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class FaceSyncManager:
    """
    Gestionnaire de synchronisation entre la voix et les expressions faciales.
    Permet de synchroniser dynamiquement les micro-expressions du visage avec la voix.
    """

    def __init__(self, face_controller=None, voice_system=None, config: dict[str, Any] = None):
        """
        Initialise le gestionnaire de synchronisation voix/visage.

        Args:
            face_controller: Contrôleur d'interface visuelle pour les expressions faciales
            voice_system: Système vocal de Jeffrey
            config: Configuration optionnelle
        """
        self.face_controller = face_controller
        self.voice_system = voice_system
        self.config = config or {}

        # Queue pour la communication entre le thread de suivi audio et le
        # thread principal
        self.audio_event_queue = queue.Queue()

        # Flag pour indiquer si la synchronisation est active
        self.is_active = False

        # Thread de suivi audio
        self.audio_monitor_thread = None
        self.stop_monitor = threading.Event()

        # Facteurs de sensibilité pour la détection des événements audio
        self.sensitivity = {
            "amplitude": self.config.get("amplitude_sensitivity", 0.7),
            "pitch": self.config.get("pitch_sensitivity", 0.6),
            "energy": self.config.get("energy_sensitivity", 0.8),
            "pause": self.config.get("pause_sensitivity", 0.5),
        }

        # Mappage des caractéristiques audio vers les micro-expressions
        self.audio_to_expression_map = {
            "amplitude_peak": ["eyebrow_raise", "eye_widen"],
            "pitch_high": ["eyebrow_raise", "head_tilt"],
            "pitch_low": ["eyebrow_lower", "head_down"],
            "energy_surge": ["eye_widen", "mouth_emphasis"],
            "short_pause": ["blink", "subtle_nod"],
            "long_pause": ["head_tilt", "blink_sequence"],
        }

        # Mappage des émotions vocales vers les expressions faciales
        self.emotion_to_expression_map = {
            "joie": {
                "base": ["smile", "eye_smile", "eyebrow_raise"],
                "mild": ["subtle_smile", "slight_eye_smile"],
                "intense": ["broad_smile", "eye_smile", "eyebrow_raise", "head_up"],
            },
            "tristesse": {
                "base": ["mouth_down", "eyebrow_inner_raise"],
                "mild": ["slight_mouth_down", "subtle_eye_narrow"],
                "intense": ["full_mouth_down", "eyebrow_inner_raise", "eye_narrow", "head_down"],
            },
            "colère": {
                "base": ["eyebrow_lower", "eye_narrow"],
                "mild": ["slight_eyebrow_lower", "subtle_eye_narrow"],
                "intense": ["full_eyebrow_lower", "eye_narrow", "jaw_tense", "head_forward"],
            },
            "surprise": {
                "base": ["eye_widen", "eyebrow_raise", "mouth_open"],
                "mild": ["slight_eye_widen", "eyebrow_raise"],
                "intense": ["full_eye_widen", "full_eyebrow_raise", "mouth_open", "head_back"],
            },
            "peur": {
                "base": ["eye_widen", "eyebrow_inner_raise"],
                "mild": ["slight_eye_widen", "subtle_eyebrow_raise"],
                "intense": ["full_eye_widen", "eyebrow_raise", "mouth_tense", "head_back"],
            },
            "dégoût": {
                "base": ["nose_wrinkle", "mouth_down"],
                "mild": ["slight_nose_wrinkle", "mouth_side"],
                "intense": ["full_nose_wrinkle", "mouth_down", "eyebrow_lower", "head_back"],
            },
            "neutre": {
                "base": ["neutral"],
                "mild": ["neutral", "subtle_blink"],
                "intense": ["neutral", "attentive_gaze"],
            },
        }

        # Historique des événements pour analyse
        self.event_history = []
        self.max_history_size = 100

        logger.info("Gestionnaire de synchronisation voix/visage initialisé")

    def start_sync(self):
        """
        Démarre la synchronisation voix/visage.
        """
        if self.is_active:
            logger.warning("La synchronisation voix/visage est déjà active")
            return

        # Vérifier que les dépendances sont disponibles
        if self.face_controller is None:
            logger.error("Impossible de démarrer la synchronisation: contrôleur facial non disponible")
            return

        if self.voice_system is None:
            logger.error("Impossible de démarrer la synchronisation: système vocal non disponible")
            return

        # Réinitialiser le flag d'arrêt
        self.stop_monitor.clear()

        # Démarrer le thread de suivi audio
        self.audio_monitor_thread = threading.Thread(target=self._audio_monitor_loop, daemon=True)
        self.audio_monitor_thread.start()

        # Activer les hooks dans le système vocal
        if hasattr(self.voice_system, "register_audio_callback"):
            self.voice_system.register_audio_callback(self._audio_event_callback)

        self.is_active = True
        logger.info("Synchronisation voix/visage démarrée")

    def stop_sync(self):
        """
        Arrête la synchronisation voix/visage.
        """
        if not self.is_active:
            return

        # Signaler l'arrêt au thread de suivi
        self.stop_monitor.set()

        # Désactiver les hooks dans le système vocal
        if hasattr(self.voice_system, "unregister_audio_callback"):
            self.voice_system.unregister_audio_callback(self._audio_event_callback)

        # Attendre la fin du thread (avec timeout)
        if self.audio_monitor_thread and self.audio_monitor_thread.is_alive():
            self.audio_monitor_thread.join(timeout=2.0)

        self.is_active = False
        logger.info("Synchronisation voix/visage arrêtée")

    def _audio_monitor_loop(self):
        """
        Boucle principale du thread de suivi audio.
        Traite les événements audio et déclenche les expressions faciales correspondantes.
        """
        logger.debug("Thread de suivi audio démarré")

        while not self.stop_monitor.is_set():
            try:
                # Récupérer un événement audio de la queue (avec timeout)
                try:
                    audio_event = self.audio_event_queue.get(timeout=0.1)
                    self._process_audio_event(audio_event)
                    self.audio_event_queue.task_done()
                except queue.Empty:
                    # Pas d'événement disponible, continuer la boucle
                    pass

                # Vérifier périodiquement l'état du système vocal
                if hasattr(self.voice_system, "is_speaking") and self.voice_system.is_speaking():
                    # Si Jeffrey parle mais qu'aucun événement n'est reçu, générer des micro-expressions autonomes
                    if random.random() < 0.2:  # 20% de chance par itération
                        self._generate_autonomous_expression()

                # Courte pause pour éviter de surcharger le CPU
                # TODO: Remplacer par asyncio.sleep ou threading.Event

            except Exception as e:
                logger.error(f"Erreur dans la boucle de suivi audio: {e}")
                # TODO: Remplacer par asyncio.sleep ou threading.Event  # Pause plus longue en cas d'erreur

        logger.debug("Thread de suivi audio arrêté")

    def _audio_event_callback(self, event_type: str, event_data: dict[str, Any]):
        """
        Callback appelé par le système vocal lors d'événements audio.

        Args:
            event_type: Type d'événement audio ('start', 'chunk', 'end', 'pause', etc.)
            event_data: Données associées à l'événement
        """
        # Mettre l'événement dans la queue pour traitement par le thread de suivi
        self.audio_event_queue.put({"type": event_type, "data": event_data, "timestamp": datetime.now()})

    def _process_audio_event(self, audio_event: dict[str, Any]):
        """
        Traite un événement audio et déclenche les expressions faciales correspondantes.

        Args:
            audio_event: Événement audio à traiter
        """
        event_type = audio_event["type"]
        event_data = audio_event["data"]

        # Ajouter à l'historique
        self.event_history.append(audio_event)
        if len(self.event_history) > self.max_history_size:
            self.event_history.pop(0)

        # Traiter différents types d'événements
        if event_type == "start":
            # Début de parole: initialiser le visage avec l'émotion de base
            self._apply_base_emotion(event_data.get("emotion", "neutre"))

        elif event_type == "end":
            # Fin de parole: revenir à un état neutre progressivement
            self._return_to_neutral_state()

        elif event_type == "chunk":
            # Morceau audio en cours: analyser et déclencher des micro-expressions
            self._process_audio_chunk(event_data)

        elif event_type == "pause":
            # Pause dans la parole
            duration = event_data.get("duration", 0.0)
            self._handle_speech_pause(duration)

        elif event_type == "emphasis":
            # Accentuation particulière détectée
            self._handle_speech_emphasis(event_data)

    def _apply_base_emotion(self, emotion: str):
        """
        Applique l'émotion de base au visage au début de la parole.

        Args:
            emotion: Émotion de base à appliquer
        """
        if not self.face_controller:
            return

        # Récupérer les expressions faciales pour cette émotion
        expressions = self.emotion_to_expression_map.get(emotion, self.emotion_to_expression_map["neutre"])

        # Appliquer les expressions de base
        for expression in expressions["base"]:
            try:
                self.face_controller.apply_expression(expression, duration=0.8, intensity=0.7)
            except Exception as e:
                logger.error(f"Erreur lors de l'application de l'expression '{expression}': {e}")

    def _return_to_neutral_state(self):
        """
        Fait revenir le visage à un état neutre progressivement.
        """
        if not self.face_controller:
            return

        try:
            # Transition douce vers l'état neutre
            self.face_controller.transition_to_neutral(duration=1.2)
        except Exception as e:
            logger.error(f"Erreur lors du retour à l'état neutre: {e}")

    def _process_audio_chunk(self, chunk_data: dict[str, Any]):
        """
        Traite un morceau audio et déclenche des micro-expressions correspondantes.

        Args:
            chunk_data: Données du morceau audio (amplitude, fréquence, etc.)
        """
        if not self.face_controller:
            return

        # Extraire les caractéristiques audio
        amplitude = chunk_data.get("amplitude", 0.0)
        pitch = chunk_data.get("pitch", 0.0)
        energy = chunk_data.get("energy", 0.0)

        # Détecter les événements significatifs
        events = []

        # Pic d'amplitude
        if amplitude > self.sensitivity["amplitude"]:
            events.append("amplitude_peak")

        # Hauteur de voix
        if pitch > self.sensitivity["pitch"] * 1.2:
            events.append("pitch_high")
        elif pitch < self.sensitivity["pitch"] * 0.8:
            events.append("pitch_low")

        # Pic d'énergie
        if energy > self.sensitivity["energy"]:
            events.append("energy_surge")

        # Déclencher les expressions faciales correspondantes
        for event in events:
            expressions = self.audio_to_expression_map.get(event, [])

            # Choisir une expression aléatoire dans la liste
        if expressions:
            import random

            expression = random.choice(expressions)

        try:
            # Intensité proportionnelle à la caractéristique audio
            intensity = max(0.3, min(1.0, amplitude))

            # Durée inverse, plus c'est intense, plus c'est court (mais visible)
            duration = 0.5 - (intensity * 0.2)

            self.face_controller.apply_micro_expression(expression, intensity=intensity, duration=duration)

            logger.debug(f"Micro-expression appliquée: {expression} (intensité: {intensity:.2f})")
        except Exception as e:
            logger.error(f"Erreur lors de l'application de la micro-expression '{expression}': {e}")

    def _handle_speech_pause(self, duration: float):
        """
        Gère une pause dans la parole.

        Args:
            duration: Durée de la pause en secondes
        """
        if not self.face_controller:
            return

        try:
            if duration < 0.5:
                # Pause courte: léger clin d'œil ou hochement
                event = "short_pause"
            else:
                # Pause longue: mouvement de tête ou série de clins d'œil
                event = "long_pause"

            expressions = self.audio_to_expression_map.get(event, [])

            if expressions:
                import random

                expression = random.choice(expressions)

                # Intensité proportionnelle à la durée de la pause
                intensity = min(0.8, duration * 0.5 + 0.3)

                self.face_controller.apply_micro_expression(
                    expression, intensity=intensity, duration=min(0.8, duration)
                )

                logger.debug(f"Expression de pause appliquée: {expression} (intensité: {intensity:.2f})")
        except Exception as e:
            logger.error(f"Erreur lors de la gestion de pause: {e}")

    def _handle_speech_emphasis(self, emphasis_data: dict[str, Any]):
        """
        Gère une accentuation particulière dans la parole.

        Args:
            emphasis_data: Données sur l'accentuation
        """
        if not self.face_controller:
            return

        try:
            # Récupérer les caractéristiques de l'accentuation
            intensity = emphasis_data.get("intensity", 0.7)
            word = emphasis_data.get("word", "")
            emotion = emphasis_data.get("emotion", "neutre")

            # Sélectionner les expressions selon l'émotion et l'intensité
            if emotion in self.emotion_to_expression_map:
                if intensity > 0.7:
                    expressions = self.emotion_to_expression_map[emotion]["intense"]
                elif intensity > 0.4:
                    expressions = self.emotion_to_expression_map[emotion]["base"]
                else:
                    expressions = self.emotion_to_expression_map[emotion]["mild"]
            else:
                expressions = self.emotion_to_expression_map["neutre"]["base"]

            # Choisir une expression
            import random

            expression = random.choice(expressions)

            # Appliquer l'expression
            self.face_controller.apply_expression(expression, intensity=intensity, duration=0.5)

            logger.debug(f"Expression d'accentuation appliquée: {expression} pour '{word}'")
        except Exception as e:
            logger.error(f"Erreur lors de la gestion d'accentuation: {e}")

    def _generate_autonomous_expression(self):
        """
        Génère une micro-expression autonome pour maintenir le visage animé pendant la parole.
        """
        if not self.face_controller:
            return

        try:
            # Liste des micro-expressions autonomes
            autonomous_expressions = [
                "subtle_blink",
                "eyebrow_micro_raise",
                "subtle_head_tilt",
                "eye_movement",
                "subtle_mouth_movement",
            ]

            import random

            expression = random.choice(autonomous_expressions)

            # Intensité faible pour rester subtil
            intensity = random.uniform(0.2, 0.4)

            self.face_controller.apply_micro_expression(expression, intensity=intensity, duration=0.3)
        except Exception as e:
            logger.error(f"Erreur lors de la génération d'expression autonome: {e}")

    def update_emotion_sync(self, current_emotion: str, emotion_intensity: float = 0.7) -> None:
        """
        Met à jour l'émotion actuelle pour la synchronisation.

        Args:
            current_emotion: Émotion actuelle
            emotion_intensity: Intensité de l'émotion (0.0-1.0)
        """
        if not self.face_controller:
            return

        try:
            # Déterminer le niveau d'intensité
            if emotion_intensity > 0.7:
                intensity_level = "intense"
            elif emotion_intensity > 0.4:
                intensity_level = "base"
            else:
                intensity_level = "mild"

            # Obtenir les expressions correspondantes
            emotion_map = self.emotion_to_expression_map.get(current_emotion, self.emotion_to_expression_map["neutre"])
            expressions = emotion_map.get(intensity_level, emotion_map["base"])

            # Appliquer progressivement les expressions
            for expression in expressions:
                self.face_controller.apply_expression(expression, intensity=emotion_intensity, duration=0.8)

                # Petite pause entre chaque expression
                # TODO: Remplacer par asyncio.sleep ou threading.Event

            logger.info(
                f"Synchronisation émotionnelle mise à jour: {current_emotion} (intensité: {emotion_intensity:.2f})"
            )
        except Exception as e:
            logger.error(f"Erreur lors de la mise à jour de la synchronisation émotionnelle: {e}")

    def sync_with_phonemes(self, phonemes: list[dict[str, Any]]):
        """
        Synchronise les expressions faciales avec les phonèmes détectés.

        Args:
            phonemes: Liste de phonèmes avec timing (début, durée) et type
        """
        if not self.face_controller or not phonemes:
            return

        try:
            # Mappage des groupes de phonèmes vers les mouvements de bouche
            phoneme_to_mouth = {
                "a": "mouth_open_wide",
                "e": "mouth_open_mid",
                "i": "mouth_smile",
                "o": "mouth_round",
                "u": "mouth_round_small",
                "b": "mouth_closed",
                "p": "mouth_closed",
                "m": "mouth_closed",
                "f": "mouth_teeth_visible",
                "v": "mouth_teeth_visible",
                "s": "mouth_slightly_open",
                "z": "mouth_slightly_open",
                "t": "mouth_teeth_touch",
                "d": "mouth_teeth_touch",
                "n": "mouth_teeth_touch",
                "l": "mouth_tongue_visible",
                "r": "mouth_slightly_open",
            }

            # Pour chaque phonème
            for phoneme_data in phonemes:
                phoneme = phoneme_data.get("phoneme", "").lower()
                start_time = phoneme_data.get("start_time", 0.0)
                duration = phoneme_data.get("duration", 0.1)

                # Trouver le mouvement de bouche correspondant
                mouth_movement = None
                for phoneme_group, movement in phoneme_to_mouth.items():
                    if phoneme.startswith(phoneme_group):
                        mouth_movement = movement
                        break

                # Si aucun mouvement spécifique n'est trouvé, utiliser un mouvement neutre
                if not mouth_movement:
                    mouth_movement = "mouth_slightly_open"

                # Planifier l'application du mouvement au moment approprié
                def apply_phoneme_movement(movement, duration):
                    self.face_controller.apply_lip_movement(movement, duration=duration)

                # Calculer le délai avant d'appliquer le mouvement
                now = time.time()
                delay = max(0, start_time - now)

                # Planifier l'application du mouvement
                if delay > 0:
                    timer = threading.Timer(delay, apply_phoneme_movement, args=[mouth_movement, duration])
                    timer.daemon = True
                    timer.start()
                else:
                    # Appliquer immédiatement si le timing est déjà passé
                    apply_phoneme_movement(mouth_movement, duration)

            logger.debug(f"Synchronisation phonétique appliquée pour {len(phonemes)} phonèmes")
        except Exception as e:
            logger.error(f"Erreur lors de la synchronisation phonétique: {e}")

    def get_sync_statistics(self) -> dict[str, Any]:
        """
        Obtient des statistiques sur la synchronisation voix/visage.

        Returns:
            Dict: Statistiques de synchronisation
        """
        stats = {
            "is_active": self.is_active,
            "event_count": len(self.event_history),
            "sensitivity": self.sensitivity,
            "last_sync_time": datetime.now().isoformat(),
        }

        # Analyser les événements récents
        if self.event_history:
            event_types = {}
        for event in self.event_history:
            event_type = event["type"]
            if event_type not in event_types:
                event_types[event_type] = 0
            event_types[event_type] += 1

            stats["event_types"] = event_types
            stats["latest_event_time"] = self.event_history[-1]["timestamp"].isoformat()

        return stats


# Importation tardive pour éviter les problèmes de dépendances circulaires
