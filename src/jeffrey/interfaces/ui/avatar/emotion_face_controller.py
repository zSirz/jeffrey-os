"""
Module de contrôle et d'intégration du visage émotionnel immersif de Jeffrey.

Ce module coordonne l'intégration du widget ImmersiveEmotionFace dans l'interface utilisateur,
en assurant sa connexion avec le moteur émotionnel, le système d'apprentissage et les autres
composants du système Jeffrey.
"""

import logging

# Import du moteur émotionnel et du système d'apprentissage
from jeffrey.core.emotions.emotional_engine import EmotionalEngine
from jeffrey.core.emotions.emotional_learning import EmotionalLearning
from kivy.clock import Clock
from widgets.energy_face import EnergyFaceWidget

# Import des widgets
from widgets.immersive_emotion_face import ImmersiveEmotionFace


class EmotionFaceController:
    """
    Contrôleur central pour l'intégration et la gestion du visage émotionnel immersif.

    Ce contrôleur:
    - Coordonne la création et l'initialisation du widget ImmersiveEmotionFace
    - Gère les connexions avec le moteur émotionnel, l'apprentissage et les recommandations
    - Assure les mises à jour périodiques et la synchronisation
    - Fournit une interface simplifiée pour le contrôle du visage
    """

    def __init__(self, screen=None, config=None):
        """
        Initialise le contrôleur du visage émotionnel.

        Args:
            screen: L'écran Kivy contenant le widget (optionnel)
            config: Dictionnaire de configuration (optionnel)
        """
        self.logger = logging.getLogger("jeffrey.emotion_face_controller")

        # Composants connectés
        self.screen = screen  # L'écran contenant le widget
        self.face_widget = None  # Sera initialisé plus tard
        self.emotional_engine = None
        self.emotional_learning = None
        self.recommendation_engine = None

        # Configuration
        self.config = config or {
            "refresh_interval": 0.5,  # Intervalle de rafraîchissement en secondes
            "enabled": True,  # Activer/désactiver le widget
            "immersion_mode": "balanced",  # Mode d'immersion (minimal, balanced, full, adaptive)
        }

        # État interne
        self.is_initialized = False
        self.current_emotion = "neutre"
        self.emotion_intensity = 0.5
        self.update_task = None  # Tâche de mise à jour périodique

        self.logger.info("Contrôleur du visage émotionnel immersif initialisé")

    def initialize(self, screen=None, emotional_learning=None, recommendation_engine=None):
        """
        Initialise le contrôleur avec les composants requis.

        Args:
            screen: L'écran Kivy contenant le widget
            emotional_learning: Instance d'EmotionalLearning
            recommendation_engine: Instance de RecommendationEngine

        Returns:
            bool: True si l'initialisation a réussi, False sinon
        """
        # Mise à jour des références
        if screen:
            self.screen = screen

        # Initialiser le moteur émotionnel si nécessaire
        if not self.emotional_engine:
            self.emotional_engine = EmotionalEngine()

        # Connecter le système d'apprentissage
        if emotional_learning:
            self.emotional_learning = emotional_learning

        # Connecter le moteur de recommandation
        if recommendation_engine:
            self.recommendation_engine = recommendation_engine

        # Vérifier si on peut continuer
        if not self.screen:
            self.logger.error("Impossible d'initialiser: écran non défini")
            return False

        # Créer le widget si ce n'est pas déjà fait
        if not self.face_widget:
            try:
                # Créer le widget immersif
                self.face_widget = ImmersiveEmotionFace(
                    size_hint=(0.3, 0.3),
                    pos_hint={"center_x": 0.5, "center_y": 0.6},
                    emotion=self.current_emotion,
                    intensity=self.emotion_intensity,
                )

                # Connecter aux systèmes
                if self.emotional_engine:
                    self.face_widget.connect_to_emotional_engine(self.emotional_engine)

                if self.emotional_learning:
                    self.face_widget.connect_to_learning_system(self.emotional_learning)

                if self.recommendation_engine:
                    self.face_widget.connect_to_recommendation_engine(self.recommendation_engine)

                # Configurer le mode d'immersion
                self.face_widget.set_immersion_mode(self.config.get("immersion_mode", "balanced"))

                self.logger.info("Widget de visage immersif créé et configuré")
            except Exception as e:
                self.logger.error(f"Erreur lors de la création du widget: {e}")
                return False

        self.is_initialized = True

        # Démarrer les mises à jour périodiques
        self._start_update_cycle()

        return True

    def integrate_to_screen(self, container_id="visual_layer", replace_existing=False):
        """
        Intègre le widget au conteneur spécifié dans l'écran.

        Args:
            container_id: ID du conteneur où ajouter le widget
            replace_existing: Si True, remplace le widget EnergyFace existant

        Returns:
            bool: True si l'intégration a réussi, False sinon
        """
        if not self.is_initialized or not self.face_widget:
            self.logger.error("Le contrôleur doit être initialisé avant l'intégration")
            return False

        # Vérifier si l'écran possède le conteneur spécifié
        if not hasattr(self.screen.ids, container_id):
            self.logger.error(f"Conteneur '{container_id}' non trouvé dans l'écran")
            return False

        container = self.screen.ids[container_id]

        # Si demandé, remplacer le widget EnergyFace existant
        if replace_existing:
            for child in list(container.children):
                if isinstance(child, EnergyFaceWidget):
                    # Sauvegarder l'émotion actuelle avant de supprimer
                    current_emotion = getattr(child, "emotion", self.current_emotion)
                    current_intensity = getattr(child, "intensity", self.emotion_intensity)

                    # Supprimer l'ancien widget
                    container.remove_widget(child)

                    # Mettre à jour l'émotion du nouveau widget
                    self.face_widget.set_emotion(current_emotion, current_intensity)
                    self.logger.info(f"Widget EnergyFace remplacé, émotion: {current_emotion}")
                    break

        # Ajouter le widget au conteneur
        container.add_widget(self.face_widget)
        self.logger.info(f"Widget intégré au conteneur '{container_id}'")

        return True

    def replace_energy_face(self):
        """
        Remplace spécifiquement le widget EnergyFace standard par le widget immersif.

        Returns:
            bool: True si le remplacement a réussi, False sinon
        """
        if not self.is_initialized or not self.face_widget:
            self.logger.error("Le contrôleur doit être initialisé avant le remplacement")
            return False

        # Vérifier si l'écran possède le widget energy_face
        if not hasattr(self.screen.ids, "energy_face"):
            self.logger.error("Widget 'energy_face' non trouvé dans l'écran")
            return False

        # Récupérer la référence à l'ancien widget
        old_widget = self.screen.ids.energy_face

        # Sauvegarder les propriétés importantes
        current_emotion = getattr(old_widget, "emotion", self.current_emotion)
        current_intensity = getattr(old_widget, "intensity", self.emotion_intensity)
        speaking_state = getattr(old_widget, "speaking_state", False)

        # Récupérer le parent de l'ancien widget
        parent = old_widget.parent
        if not parent:
            self.logger.error("Widget parent non trouvé")
            return False

        # Position et taille de l'ancien widget
        old_pos = old_widget.pos
        old_size = old_widget.size
        old_size_hint = old_widget.size_hint
        old_pos_hint = old_widget.pos_hint

        # Supprimer l'ancien widget
        parent.remove_widget(old_widget)

        # Configurer le nouveau widget avec les mêmes propriétés
        self.face_widget.size_hint = old_size_hint
        self.face_widget.pos_hint = old_pos_hint
        self.face_widget.size = old_size
        self.face_widget.pos = old_pos

        # Mettre à jour l'état émotionnel
        self.face_widget.set_emotion(current_emotion, current_intensity)

        # Ajouter le nouveau widget
        parent.add_widget(self.face_widget)

        # Mettre à jour l'ID dans l'écran
        self.screen.ids.energy_face = self.face_widget

        self.logger.info("Widget EnergyFace remplacé par ImmersiveEmotionFace")
        return True

    def update_emotion(self, emotion, intensity=None):
        """
        Met à jour l'émotion affichée par le visage.

        Args:
            emotion: Nouvelle émotion à afficher
            intensity: Intensité de l'émotion (0.0 à 1.0, optionnel)

        Returns:
            bool: True si la mise à jour a réussi, False sinon
        """
        if not self.face_widget:
            self.logger.error("Widget non initialisé")
            return False

        # Mettre à jour l'état interne
        self.current_emotion = emotion
        if intensity is not None:
            self.emotion_intensity = max(0.0, min(1.0, intensity))

        # Mettre à jour le widget
        self.face_widget.set_emotion(emotion, self.emotion_intensity)

        # Informer le système d'apprentissage
        if self.emotional_learning:
            self.emotional_learning.observe_emotion(emotion)

        self.logger.debug(f"Émotion mise à jour: {emotion} (intensité: {self.emotion_intensity:.2f})")
        return True

    def toggle_immersive_mode(self, enabled=None):
        """
        Active ou désactive le mode immersif.

        Args:
            enabled: True pour activer, False pour désactiver, None pour basculer

        Returns:
            bool: Le nouvel état du mode immersif
        """
        if not self.face_widget:
            self.logger.error("Widget non initialisé")
            return False

        # Déterminer la nouvelle valeur
        if enabled is None:
            enabled = not self.config.get("enabled", True)

        # Mettre à jour la configuration
        self.config["enabled"] = enabled

        # Appliquer au widget
        if enabled:
            # Restaurer le mode précédent
            self.face_widget.set_immersion_mode(self.config.get("immersion_mode", "balanced"))
        else:
            # Mode minimal pour désactivation
            self.face_widget.set_immersion_mode("minimal")
            self.face_widget.emotion_transparency = 0.3

        self.logger.info(f"Mode immersif {'activé' if enabled else 'désactivé'}")
        return enabled

    def set_immersion_level(self, level):
        """
        Définit le niveau d'immersion.

        Args:
            level: Niveau d'immersion (0.0 à 1.0)

        Returns:
            float: Le niveau d'immersion appliqué
        """
        if not self.face_widget:
            self.logger.error("Widget non initialisé")
            return 0.0

        # Normaliser la valeur
        level = max(0.0, min(1.0, level))

        # Appliquer au widget
        self.face_widget.immersion_level = level

        return level

    def set_relationship_level(self, level):
        """
        Définit le niveau de relation pour l'affichage.

        Args:
            level: Niveau de relation (0.0 à 1.0)

        Returns:
            float: Le niveau de relation appliqué
        """
        if not self.face_widget:
            self.logger.error("Widget non initialisé")
            return 0.0

        # Normaliser la valeur
        level = max(0.0, min(1.0, level))

        # Appliquer au widget
        self.face_widget.set_relationship_level(level)

        return level

    def trigger_effect(self, effect_name, intensity=0.7, duration=2.0):
        """
        Déclenche un effet visuel spécifique.

        Args:
            effect_name: Nom de l'effet à déclencher
            intensity: Intensité de l'effet (0.0 à 1.0)
            duration: Durée en secondes

        Returns:
            bool: True si l'effet a été déclenché, False sinon
        """
        if not self.face_widget:
            self.logger.error("Widget non initialisé")
            return False

        # Vérifier si l'effet existe
        if not hasattr(self.face_widget, "_trigger_effect"):
            self.logger.error("Méthode de déclenchement d'effet non trouvée")
            return False

        try:
            # Déclencher l'effet
            self.face_widget._trigger_effect(effect_name, duration, intensity)
            return True
        except Exception as e:
            self.logger.error(f"Erreur lors du déclenchement de l'effet '{effect_name}': {e}")
            return False

    def trigger_immersive_scene(self, scene_type, intensity=0.8, duration=5.0):
        """
        Déclenche une scène immersive complète.

        Args:
            scene_type: Type de scène (emotional, relationship, memory, focus)
            intensity: Intensité de la scène (0.0 à 1.0)
            duration: Durée en secondes

        Returns:
            bool: True si la scène a été déclenchée, False sinon
        """
        if not self.face_widget:
            self.logger.error("Widget non initialisé")
            return False

        try:
            # Déclencher la scène immersive
            self.face_widget.trigger_immersive_scene(scene_type, intensity, duration)
            return True
        except Exception as e:
            self.logger.error(f"Erreur lors du déclenchement de la scène '{scene_type}': {e}")
            return False

    def on_speaking_state_change(self, is_speaking):
        """
        Met à jour l'état de parole du visage.

        Args:
            is_speaking: True si Jeffrey est en train de parler, False sinon
        """
        if not self.face_widget:
            return

        # Mettre à jour l'état de parole
        if hasattr(self.face_widget, "speaking_state"):
            self.face_widget.speaking_state = is_speaking

        # Déclencher les animations appropriées
        if is_speaking:
            # Démarrer les animations de parole
            if hasattr(self.face_widget, "on_talk_start"):
                self.face_widget.on_talk_start()
            elif hasattr(self.face_widget, "animate_mouth"):
                self.face_widget.animate_mouth(True)
        else:
            # Arrêter les animations de parole
            if hasattr(self.face_widget, "on_talk_end"):
                self.face_widget.on_talk_end()
            elif hasattr(self.face_widget, "animate_mouth"):
                self.face_widget.animate_mouth(False)

    def _start_update_cycle(self):
        """
        Démarre le cycle de mise à jour périodique du visage.
        """
        # Annuler la tâche existante si présente
        if self.update_task:
            self.update_task.cancel()

        # Démarrer une nouvelle tâche de mise à jour
        interval = self.config.get("refresh_interval", 0.5)
        self.update_task = Clock.schedule_interval(self._update_emotions, interval)

        self.logger.debug(f"Cycle de mise à jour démarré (intervalle: {interval}s)")

    def _update_emotions(self, dt):
        """
        Met à jour périodiquement l'état émotionnel du visage.

        Args:
            dt: Delta temps depuis la dernière mise à jour
        """
        if not self.face_widget or not self.config.get("enabled", True):
            return

        # Vérifier si des mises à jour sont nécessaires depuis les sources externes

        # 1. Mise à jour depuis le moteur émotionnel
        if self.emotional_engine:
            try:
                dominant_emotion = self.emotional_engine.get_dominant_emotion()
                if dominant_emotion and dominant_emotion != self.current_emotion:
                    intensity = self.emotional_engine.get_emotion_intensity(dominant_emotion)
                    self.update_emotion(dominant_emotion, intensity)
            except Exception as e:
                self.logger.warning(f"Erreur lors de la mise à jour depuis le moteur émotionnel: {e}")

        # 2. Mise à jour depuis le système d'apprentissage
        if self.emotional_learning:
            try:
                profile = self.emotional_learning.get_profile()
                dominant_emotions = profile.get("dominant_emotions", [])

                # Ajouter les émotions dominantes pour le mélange
                if dominant_emotions and len(dominant_emotions) > 1:
                    if hasattr(self.face_widget, "add_emotion_to_display"):
                        for emotion in dominant_emotions[:2]:
                            if emotion != self.current_emotion:
                                self.face_widget.add_emotion_to_display(
                                    emotion, as_secondary=(emotion != dominant_emotions[0])
                                )
            except Exception as e:
                self.logger.warning(f"Erreur lors de la mise à jour depuis le système d'apprentissage: {e}")

        # 3. Mise à jour depuis le moteur de recommandation (style visuel)
        if self.recommendation_engine and hasattr(self.recommendation_engine, "get_visual_recommendations"):
            try:
                recommendations = self.recommendation_engine.get_visual_recommendations()
                if recommendations and hasattr(self.face_widget, "_apply_visual_recommendations"):
                    self.face_widget._apply_visual_recommendations(recommendations)
            except Exception as e:
                self.logger.warning(f"Erreur lors de la mise à jour depuis le moteur de recommandation: {e}")

    def shutdown(self):
        """
        Arrête proprement le contrôleur et libère les ressources.
        """
        # Arrêter les mises à jour périodiques
        if self.update_task:
            self.update_task.cancel()
            self.update_task = None

        # Nettoyer les références
        self.face_widget = None
        self.emotional_engine = None
        self.emotional_learning = None
        self.recommendation_engine = None

        self.logger.info("Contrôleur du visage émotionnel arrêté")


# Factory pour faciliter la création
def create_emotion_face_controller(screen=None, config=None):
    """
    Crée et initialise un contrôleur de visage émotionnel.

    Args:
        screen: L'écran Kivy contenant le widget
        config: Dictionnaire de configuration

    Returns:
        EmotionFaceController: Le contrôleur initialisé
    """
    controller = EmotionFaceController(screen, config)

    # Initialiser avec les systèmes disponibles
    try:
        emotional_learning = EmotionalLearning()
        emotional_learning.load_profile()

        from jeffrey.core.ia.recommendation_engine import RecommendationEngine

        recommendation_engine = RecommendationEngine()

        controller.initialize(
            screen=screen,
            emotional_learning=emotional_learning,
            recommendation_engine=recommendation_engine,
        )
    except Exception as e:
        logging.getLogger("jeffrey.factory").error(f"Erreur lors de l'initialisation du contrôleur: {e}")

    return controller
