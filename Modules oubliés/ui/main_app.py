#!/usr/bin/env python3

"""
Application principale Jeffrey en mode interface graphique ou console.
Ce module gère la logique d'affichage et la coordination entre les composants.
"""

import logging
import time

# Import de l'interface console
from ui.console_ui import get_ui

logger = logging.getLogger(__name__)


class JeffreyApp:
    """Classe principale de l'application Jeffrey."""

    def __init__(self, jeffrey_engine=None, mode="vocal", use_gui=False, debug=False):
        """
        Initialise l'application principale.

        Args:
            jeffrey_engine: Instance du moteur Jeffrey initialisé
                          (Si None, un moteur sera initialisé automatiquement)
            mode (str): Mode de fonctionnement ('vocal', 'texte', 'complet')
                       (Utilisé seulement si jeffrey_engine est None)
            use_gui (bool): Si True, utilise l'interface graphique
                          (Utilisé seulement si jeffrey_engine est None)
            debug (bool): Si True, active le mode debug
        """
        self.mode = mode
        self.use_gui = use_gui
        self.debug = debug
        self.running = False
        self.ui = get_ui()  # Récupérer l'instance singleton de l'UI console
        self.jeffrey_components = jeffrey_engine
        self.recognition_active = False
        self.voice_thread = None

        # État émotionnel simulé (à remplacer par le vrai)
        self.current_emotion = "neutral"
        self.current_intensity = 0.5

        # Configurer l'UI
        self.ui.configure(use_colors=True, debug_mode=debug)

    def initialize(self):
        """Initialise l'application et ses composants."""
        try:
            # Afficher l'écran d'accueil
            self.ui.clear_screen()
            self.ui.display_header()
            self.ui.display_welcome()
            self.ui.display_loading()

            # Charger les composants de Jeffrey
            try:
                from orchestrateur.core_loader import initialize_jeffrey

                # Créer un dictionnaire de configuration
                config = {
                    "voice_enabled": self.mode == "vocal",
                    "offline_mode": False,
                    "ui_mode": "graphical" if self.use_gui else "console",
                    "debug": self.debug,
                    "voice_config": {
                        "sensitivity": 0.7,
                        "wake_word": "Jeffrey",
                        "continuous_listening": True,
                    },
                }
                self.jeffrey_components = initialize_jeffrey(config)

                if not self.jeffrey_components:
                    raise ImportError("Échec d'initialisation des composants")

                self.ui.display_success("Jeffrey initialisé avec succès")
                return True

            except ImportError as e:
                logger.error(f"Erreur d'importation: {e}")
                self.ui.display_error(f"Erreur d'importation: {e}", str(e))
                return False

        except Exception as e:
            logger.error(f"Erreur d'initialisation: {e}")
            self.ui.display_error(f"Erreur critique: {e}", str(e))
            return False

    def start_voice_recognition(self):
        """Démarre la reconnaissance vocale."""
        if not self.jeffrey_components or "start_voice_recognition" not in self.jeffrey_components:
            self.ui.display_warning("Module de reconnaissance vocale non disponible")
            return False

        # Configuration du callback pour la reconnaissance vocale
        def voice_callback(text, confidence):
            self.ui.display_voice_input(text, confidence)
            # À compléter avec le traitement du texte

        # Paramètres de la reconnaissance vocale
        try:
            from Orchestrateur_IA.recognition.jeffrey_voice_recognition import set_recognition_params

            set_recognition_params(
                {
                    "wake_word": "Jeffrey",
                    "continuous_listening": True,
                    "sensitivity": 0.7,
                    "language": "fr-FR",
                    "callback": voice_callback,
                }
            )
        except ImportError:
            # Configuration fallback si le module est importé différemment
            self.ui.display_warning("Configuration vocale standard utilisée")

        # Démarrer la reconnaissance vocale
        self.recognition_active = True
        start_voice = self.jeffrey_components["start_voice_recognition"]
        start_voice()
        self.ui.display_success("Reconnaissance vocale activée")
        return True

    def stop_voice_recognition(self):
        """Arrête la reconnaissance vocale."""
        if not self.jeffrey_components or "stop_voice_recognition" not in self.jeffrey_components:
            return

        if self.recognition_active:
            stop_voice = self.jeffrey_components["stop_voice_recognition"]
            stop_voice()
            self.recognition_active = False
            self.ui.display_info("Reconnaissance vocale désactivée")

    def run(self):
        """Exécute la boucle principale de l'application."""
        if not self.initialize():
            return

        self.running = True

        # Démarrer la reconnaissance vocale si en mode vocal
        if self.mode == "vocal":
            self.start_voice_recognition()

        # Boucle principale simplifiée
        try:
            # En mode vocal, on attend juste des interruptions clavier
            if self.mode == "vocal":
                while self.running:
                    try:
                        time.sleep(0.1)  # Économise le CPU
                    except KeyboardInterrupt:
                        self.running = False
                        break
            # En mode texte, on attend des entrées utilisateur
            else:
                while self.running:
                    user_input = self.ui.get_text_input()
                    if user_input.lower() in ["/exit", "/quit"]:
                        self.running = False
                        break
                    elif user_input.lower() == "/clear":
                        self.ui.clear_screen()
                        self.ui.display_header()
                    elif user_input.lower() == "/help":
                        self.ui.show_help()
                    elif user_input.lower() == "/history":
                        self.ui.show_history()
                    elif user_input.strip():
                        # Traiter le texte utilisateur
                        self.ui.display_user_input(user_input)
                        # Simulation de réponse
                        self.ui.display_thinking()
                        self.ui.display_response(
                            f"Vous avez dit: {user_input}. Désolé, je suis en mode minimal pour le moment.",
                            self.current_emotion,
                            self.current_intensity,
                        )

        except Exception as e:
            logger.error(f"Erreur dans la boucle principale: {e}")
            self.ui.display_error(f"Erreur: {e}", str(e))
        finally:
            self.cleanup()

    def cleanup(self):
        """Nettoie les ressources avant de quitter."""
        # Arrêter la reconnaissance vocale
        self.stop_voice_recognition()

        # Sauvegarder l'état si nécessaire
        logger.info("Sauvegarde de l'état avant de quitter...")

        # Afficher le message d'au revoir
        self.ui.display_goodbye()


def create_app(jeffrey_engine=None, mode="vocal", use_gui=False, debug=False) -> JeffreyApp:
    """
    Crée et configure l'application principale.

    Args:
        jeffrey_engine: Instance du moteur Jeffrey initialisé
        mode: Mode de fonctionnement ('vocal', 'texte', 'complet')
        use_gui: Si True, utilise l'interface graphique
        debug: Si True, active le mode debug

    Returns:
        L'instance de l'application
    """
    app = JeffreyApp(jeffrey_engine=jeffrey_engine, mode=mode, use_gui=use_gui, debug=debug)
    return app


if __name__ == "__main__":
    # Configuration du logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Créer et exécuter l'application
    app = create_app(mode="texte", debug=True)
    app.run()
