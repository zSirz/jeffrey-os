#!/usr/bin/env python3

"""
Interface console pour Jeffrey.
Gère l'affichage et l'interaction en mode texte.
"""

import logging
import os
import time
from datetime import datetime

logger = logging.getLogger(__name__)


# Couleurs pour la console
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    END = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class ConsoleUI:
    """Interface utilisateur console pour Jeffrey."""

    def __init__(self):
        """Initialise l'interface console."""
        self.use_colors = True
        self.debug_mode = False
        self.history = []
        self.max_history = 100
        self.last_response_time = None

    def configure(self, use_colors: bool = True, debug_mode: bool = False):
        """Configure l'interface."""
        self.use_colors = use_colors
        self.debug_mode = debug_mode

    def _format_text(self, text: str, color: str = None, bold: bool = False, underline: bool = False) -> str:
        """Formate le texte avec couleurs et styles."""
        if not self.use_colors:
            return text

        result = text
        if color:
            result = color + result + Colors.END
        if bold:
            result = Colors.BOLD + result + Colors.END
        if underline:
            result = Colors.UNDERLINE + result + Colors.END

        return result

    def clear_screen(self):
        """Efface l'écran de la console."""
        os.system("cls" if os.name == "nt" else "clear")

    def display_header(self):
        """Affiche l'en-tête Jeffrey."""
        jeffrey_header = """
================================================================================

       __         ______   _______   _______   _______   _______  __      __
      |  |       |   ___| |   ____| |   ____| |   ____| |   ____||  |    |  |
      |  |       |  |__   |  |__    |  |__    |  |__    |  |__   |  |    |  |
      |  |       |   __|  |   __|   |   __|   |   __|   |   __|  |  |    |  |
      |  |_____  |  |___  |  |      |  |      |  |____  |  |     |  |___ |  |_____
      |________| |______| |__|      |__|      |_______| |__|     |______||________|

                      Assistant IA Émotionnel V1.1 - Mode Vocal

================================================================================
"""
        print(self._format_text(jeffrey_header, Colors.BLUE))

    def display_welcome(self):
        """Affiche le message de bienvenue."""
        welcome_message = "\n🎤 Reconnaissance vocale activée - Jeffrey vous écoute\n"
        print(self._format_text(welcome_message, Colors.GREEN, bold=True))

        help_messages = [
            "💡 Dites 'Jeffrey' pour l'activer, puis parlez normalement",
            "💡 Dites 'Arrête-toi' pour mettre en pause l'écoute",
            "💡 Dites 'Au revoir' pour terminer la session",
        ]

        for message in help_messages:
            print(self._format_text(message, Colors.CYAN))

        print()  # Ligne vide

    def display_loading(self, message: str = "Chargement des modules Jeffrey..."):
        """Affiche un message de chargement."""
        print(self._format_text(f"🔄 {message}", Colors.BLUE))

    def display_error(self, error_message: str, detailed_error: str = None):
        """Affiche un message d'erreur."""
        print(self._format_text(f"❌ {error_message}", Colors.RED))

        if detailed_error and self.debug_mode:
            print(self._format_text(f"   Détails: {detailed_error}", Colors.RED))

        print(self._format_text("   Assurez-vous que toutes les dépendances sont installées:", Colors.YELLOW))
        print(self._format_text("   pip install -r requirements.txt", Colors.YELLOW))
        print()

    def display_success(self, message: str):
        """Affiche un message de succès."""
        print(self._format_text(f"✅ {message}", Colors.GREEN))

    def display_info(self, message: str):
        """Affiche un message d'information."""
        print(self._format_text(f"ℹ️ {message}", Colors.CYAN))

    def display_warning(self, message: str):
        """Affiche un message d'avertissement."""
        print(self._format_text(f"⚠️ {message}", Colors.YELLOW))

    def display_response(self, response: str, emotion: str = None, intensity: float = None):
        """Affiche une réponse de Jeffrey."""
        # Marquer le timestamp de la réponse
        self.last_response_time = datetime.now()

        # Préparer l'en-tête avec informations émotionnelles
        header = "Jeffrey"
        if emotion:
            emotion_display = emotion.capitalize()
            intensity_display = f" ({intensity:.1f})" if intensity is not None else ""
            header += f" [{emotion_display}{intensity_display}]"

        header += ": "

        # Définir la couleur en fonction de l'émotion
        color = Colors.CYAN
        if emotion:
            emotion_colors = {
                "joie": Colors.GREEN,
                "happy": Colors.GREEN,
                "joy": Colors.GREEN,
                "tristesse": Colors.BLUE,
                "sad": Colors.BLUE,
                "sadness": Colors.BLUE,
                "colère": Colors.RED,
                "angry": Colors.RED,
                "anger": Colors.RED,
                "peur": Colors.YELLOW,
                "fear": Colors.YELLOW,
                "surprised": Colors.YELLOW,
                "surprise": Colors.YELLOW,
            }
            color = emotion_colors.get(emotion.lower(), Colors.CYAN)

        # Afficher la réponse
        print(self._format_text(header, color, bold=True) + self._format_text(response, color))
        print()  # Ligne vide après la réponse

        # Ajouter à l'historique
        self.history.append(
            {
                "type": "response",
                "content": response,
                "emotion": emotion,
                "intensity": intensity,
                "timestamp": self.last_response_time,
            }
        )

        # Limiter la taille de l'historique
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

    def display_user_input(self, input_text: str):
        """Affiche l'entrée de l'utilisateur."""
        print(self._format_text("Vous: ", Colors.GREEN, bold=True) + input_text)

        # Ajouter à l'historique
        self.history.append({"type": "user_input", "content": input_text, "timestamp": datetime.now()})

    def display_voice_input(self, input_text: str, confidence: float = None):
        """Affiche l'entrée vocale de l'utilisateur."""
        confidence_display = f" ({confidence:.1f})" if confidence is not None else ""
        print(self._format_text(f"🎤 Vous{confidence_display}: ", Colors.GREEN, bold=True) + input_text)

        # Ajouter à l'historique
        self.history.append(
            {
                "type": "voice_input",
                "content": input_text,
                "confidence": confidence,
                "timestamp": datetime.now(),
            }
        )

    def display_thinking(self):
        """Affiche une animation de réflexion."""
        print(self._format_text("Jeffrey réfléchit...", Colors.BLUE), end="", flush=True)
        time.sleep(0.5)
        for _ in range(3):
            print(self._format_text(".", Colors.BLUE), end="", flush=True)
            time.sleep(0.3)
        print()  # Nouvelle ligne

    def display_listening(self):
        """Affiche un indicateur d'écoute."""
        print(self._format_text("🎤 En écoute...", Colors.GREEN))

    def display_goodbye(self):
        """Affiche un message d'au revoir."""
        print("\n" + self._format_text("👋 Merci d'avoir utilisé Jeffrey V1.1!", Colors.GREEN, bold=True))

    def get_text_input(self, prompt: str = "Votre message: ") -> str:
        """Récupère l'entrée texte de l'utilisateur."""
        try:
            user_input = input(self._format_text(prompt, Colors.GREEN, bold=True))
            return user_input
        except (KeyboardInterrupt, EOFError):
            print("\n" + self._format_text("Interruption détectée. Au revoir!", Colors.YELLOW))
            return "/exit"

    def show_help(self):
        """Affiche l'aide de Jeffrey."""
        help_text = """
Aide de Jeffrey V1.1
===================

Commandes disponibles:
  /help         - Affiche cette aide
  /exit, /quit  - Quitte Jeffrey
  /clear        - Efface l'écran
  /debug        - Active/désactive le mode debug
  /history      - Affiche l'historique des conversations

En mode vocal:
  - Dites "Jeffrey" pour activer l'écoute
  - Dites "Arrête-toi" pour mettre en pause l'écoute
  - Dites "Au revoir" pour terminer la session
"""
        print(self._format_text(help_text, Colors.CYAN))

    def show_history(self, count: int = 10):
        """Affiche l'historique des dernières interactions."""
        if not self.history:
            print(self._format_text("Aucun historique disponible.", Colors.YELLOW))
            return

        print(self._format_text("\nHistorique des dernières interactions:", Colors.BLUE, bold=True))
        history_to_show = self.history[-count:] if count < len(self.history) else self.history

        for i, entry in enumerate(history_to_show):
            timestamp = entry["timestamp"].strftime("%H:%M:%S")

            if entry["type"] == "user_input":
                print(f"{timestamp} | " + self._format_text("Vous: ", Colors.GREEN, bold=True) + entry["content"])
            elif entry["type"] == "voice_input":
                confidence = f" ({entry.get('confidence', 0.0):.1f})" if "confidence" in entry else ""
                print(
                    f"{timestamp} | "
                    + self._format_text(f"🎤 Vous{confidence}: ", Colors.GREEN, bold=True)
                    + entry["content"]
                )
            elif entry["type"] == "response":
                emotion = (
                    f" [{entry.get('emotion', '').capitalize()}]" if "emotion" in entry and entry["emotion"] else ""
                )
                print(
                    f"{timestamp} | "
                    + self._format_text(f"Jeffrey{emotion}: ", Colors.CYAN, bold=True)
                    + entry["content"]
                )

        print()  # Ligne vide


# Instance singleton pour l'UI
console_ui = ConsoleUI()


def get_ui() -> ConsoleUI:
    """Retourne l'instance singleton de l'UI."""
    return console_ui


if __name__ == "__main__":
    # Test de l'interface console
    ui = ConsoleUI()
    ui.clear_screen()
    ui.display_header()
    ui.display_welcome()
    ui.display_loading()
    ui.display_success("Modules chargés avec succès")
    ui.display_info("Prêt à converser")

    # Simuler une conversation
    ui.display_voice_input("Bonjour Jeffrey, comment vas-tu aujourd'hui?", 0.85)
    ui.display_thinking()
    ui.display_response(
        "Bonjour ! Je vais très bien, merci de demander. Comment puis-je vous aider aujourd'hui ?",
        "joie",
        0.7,
    )

    ui.display_voice_input("Raconte-moi une blague", 0.92)
    ui.display_thinking()
    ui.display_response(
        "Pourquoi les plongeurs plongent-ils toujours en arrière et jamais en avant ? Parce que sinon ils tomberaient dans le bateau !",
        "happy",
        0.8,
    )

    ui.display_voice_input("Au revoir", 0.95)
    ui.display_response("C'était un plaisir de discuter avec vous. À bientôt !", "peaceful", 0.6)
    ui.display_goodbye()
