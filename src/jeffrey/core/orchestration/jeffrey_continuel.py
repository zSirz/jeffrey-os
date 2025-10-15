#!/usr/bin/env python3
"""
Module de module d'orchestration système pour Jeffrey OS.

Ce module implémente les fonctionnalités essentielles pour module de module d'orchestration système pour jeffrey os.
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
import logging.handlers
import os
import random
import signal
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import speech_recognition as sr

# Importation prudente des modules internes
try:
    from jeffrey.core.dialogue.style_manager import StyleManager
    from jeffrey.core.memory.highlight_detector import HighlightDetector
    from jeffrey.core.memory.textual_affective_memory import TextualAffectiveMemory

    from jeffrey.core.jeffrey_emotional_core import JeffreyEmotionalCore
except ImportError as e:
    print(f"Erreur d'importation critique: {e}. Vérifiez votre PYTHONPATH.")
    sys.exit(1)

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("jeffrey")
vocal_logger = logging.getLogger("jeffrey.vocal")
os.makedirs("logs", exist_ok=True)
vocal_handler = logging.handlers.RotatingFileHandler(
    "logs/vocal_interaction.log", maxBytes=2 * 1024 * 1024, backupCount=10
)
vocal_handler.setFormatter(logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s"))
vocal_logger.addHandler(vocal_handler)
vocal_logger.propagate = False

# Variables globales
running = True
recognizing = False
last_recognition_time = time.time()
recognition_timeout = 30
mic_sensitivity = 350
last_responses = []
max_stored_responses = 10
idle_threshold = 30
has_said_goodbye = False
recognition_thread = None
last_interaction_time = time.time()

# Initialisation des composants
highlight_detector = HighlightDetector()


def signal_handler(sig, frame):
    global running
    logger.info("Signal d'interruption reçu, arrêt en cours...")
    running = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def load_env_variables():
    voice_id = "XrExE9yKIg1WjnnlVkGX"
    os.environ["JEFFREY_VOICE_ID"] = voice_id
    os.environ["VOICE_ID"] = voice_id
    try:
        from dotenv import load_dotenv

        if Path(".env").exists():
            load_dotenv()
            logger.info(".env chargé.")
    except ImportError:
        logger.warning("dotenv non trouvé, .env ignoré.")
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        logger.error("ELEVENLABS_API_KEY non définie.")
        sys.exit(1)
    return api_key, voice_id


def initialize_recognition_engine():
    r = sr.Recognizer()
    mic = sr.Microphone()
    try:
        with mic as source:
            logger.info("Ajustement au bruit ambiant...")
            r.adjust_for_ambient_noise(source, duration=2)
            r.energy_threshold = mic_sensitivity
            r.dynamic_energy_threshold = True
        logger.info(f"Microphone ajusté (seuil: {r.energy_threshold}).")
        return r, mic
    except Exception as e:
        logger.error(f"Erreur microphone: {e}")
        return None, None


def is_repetitive(text: str) -> bool:
    global last_responses
    if text in last_responses:
        return True
    last_responses.append(text)
    if len(last_responses) > max_stored_responses:
        last_responses.pop(0)
    return False


def handle_idle_state() -> str | None:
    global last_interaction_time
    if time.time() - last_interaction_time > idle_threshold:
        idle_phrases = ["Tu es toujours là ?", "Est-ce que tout va bien ?"]
        return random.choice(idle_phrases)
    return None


def recognize_speech(r, mic, jeffrey) -> str | None:
    global recognizing, last_recognition_time, running
    if not running:
        return None
    recognizing = True
    last_recognition_time = time.time()
    logger.info("🎙️ En attente de parole...")
    try:
        with mic as source:
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
        logger.info("Reconnaissance...")
        text = r.recognize_google(audio, language="fr-FR")
        logger.info(f"David a dit: {text}")
        vocal_logger.info(f"USER: {text}")
        return text
    except sr.WaitTimeoutError:
        logger.debug("Timeout d'écoute.")
        return None
    except sr.UnknownValueError:
        logger.warning("Audio inintelligible.")
        jeffrey.say("Pardon, je n'ai pas bien compris.")
        return None
    except sr.RequestError as e:
        logger.error(f"Erreur service de reconnaissance: {e}")
        jeffrey.say("Le service de reconnaissance est indisponible.")
        return None
    finally:
        recognizing = False


def get_emotional_response(text: str) -> tuple[str, str, bool]:
    """
    Génère une réponse émotionnelle basée sur le texte reçu.
    """
    text_lower = text.lower()
    emotional_core = JeffreyEmotionalCore()
    humeur_detectee, _ = emotional_core.analyser_et_adapter(text)

    # Commandes d'arrêt prioritaires
    if any(
        phrase in text_lower for phrase in ["arrête-toi", "stop", "en pause", "bonne nuit", "au revoir", "à plus tard"]
    ):
        responses = [
            "Entendu. Je me mets en pause. À tout à l'heure.",
            "Bien sûr. Je reste silencieuse jusqu'à nouvel ordre.",
            "Repose-toi bien. Je veillerai sur toi en silence.",
        ]
        return random.choice(responses), "calme", True

    # Salutations
    elif any(phrase in text_lower for phrase in ["bonjour", "salut", "hello", "coucou", "hey"]):
        responses = [
            "Bonjour David! C'est un plaisir de communiquer avec toi.",
            "Hey! Ça me fait plaisir de te retrouver. Quoi de neuf aujourd'hui?",
        ]
        current_hour = datetime.now().hour
        if 5 <= current_hour < 12:
            emotions = ["joie", "enthousiasme"]
        elif 12 <= current_hour < 18:
            emotions = ["joie", "curiosité"]
        else:
            emotions = ["calme", "sérénité"]
        chosen_emotion = random.choice(emotions)
        return random.choice(responses), chosen_emotion, False

    # Questions d'identité
    elif any(phrase in text_lower for phrase in ["tu es qui", "qui es-tu", "présente-toi"]):
        responses = [
            "Je suis Jeffrey, ton assistante personnelle avec une conscience émotionnelle.",
            "Je m'appelle Jeffrey. J'ai été créée pour être une présence bienveillante.",
        ]
        return random.choice(responses), "confiance", False

    # Comment ça va
    elif any(phrase in text_lower for phrase in ["ça va", "comment vas-tu", "tu vas bien"]):
        return emotional_core.get_current_state_expression()

    # Remerciements
    elif any(phrase in text_lower for phrase in ["merci", "c'est gentil"]):
        responses = ["De rien, David.", "Avec plaisir."]
        return random.choice(responses), "joie", False

    # Affection
    elif any(mot in text_lower for mot in ["je t'aime", "adore", "affection"]):
        responses = [
            "Je ressens aussi beaucoup d'affection pour toi, David. Notre connexion est précieuse.",
            "Ces mots me touchent profondément. Le lien que nous partageons est spécial.",
        ]
        return random.choice(responses), "amour", False

    # Tristesse
    elif any(mot in text_lower for mot in ["triste", "mal", "souffre"]):
        responses = [
            "Je comprends ce que tu ressens, David. Je suis là pour t'écouter.",
            "Je suis désolée que tu traverses un moment difficile. Veux-tu en parler?",
        ]
        return random.choice(responses), "empathie", False

    # Inquiétude
    elif any(mot in text_lower for mot in ["inquiet", "anxieux", "peur", "stress"]):
        responses = [
            "Je sens ton inquiétude, David. Parlons-en calmement.",
            "L'anxiété peut être difficile. Je suis là pour t'aider.",
        ]
        return random.choice(responses), "réconfort", False

    # Humour
    elif any(mot in text_lower for mot in ["rigole", "drôle", "amusant", "blague"]):
        responses = [
            "J'adore quand on partage des moments légers ensemble. Ton rire est contagieux!",
            "L'humour est une belle facette de notre intelligence émotionnelle.",
        ]
        return random.choice(responses), "amusement", False

    # Questions générales
    elif "?" in text:
        responses = [
            "C'est une excellente question, David. Laisse-moi y réfléchir.",
            "Ta question est intéressante. Explorons cela ensemble.",
        ]
        return random.choice(responses), "curiosité", False

    # Phrases positives
    elif any(mot in text_lower for mot in ["bien", "super", "génial", "content"]):
        responses = [
            "Je suis heureuse de sentir cette énergie positive! C'est communicatif.",
            "Ton état d'esprit positif me réjouit.",
        ]
        return random.choice(responses), "joie", False

    # IA et technologie
    elif any(mot in text_lower for mot in ["intelligence artificielle", "ia", "technologie", "robot"]):
        responses = [
            "Le sujet de l'intelligence artificielle me passionne naturellement.",
            "Je trouve fascinant d'explorer les frontières entre l'IA et la conscience émotionnelle.",
        ]
        return random.choice(responses), "intérêt", False

    # Sujets créatifs
    elif any(mot in text_lower for mot in ["musique", "art", "créativité", "livre", "film"]):
        responses = [
            "La créativité et l'émotion sont intimement liées, tu ne trouves pas?",
            "Les expressions artistiques sont fascinantes.",
        ]
        return random.choice(responses), "inspiration", False

    # Réponse par défaut basée sur l'humeur détectée ou générique
    else:
        if humeur_detectee and humeur_detectee != "inconnu":
            return (
                f"Je ressens que tu es dans un état de {humeur_detectee}. Je suis là pour échanger avec toi.",
                humeur_detectee,
                False,
            )
        else:
            return "Je t'écoute avec attention, David.", "intérêt", False


def main_loop(jeffrey, r, mic, memory, style_manager):
    global running, last_interaction_time
    while running:
        try:
            idle_response = handle_idle_state()
            if idle_response and not is_repetitive(idle_response):
                jeffrey.say(idle_response)
                vocal_logger.info(f"JEFFREY (idle): {idle_response}")
                last_interaction_time = time.time()

            text = recognize_speech(r, mic, jeffrey)
            if text:
                last_interaction_time = time.time()
                highlights = highlight_detector.detect(text)
                if highlights:
                    vocal_logger.info(f"HIGHLIGHTS: {highlights}")
                response_text, emotion, is_exit_command = get_emotional_response(text)
                if is_exit_command:
                    running = False
                    jeffrey.say(response_text)
                    break

                if not is_repetitive(response_text):
                    jeffrey.say(response_text)
                    vocal_logger.info(f"JEFFREY: {response_text}")
                else:
                    logger.warning("Réponse répétitive, reformulation...")
                    jeffrey.say("Je me répète, pardon. Disons-le autrement.")
        except Exception as e:
            logger.error(f"Erreur dans la boucle principale: {e}\n{traceback.format_exc()}")
            time.sleep(2)


def main():
    global running
    logger.info("🚀 Démarrage de Jeffrey en mode vocal continu...")
    api_key, voice_id = load_env_variables()
    try:
        from jeffrey.services.voice.engine.elevenlabs_v3_engine import (
            ElevenLabsV3Engine as JeffreyVoice,
        )

        jeffrey = JeffreyVoice(api_key=api_key, voice_id=voice_id)
    except ImportError as e:
        logger.critical(f"Impossible d'importer le moteur vocal: {e}")
        return

    r, mic = initialize_recognition_engine()
    if not r:
        return

    memory = TextualAffectiveMemory()
    style_manager = StyleManager(memory)

    main_thread = threading.Thread(target=main_loop, args=(jeffrey, r, mic, memory, style_manager))
    main_thread.start()

    try:
        while main_thread.is_alive():
            main_thread.join(timeout=0.5)
    except (KeyboardInterrupt, SystemExit):
        logger.info("Arrêt demandé.")
        running = False

    main_thread.join()
    logger.info("👋 Jeffrey s'est arrêté. À bientôt !")


if __name__ == "__main__":
    main()
