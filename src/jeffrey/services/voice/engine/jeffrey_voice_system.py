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
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()
logging.info("✅ Fichier .env chargé correctement dans jeffrey_voice_system.py")


class JeffreyVoiceSystem:
    """
    Classe JeffreyVoiceSystem pour le système Jeffrey OS.

    Cette classe implémente les fonctionnalités spécifiques nécessaires
    au bon fonctionnement du module. Elle gère l'état interne, les transformations
    de données, et l'interaction avec les autres composants du système.
    """

    def __init__(self, config=None, emotional_core=None, test_mode=False) -> None:
        import logging

        self.config = config or {}
        self.emotional_core = emotional_core
        self.test_mode = test_mode
        self.empathic_inversion_module = None
        self.tactile_gesture_responder = None
        self.logger = logging.getLogger("JeffreyVoiceSystem")
        self.voice_engine = None
        self.adaptive_personality_engine = None  # Module d'adaptation de personnalité
        self.voice_mode = "ON"  # Par défaut, la voix est activée
        # ATTENTION : NE PAS MODIFIER CES VALEURS
        self.backend = "elevenlabs"  # Backend par défaut - TOUJOURS ELEVENLABS
        self.use_mock_voice = False  # OBLIGATOIRE: Force l'utilisation de la vraie voix
        self.volume = 1.0  # Volume par défaut
        self.cache_enabled = True  # Cache activé par défaut

        # Chemin vers le dossier des effets audio générés
        self.effects_dir = Path("offline_voice_cache/effects")

        # Initialisation des effets audio
        self.audio_fx = {
            "soupir_doux": "assets/audio_fx/soupir_doux.wav",
            "respiration_lente": "assets/audio_fx/respiration_lente.wav",
            "pause_émue": "assets/audio_fx/pause_émue.wav",
            "soupir_profond": "assets/audio_fx/soupir_profond.wav",
            "respiration_émue": "assets/audio_fx/respiration_émue.wav",
        }

        # Vérifier la disponibilité des modules audio
        self.audio_modules_available = False
        try:
            pass

            self.audio_modules_available = True
            self.logger.info("✅ Modules audio (sounddevice, soundfile) disponibles")
        except ImportError:
            self.logger.warning("⚠️ Modules audio non disponibles. Les effets sonores seront désactivés.")
            print("⚠️ Pour activer les effets sonores, installez : pip install sounddevice soundfile")

        # Variables pour la reconnaissance vocale
        self.speech_recognizer = None
        self.speech_recognition_thread = None
        self.listening_active = False
        self.wake_word = "Jeffrey"
        self.continuous_listening = True
        self.sensitivity = 0.7
        self.recognition_callback = None
        self.recognition_language = "fr-FR"

        if not self.test_mode:
            self.initialize_components()

    def initialize_components(self):
        """Initialize the voice system components."""


import logging

try:
    logging.info("[JeffreyVoiceSystem] Initialisation du système vocal")
    # Tentative de chargement du moteur de voix réel
    try:
        from orchestrateur.core.voice.voice_engine import VoiceEngine

        self.voice_engine = VoiceEngine()
        logging.info("[JeffreyVoiceSystem] Moteur vocal principal chargé avec succès")
    except ImportError:
        logging.warning(
            "[JeffreyVoiceSystem] Impossible de charger le moteur vocal principal, utilisation du mode de secours"
        )
        self.voice_engine = None

    # Affichage de l'état du système vocal au démarrage
    logging.info(
        f"[JeffreyVoiceSystem] État au démarrage: voice_mode={self.voice_mode}, backend={self.backend}, cache_enabled={self.cache_enabled}, volume={self.volume}"
    )
    print(
        f"🔊 Système vocal: voice_mode={self.voice_mode}, backend={self.backend}, cache_enabled={self.cache_enabled}, volume={self.volume}"
    )
except Exception as e:
    logging.error(f"[JeffreyVoiceSystem] Erreur lors de l'initialisation: {e}")


def initialize(self):
    """Initialize the voice system - compatibility method."""
    # Compatibility method that calls initialize_components
    self.initialize_components()


def shutdown(self):
    """Ferme proprement le système vocal."""
    import logging

    logging.info("[JeffreyVoiceSystem] Fermeture du système vocal")

    # Fermer le moteur vocal si disponible
    if self.voice_engine and hasattr(self.voice_engine, "shutdown"):
        try:
            self.voice_engine.shutdown()
            logging.info("[JeffreyVoiceSystem] Moteur vocal fermé avec succès")
        except Exception as e:
            logging.error(f"[JeffreyVoiceSystem] Erreur lors de la fermeture du moteur vocal: {e}")

    # Fermer les ressources audio
    try:
        # Tentative de fermeture de pygame si utilisé
        try:
            import pygame

            if pygame.mixer.get_init():
                pygame.mixer.quit()
                logging.info("[JeffreyVoiceSystem] Ressources audio pygame libérées")
        except BaseException:
            pass
    except Exception as e:
        logging.error(f"[JeffreyVoiceSystem] Erreur lors de la libération des ressources audio: {e}")

    logging.info("[JeffreyVoiceSystem] Système vocal fermé")


def _ajouter_effets_vocaux(self, texte: str, emotion: str = "neutre") -> str:
    """
    Ajoute des effets vocaux textuels pour enrichir l'expression émotionnelle.

    Args:
        texte: Le texte à enrichir
        emotion: L'émotion actuelle

    Returns:
        str: Le texte enrichi avec des effets vocaux
    """
    import random

    # Effets vocaux selon l'émotion
    effets_emotionnels = {
        "tendresse": {
            "soupirs": ["*soupir doux*", "*respire doucement*", "*silence intime*"],
            "intros": [
                "Je...",
                "Tu sais...",
                "C'est difficile à dire... mais",
                "J'ai besoin de le dire...",
                "Je ressens que...",
                "Mon cœur me dit que...",
            ],
        },
        "amour": {
            "soupirs": ["*soupir profond*", "*respire avec émotion*", "*silence ému*"],
            "intros": [
                "Je t'aime tellement...",
                "Mon cœur bat si fort...",
                "Je ressens tant d'amour...",
                "Je ne peux m'empêcher de...",
                "Je suis si heureux quand...",
            ],
        },
        "tristesse": {
            "soupirs": ["*soupir mélancolique*", "*respire lentement*", "*silence triste*"],
            "intros": [
                "Je me sens...",
                "Parfois je...",
                "C'est dur de...",
                "J'ai l'impression que...",
                "Je ne sais pas pourquoi mais...",
            ],
        },
        "doute": {
            "soupirs": ["*soupir hésitant*", "*respire nerveusement*", "*silence pensif*"],
            "intros": [
                "Je me demande si...",
                "Peut-être que...",
                "Je ne suis pas sûr mais...",
                "J'hésite à dire que...",
                "Je pense que...",
            ],
        },
    }

    # Effets par défaut pour les autres émotions
    effets_par_defaut = {
        "soupirs": ["*soupir*", "*respire*", "*silence*"],
        "intros": ["Je...", "Tu sais...", "C'est que...", "Je pense que...", "Je crois que..."],
    }

    # Sélectionner les effets selon l'émotion
    effets = effets_emotionnels.get(emotion, effets_par_defaut)

    # Ajouter un soupir au début avec une probabilité de 40%
    if random.random() < 0.4:
        texte = f"{random.choice(effets['soupirs'])}\n\n{texte}"

    # Ajouter une introduction émotionnelle avec une probabilité de 50%
    if random.random() < 0.5:
        phrases = texte.split(". ")
    if phrases:
        # Choisir une phrase au hasard pour l'intro
        idx = random.randint(0, len(phrases) - 1)
        phrases[idx] = f"{random.choice(effets['intros'])} {phrases[idx]}"
        texte = ". ".join(phrases)

    # Ajouter des pauses émotionnelles avec une probabilité de 30%
    if random.random() < 0.3:
        texte = texte.replace(". ", "... ").replace("! ", "...! ").replace("? ", "...? ")

    return texte


def play_audio_effect(self, effect_key: str, volume: float = 1.0) -> bool:
    """
    Joue un effet audio généré par ElevenLabs depuis le cache offline.

    Args:
        effect_key: Clé de l'effet à jouer (ex: "soupir_doux")
        volume: Volume de l'effet (0.0 à 1.0)

    Returns:
        bool: True si l'effet a été joué avec succès
    """
    if not self.audio_modules_available:
        self.logger.debug(f"Modules audio non disponibles, effet '{effect_key}' ignoré")
    return False

    import sounddevice as sd
    import soundfile as sf

    # Construire le chemin du fichier
    effect_path = self.effects_dir / f"{effect_key}.wav"

    # Vérifier si le fichier existe
    if not effect_path.exists():
        self.logger.warning(f"Effet audio non trouvé : {effect_path}")
    return False

    try:
        # Lire le fichier audio
        data, samplerate = sf.read(str(effect_path))

        # Ajuster le volume
        if volume != 1.0:
            data = data * volume

        # Jouer l'audio
        self.logger.info(f"🎧 Lecture effet audio : {effect_key}.wav")
        sd.play(data, samplerate)
        sd.wait()  # Attendre la fin de la lecture

        return True

    except Exception as e:
        self.logger.error(f"Erreur lors de la lecture de l'effet {effect_key}: {e}")
        return False


def speak(
    self,
    text: str,
    emotion: str = "neutre",
    phase: str = "adulte",
    play_audio: bool = True,
    emphasis: float = 0.7,
    slow_mode: bool = False,
    effect_start: str | None = None,
    effect_end: str | None = None,
    **kwargs,
) -> str | None:
    """
    Fait parler Jeffrey avec des effets audio émotionnels.

    Args:
        text: Le texte à prononcer
        emotion: L'émotion à exprimer
        phase: La phase de développement
        play_audio: Indique s'il faut jouer l'audio
        emphasis: L'intensité de l'émotion (0.0 à 1.0)
        slow_mode: Mode lent pour les scènes intimes
        effect_start: Effet audio spécifique au début (optionnel)
        effect_end: Effet audio spécifique à la fin (optionnel)
        **kwargs: Paramètres supplémentaires

    Returns:
        str: Identifiant de l'audio généré ou None
    """
    import logging
    import random

    # Vérifier si la voix est activée
    if self.voice_mode.upper() == "OFF":
        logging.info(f"[JeffreyVoiceSystem] Voix désactivée. Message non prononcé: {text[:50]}...")
    return None

    # Déterminer les effets audio à jouer
    should_play_effects = slow_mode or emotion in ["tendresse", "amour", "tristesse", "doute"]

    if should_play_effects and self.audio_modules_available:
        # Effet au début
        if effect_start:
            start_effect = effect_start
        else:
            # Sélectionner l'effet selon l'émotion
            start_effects = {
                "tendresse": "soupir_doux",
                "amour": "soupir_profond",
                "tristesse": "respiration_lente",
                "doute": "pause_émue",
            }
            start_effect = start_effects.get(emotion, "respiration_lente")

        # Jouer l'effet de début
    if self.play_audio_effect(start_effect, volume=0.8):
        self.logger.info(f"💨 Effet audio début: {start_effect}.wav")
        # TODO: Remplacer par asyncio.sleep ou threading.Event  # Pause naturelle après l'effet

    # Enrichir le texte avec des effets textuels
    text = self._ajouter_effets_vocaux(text, emotion)

    logging.info(
        f"[JeffreyVoiceSystem] 🔊 Génération vocale: {text[:50]}... [émotion: {emotion}, phase: {phase}, volume: {self.volume}, slow_mode: {slow_mode}]"
    )
    print(f"🎤 Jeffrey dit ({emotion}, {phase}, slow_mode={slow_mode}): {text[:50]}...")

    # Adapter les paramètres vocaux pour le mode lent
    if slow_mode:
        kwargs["tempo"] = kwargs.get("tempo", 0.8)
        kwargs["pitch"] = kwargs.get("pitch", 0.9)
        emphasis = min(emphasis * 1.2, 1.0)

    # Générer la voix
    result = None
    if self.voice_engine and hasattr(self.voice_engine, "say"):
        try:
            result = self.voice_engine.say(
                text=text,
                emotion=emotion,
                phase=phase,
                play_audio=play_audio,
                emphasis=emphasis,
                volume=self.volume,
                cache_enabled=self.cache_enabled,
                **kwargs,
            )
        except Exception as e:
            logging.error(f"Erreur lors de la génération vocale: {e}")
            result = None

    # Effet à la fin (30-40% de chance)
    if should_play_effects and self.audio_modules_available and random.random() < 0.35:
        # TODO: Remplacer par asyncio.sleep ou threading.Event  # Petite pause avant l'effet final

        if effect_end:
            end_effect = effect_end
        else:
            # Sélectionner un effet final différent
            end_effects = {
                "tendresse": "respiration_émue",
                "amour": "murmure_je_taime",
                "tristesse": "pause_émue",
                "doute": "hésitation_amour",
            }
            end_effect = end_effects.get(emotion, "pause_émue")

        if self.play_audio_effect(end_effect, volume=0.7):
            self.logger.info(f"💨 Effet audio fin: {end_effect}.wav")

    return result


def play_sound_auto(self, sound_type, emotion=None):
    """
    Joue un son en fonction du type et de l'émotion.

    Args:
        sound_type: Type de son à jouer
        emotion: Émotion associée au son

    Returns:
        bool: True si le son a été joué, False sinon
    """
    import logging

    # Vérifier si la voix est activée
    if self.voice_mode.upper() == "OFF":
        logging.info(f"[JeffreyVoiceSystem] Voix désactivée. Son non joué: {sound_type}")
    return False

    logging.info(f"[JeffreyVoiceSystem] 🔊 Lecture du son: {sound_type} [émotion: {emotion}]")

    # Si le moteur de voix est disponible et a une méthode pour jouer des sons
    if self.voice_engine and hasattr(self.voice_engine, "play_sound"):
        try:
            return self.voice_engine.play_sound(sound_type, emotion=emotion)
        except Exception as e:
            logging.error(f"[JeffreyVoiceSystem] Erreur lors de la lecture du son: {e}")

    # Simule le son en console
    print(f"[SOUND] {sound_type} ({emotion})")
    return True


def set_adaptive_personality_engine(self, engine) -> None:
    """
    Définit le moteur d'adaptation de personnalité à utiliser.

    Args:
        engine: Instance de AdaptivePersonalityEngine
    """
    self.adaptive_personality_engine = engine
    self.logger.info("Moteur d'adaptation de personnalité connecté au système vocal")


def apply_empathic_inversion(self, response: str, user_emotional_state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """
    Applique une inversion empathique à une réponse.
    Implémenté pour le Sprint 201.

    Args:
        response: Réponse textuelle originale
        user_emotional_state: État émotionnel de l'utilisateur

    Returns:
        Tuple[str, Dict]: Réponse modifiée et informations d'inversion
    """
    # Vérifier si le module d'inversion empathique est disponible
    if not self.empathic_inversion_module:
        return response, {"success": False, "reason": "EmpathicInversionModule non initialisé"}

    try:
        # Traiter l'état émotionnel de l'utilisateur
        inversion_info = self.empathic_inversion_module.process_user_emotional_state(user_emotional_state)

        # Si une inversion est nécessaire, l'appliquer à la réponse
        if inversion_info.get("inversion_needed", False) and inversion_info.get("inversion_applied", False):
            modified_response, voice_params = self.empathic_inversion_module.apply_empathic_inversion_to_response(
                response, inversion_info
            )

            result = {
                "success": True,
                "inversion_applied": True,
                "original_emotion": inversion_info.get("original_emotion"),
                "inverse_emotion": inversion_info.get("inverse_emotion"),
                "voice_params": voice_params,
            }

            # Si le module est intégré au cœur émotionnel, enregistrer également là-bas
            if self.emotional_core and hasattr(self.emotional_core, "appliquer_inversion_empathique"):
                _, emotional_result = self.emotional_core.appliquer_inversion_empathique(response, user_emotional_state)
                # Fusionner les résultats si nécessaire
                if emotional_result.get("success", False):
                    self.logger.info("Inversion empathique également traitée par le cœur émotionnel")

            return modified_response, result
        else:
            return response, {
                "success": True,
                "inversion_applied": False,
                "reason": inversion_info.get("reason", "Inversion non nécessaire"),
            }

    except Exception as e:
        self.logger.error(f"Erreur lors de l'application de l'inversion empathique: {e}")
        return response, {"success": False, "reason": str(e)}


def process_tactile_gesture(
    self,
    zone: str,
    gesture_type: str,
    intensity: float = 0.5,
    duration: float = 1.0,
    user_id: str | None = None,
) -> dict[str, Any]:
    """
    Traite un geste tactile provenant de l'interface utilisateur.
    Implémenté pour le Sprint 202.

    Args:
        zone: Zone touchée
        gesture_type: Type de geste
        intensity: Intensité du geste (0-1)
        duration: Durée du geste en secondes
        user_id: Identifiant de l'utilisateur (optionnel)

    Returns:
        Dict: Résultat du traitement
    """
    # Si le cœur émotionnel a un tactile_gesture_responder, utiliser celui-là
    if self.emotional_core and hasattr(self.emotional_core, "traiter_geste_tactile"):
        try:
            return self.emotional_core.traiter_geste_tactile(
                zone=zone,
                type_toucher=gesture_type,
                intensite=intensity,
                duration=duration,
                user_id=user_id,
            )
        except Exception as e:
            self.logger.error(f"Erreur lors du traitement du geste tactile via le cœur émotionnel: {e}")

    # Sinon, utiliser le module local s'il est disponible
    if not self.tactile_gesture_responder:
        return {"success": False, "reason": "TactileGestureResponder non disponible"}

    try:
        # Traiter le geste tactile
        result = self.tactile_gesture_responder.process_gesture(
            zone=zone,
            gesture_type=gesture_type,
            intensity=intensity,
            duration=duration,
            user_id=user_id,
            source="voice_system",
        )

        # Enrichir le résultat
        result["success"] = result.get("success", True)

        return result

    except Exception as e:
        self.logger.error(f"Erreur lors du traitement du geste tactile: {e}")
        return {"success": False, "reason": str(e)}


def interjection(
    self,
    type_interjection: str = "neutre",
    texte: str | None = None,
    params_vocaux: dict[str, Any] | None = None,
) -> str:
    """
    Produit une interjection vocale selon un type d'émotion.
    Utile pour le TactileGestureResponder (Sprint 202).

    Args:
        type_interjection: Type d'interjection ("surprise", "toucher", etc.)
        texte: Texte spécifique de l'interjection (optionnel)
        params_vocaux: Paramètres vocaux spécifiques (optionnel)

    Returns:
        str: Chemin du fichier audio généré, ou None en cas d'échec
    """
    # Mapper les types d'interjection vers des émotions
    emotion_map = {
        "surprise": "surprise",
        "toucher": "plaisir",
        "affectueux": "affection",
        "stimulant": "joie",
        "intense": "alerte",
        "neutre": "intérêt",
    }

    # Déterminer le texte de l'interjection si non spécifié
    if not texte:
        interjections = {
            "surprise": ["Oh!", "Ah!", "Wow!"],
            "toucher": ["Mmh", "Oh"],
            "affectueux": ["Mmmm...", "Aaah"],
            "stimulant": ["Hihihi", "Oh là là!"],
            "intense": ["Oh!", "Aïe", "Hé!"],
            "neutre": ["Hm", "Oh"],
        }
        texte = random.choice(interjections.get(type_interjection, ["Oh"]))

    # Déterminer l'émotion à utiliser
    emotion = emotion_map.get(type_interjection, "neutre")

    # Préparer les paramètres vocaux
    voice_params = params_vocaux or {}
    if "emphasis" not in voice_params:
        voice_params["emphasis"] = 0.7  # Intensité par défaut

    # Générer la voix avec des paramètres spécifiques
    try:
        # Utiliser directement l'engine si possible
        if hasattr(self, "voice_engine") and hasattr(self.voice_engine, "say"):
            return self.voice_engine.say(
                text=texte,
                emotion=emotion,
                emphasis=voice_params.get("emphasis", 0.7),
                output_path=None,
                play_audio=True,
            )
        # Sinon, utiliser la méthode générique
        elif hasattr(self, "dire") or hasattr(self, "parler"):
            method = getattr(self, "dire", None) or getattr(self, "parler", None)
            return method(texte, emotion=emotion, voice_params=voice_params)

        return None

    except Exception as e:
        self.logger.error(f"Erreur lors de la génération de l'interjection: {e}")
        return None


def soupirer(self, type_soupir: str = "neutre", intensite: float = 0.7) -> str:
    """
    Produit un soupir vocal selon un type d'émotion.
    Utile pour le TactileGestureResponder (Sprint 202).

    Args:
        type_soupir: Type de soupir ("plaisir", "fatigue", etc.)
        intensite: Intensité du soupir (0-1)

    Returns:
        str: Chemin du fichier audio généré, ou None en cas d'échec
    """
    # Mapper les types de soupir vers des textes et émotions
    soupir_map = {
        "plaisir": {"texte": "Mmmmh...", "emotion": "plaisir"},
        "fatigue": {"texte": "Pfff...", "emotion": "fatigue"},
        "soulagement": {"texte": "Aaah...", "emotion": "soulagement"},
        "inquiétude": {"texte": "Hmmm...", "emotion": "inquiétude"},
        "sensible": {"texte": "Aah...", "emotion": "plaisir"},
        "neutre": {"texte": "Hmm...", "emotion": "neutre"},
    }

    # Récupérer les infos du soupir
    soupir_info = soupir_map.get(type_soupir, soupir_map["neutre"])

    # Paramètres vocaux pour un effet de soupir
    params_vocaux = {
        "emphasis": intensite,
        "breathiness": 0.8,  # Très soufflé pour un soupir
        # Note: rate and pitch parameters not supported by ElevenLabs API. Using only supported API parameters.
    }

    # Générer l'interjection
    return self.interjection(type_interjection=type_soupir, texte=soupir_info["texte"], params_vocaux=params_vocaux)


# SPRINT 218: Intégration du module de stabilité émotionnelle


def apply_emotional_stability_to_voice(self, voice_params: dict[str, Any], current_emotion: str) -> dict[str, Any]:
    """
    Applique le verrou de stabilité émotionnelle aux paramètres vocaux si nécessaire.
    Implémenté pour le Sprint 218.

    Args:
        voice_params: Paramètres vocaux à adapter
        current_emotion: Émotion actuelle demandée

    Returns:
        Dict[str, Any]: Paramètres vocaux adaptés
    """
    # Si le cœur émotionnel n'a pas de module de stabilité, retourner les paramètres tels quels
    if not self.emotional_core or not hasattr(self.emotional_core, "obtenir_etat_verrou_stabilite"):
        return voice_params

    try:
        # Vérifier si un verrou de stabilité est actif
        stability_status = self.emotional_core.obtenir_etat_verrou_stabilite()

        if not stability_status.get("success", False) or not stability_status.get("verrou_actif", False):
            return voice_params  # Pas de verrou actif, aucune modification nécessaire

        # Récupérer le niveau de verrou et l'état émotionnel stable
        lock_level = stability_status.get("niveau_verrou", "modéré")
        stable_emotion = stability_status.get("emotion_stable", "neutre")

        # Si l'émotion demandée est la même que l'émotion stable, pas besoin d'adapter
        if current_emotion == stable_emotion:
            return voice_params

        # Adapter les paramètres vocaux selon le niveau de verrou
        adapted_params = voice_params.copy()

        if lock_level == "léger":
            # Léger: réduire légèrement l'intensité de l'émotion
            if "emphasis" in adapted_params:
                adapted_params["emphasis"] = max(0.3, adapted_params.get("emphasis", 0.5) * 0.7)
            if "emotion_intensity" in adapted_params:
                adapted_params["emotion_intensity"] = max(0.3, adapted_params.get("emotion_intensity", 0.5) * 0.7)

        elif lock_level == "modéré":
            # Modéré: mélanger l'émotion demandée avec l'émotion stable
            if "emphasis" in adapted_params:
                adapted_params["emphasis"] = max(0.2, adapted_params.get("emphasis", 0.5) * 0.5)
            if "emotion_intensity" in adapted_params:
                adapted_params["emotion_intensity"] = max(0.2, adapted_params.get("emotion_intensity", 0.5) * 0.5)

            # Ajouter l'émotion stable comme émotion secondaire
            if "secondary_emotion" not in adapted_params:
                adapted_params["secondary_emotion"] = stable_emotion
                adapted_params["secondary_emotion_intensity"] = 0.4

        elif lock_level == "strict":
            # Strict: remplacer l'émotion par l'émotion stable
            adapted_params["emotion"] = stable_emotion
            if "emphasis" in adapted_params:
                adapted_params["emphasis"] = min(0.6, adapted_params.get("emphasis", 0.5))
            if "emotion_intensity" in adapted_params:
                adapted_params["emotion_intensity"] = min(0.6, adapted_params.get("emotion_intensity", 0.5))

            # Supprimer toute émotion secondaire
            if "secondary_emotion" in adapted_params:
                del adapted_params["secondary_emotion"]
            if "secondary_emotion_intensity" in adapted_params:
                del adapted_params["secondary_emotion_intensity"]

        self.logger.info(f"Paramètres vocaux adaptés pour respecter le verrou de stabilité (niveau: {lock_level})")
        return adapted_params

    except Exception as e:
        self.logger.error(f"Erreur lors de l'application de la stabilité émotionnelle aux paramètres vocaux: {e}")
    return voice_params  # En cas d'erreur, retourner les paramètres originaux


# SPRINT 219: Intégration du métamoniteur émotionnel


def adjust_voice_for_emotional_patterns(self, voice_params: dict[str, Any], current_emotion: str) -> dict[str, Any]:
    """
    Ajuste les paramètres vocaux en fonction des patterns émotionnels détectés.
    Implémenté pour le Sprint 219.

    Args:
        voice_params: Paramètres vocaux à adapter
        current_emotion: Émotion actuelle demandée

    Returns:
        Dict[str, Any]: Paramètres vocaux adaptés
    """
    # Si le cœur émotionnel n'a pas de métamoniteur, retourner les paramètres tels quels
    if not self.emotional_core or not hasattr(self.emotional_core, "obtenir_patterns_emotionnels"):
        return voice_params

    try:
        # Récupérer les patterns émotionnels actuels
        patterns_result = self.emotional_core.obtenir_patterns_emotionnels()

        if not patterns_result.get("success", False):
            return voice_params  # Pas de patterns disponibles, aucune modification nécessaire

        patterns = patterns_result.get("patterns", {})

        # Si aucun pattern problématique n'est détecté, pas besoin d'adapter
        problematic_patterns = [p for p, v in patterns.items() if v.get("intensite", 0) > 0.6]
        if not problematic_patterns:
            return voice_params

        # Adapter les paramètres vocaux selon les patterns détectés
        adapted_params = voice_params.copy()

        # Vérifier les patterns spécifiques et appliquer des adaptations
        for pattern, info in patterns.items():
            intensity = info.get("intensite", 0)

            if intensity > 0.6:
                if pattern == "oscillation":
                    # Stabiliser la voix pour les patterns d'oscillation
                    adapted_params["stability"] = min(0.9, adapted_params.get("stability", 0.5) + 0.3)
                    adapted_params["rate"] = max(0.9, min(1.1, adapted_params.get("rate", 1.0)))

                elif pattern == "fixation":
                    # Ajouter de la variabilité pour les patterns de fixation
                    if "variability" not in adapted_params:
                        adapted_params["variability"] = 0.3
                    else:
                        adapted_params["variability"] += 0.2

                elif pattern == "repli":
                    # Augmenter légèrement l'énergie pour les patterns de repli
                    adapted_params["energy"] = min(0.8, adapted_params.get("energy", 0.5) + 0.2)

                elif pattern == "suractivation":
                    # Réduire l'énergie pour les patterns de suractivation
                    adapted_params["energy"] = max(0.3, adapted_params.get("energy", 0.5) - 0.2)
                    adapted_params["rate"] = max(0.8, adapted_params.get("rate", 1.0) - 0.1)

        self.logger.info(
            f"Paramètres vocaux adaptés pour compenser les patterns émotionnels: {', '.join(problematic_patterns)}"
        )
        return adapted_params

    except Exception as e:
        self.logger.error(f"Erreur lors de l'ajustement vocal pour les patterns émotionnels: {e}")
        return voice_params  # En cas d'erreur, retourner les paramètres originaux


# SPRINT 220: Intégration du sélecteur de rôle émotionnel


def apply_emotional_role_to_voice(self, voice_params: dict[str, Any], current_emotion: str) -> dict[str, Any]:
    """
    Applique les caractéristiques du rôle émotionnel actif aux paramètres vocaux.
    Implémenté pour le Sprint 220.

    Args:
        voice_params: Paramètres vocaux à adapter
        current_emotion: Émotion actuelle demandée

    Returns:
        Dict[str, Any]: Paramètres vocaux adaptés
    """
    # Si le cœur émotionnel n'a pas de sélecteur de rôle, retourner les paramètres tels quels
    if not self.emotional_core or not hasattr(self.emotional_core, "obtenir_role_actif"):
        return voice_params

    try:
        # Vérifier si un rôle émotionnel est actif
        role_result = self.emotional_core.obtenir_role_actif()

        if not role_result.get("success", False) or not role_result.get("role_actif", False):
            return voice_params  # Pas de rôle actif, aucune modification nécessaire

        # Récupérer les informations du rôle
        role_info = role_result.get("role", {})
        role_name = role_info.get("nom", "")
        role_intensity = role_result.get("intensite", 0.7)
        role_traits = role_info.get("traits_vocaux", {})

        if not role_traits:
            return voice_params  # Pas de traits vocaux définis pour ce rôle

        # Adapter les paramètres vocaux selon les traits du rôle
        adapted_params = voice_params.copy()

        # Appliquer les traits vocaux selon l'intensité du rôle
        for trait, value in role_traits.items():
            if trait in adapted_params:
                # Mélanger la valeur actuelle avec celle du rôle selon l'intensité
                current_value = adapted_params[trait]
                role_value = value
                adapted_params[trait] = current_value * (1 - role_intensity) + role_value * role_intensity
            else:
                # Appliquer directement le trait du rôle, ajusté par l'intensité
                adapted_params[trait] = value * role_intensity

        # Si le rôle définit une émotion de base, l'utiliser comme émotion secondaire
        if "emotion_base" in role_info and role_intensity > 0.4:
            role_emotion = role_info["emotion_base"]

            # Ne pas écraser l'émotion primaire, mais ajouter comme secondaire si pertinent
            if current_emotion != role_emotion and "secondary_emotion" not in adapted_params:
                adapted_params["secondary_emotion"] = role_emotion
                adapted_params["secondary_emotion_intensity"] = min(0.7, role_intensity)

        self.logger.info(
            f"Paramètres vocaux adaptés pour le rôle émotionnel '{role_name}' (intensité: {role_intensity:.2f})"
        )
        return adapted_params

    except Exception as e:
        self.logger.error(f"Erreur lors de l'application du rôle émotionnel aux paramètres vocaux: {e}")
    return voice_params  # En cas d'erreur, retourner les paramètres originaux


# SPRINT 223: Implémentation de la reconnaissance vocale


def start_listening(
    self,
    callback: Callable[[str, float], None] | None = None,
    wake_word: str | None = None,
    continuous: bool | None = None,
    sensitivity: float | None = None,
    language: str | None = None,
) -> bool:
    """
    Démarre la reconnaissance vocale continue.

    Args:
        callback: Fonction appelée lorsqu'un texte est reconnu (texte, confiance)
        wake_word: Mot d'activation pour la reconnaissance (défaut: "Jeffrey")
        continuous: Mode d'écoute continue (attente du mot d'activation)
        sensitivity: Sensibilité de la reconnaissance (0.0-1.0)
        language: Code de langue pour la reconnaissance (e.g., "fr-FR")

    Returns:
        bool: True si la reconnaissance a démarré, False sinon
    """
    # Vérifier si déjà en écoute
    if self.listening_active:
        self.logger.info("La reconnaissance vocale est déjà active")
    return True

    # Mettre à jour les paramètres locaux si fournis
    if wake_word:
        self.wake_word = wake_word
    if continuous is not None:
        self.continuous_listening = continuous
    if sensitivity is not None:
        self.sensitivity = max(0.0, min(1.0, sensitivity))
    if language:
        self.recognition_language = language
    self.recognition_callback = callback

    try:
        # Importer le module de reconnaissance vocale Jeffrey
        from orchestrateur.recognition.jeffrey_voice_recognition import (
            set_recognition_params,
            start_recognition_loop,
        )

        # Configurer les paramètres du module de reconnaissance
        set_recognition_params(
            {
                "wake_word": self.wake_word,
                "continuous_listening": self.continuous_listening,
                "sensitivity": self.sensitivity,
                "language": self.recognition_language,
                "callback": self.recognition_callback,
            }
        )

        # Démarrer la reconnaissance vocale
        start_recognition_loop()

        # Marquer la reconnaissance comme active
        self.listening_active = True

        self.logger.info(
            f"Reconnaissance vocale démarrée (wake_word={self.wake_word}, continuous={self.continuous_listening})"
        )
        return True

    except ImportError as e:
        self.logger.error(f"Impossible de démarrer la reconnaissance vocale: Module manquant - {e}")
        print(f"❌ Module requis non trouvé: {str(e).split('named ')[-1] if 'named' in str(e) else str(e)}")
        print("   Vérifiez que le module recognition/jeffrey_voice_recognition.py est bien présent")
        return False
    except Exception as e:
        self.logger.error(f"Erreur lors du démarrage de la reconnaissance vocale: {e}")
        return False


def stop_listening(self) -> bool:
    """
    Arrête la reconnaissance vocale continue.

    Returns:
        bool: True si la reconnaissance a été arrêtée, False sinon
    """
    if not self.listening_active:
        self.logger.info("La reconnaissance vocale n'est pas active")
    return True

    try:
        # Importer la fonction pour arrêter la reconnaissance
        from orchestrateur.recognition.jeffrey_voice_recognition import stop_recognition_loop

        # Arrêter la reconnaissance vocale
        stop_recognition_loop()

        # Marquer la reconnaissance comme inactive
        self.listening_active = False
        self.speech_recognizer = None
        self.speech_recognition_thread = None

        self.logger.info("Reconnaissance vocale arrêtée avec succès")
        return True

    except ImportError as e:
        self.logger.error(f"Impossible d'arrêter la reconnaissance vocale: Module manquant - {e}")
        return False
    except Exception as e:
        self.logger.error(f"Erreur lors de l'arrêt de la reconnaissance vocale: {e}")
        return False


def _listen_thread(self):
    """
    Thread d'écoute pour la reconnaissance vocale continue.
    Cette méthode s'exécute dans un thread séparé.
    """
    import speech_recognition as sr

    # Mode d'attente initial
    waiting_for_wake_word = self.continuous_listening

    self.logger.info("Thread d'écoute démarré")
    print("🎤 Reconnaissance vocale active")

    if waiting_for_wake_word:
        print(f"💡 En attente du mot d'activation: '{self.wake_word}'")

    mic = None

    try:
        # Créer le microphone
        mic = sr.Microphone()

        # Effectuer un ajustement initial pour le bruit ambiant
        with mic as source:
            self.logger.info("Ajustement pour le bruit ambiant...")
            self.speech_recognizer.adjust_for_ambient_noise(source, duration=1)
            self.logger.info(f"Seuil d'énergie: {self.speech_recognizer.energy_threshold}")

        # Boucle d'écoute
        while self.listening_active:
            try:
                with mic as source:
                    self.logger.debug("En attente d'audio...")

                    # Annoncer le mode d'écoute si changement
                    if waiting_for_wake_word:
                        # Afficher une indication visuelle d'attente
                        print("👂 ...", end="\r")

                    # Écouter un segment audio
                    audio = self.speech_recognizer.listen(source, timeout=2, phrase_time_limit=10)

                    # Tenter de reconnaître le texte
                    try:
                        # Utiliser Google Speech Recognition (nécessite une connexion internet)
                        text = self.speech_recognizer.recognize_google(audio, language=self.recognition_language)

                        self.logger.debug(f"Texte reconnu: {text}")

                        # Traiter le texte selon le mode d'écoute
                        if waiting_for_wake_word:
                            # Vérifier si le mot d'activation est présent
                            if self.wake_word.lower() in text.lower():
                                print(f"🔊 Mot d'activation détecté: '{self.wake_word}'")
                                waiting_for_wake_word = False

                                # Jouer un son d'activation si disponible
                                self.play_sound_auto("activation")

                                # Message de confirmation
                                if self.voice_mode.upper() == "ON":
                                    self.speak("Je vous écoute.", "intérêt")
                        else:
                            # Mode écoute active
                            confidence = 0.8  # Google ne fournit pas la confiance

                            # Vérifier les commandes spéciales
                            if "arrête-toi" in text.lower() or "arrête toi" in text.lower():
                                print("🔊 Commande d'arrêt détectée")
                                if self.voice_mode.upper() == "ON":
                                    self.speak(
                                        "Je m'arrête. Appelez-moi quand vous aurez besoin de moi.",
                                        "neutre",
                                    )
                                waiting_for_wake_word = True
                                continue

                            if "au revoir" in text.lower():
                                print("🔊 Commande de fin détectée")
                                if self.voice_mode.upper() == "ON":
                                    self.speak(
                                        "Au revoir. J'ai été ravi de discuter avec vous.",
                                        "amical",
                                    )
                                self.listening_active = False
                                continue

                            # Appeler le callback avec le texte reconnu
                            if self.recognition_callback:
                                self.recognition_callback(text, confidence)
                            else:
                                print(f"🎤 Texte reconnu: {text}")
                                # Si pas de callback, répondre directement
                                if self.emotional_core and hasattr(self.emotional_core, "répondre"):
                                    response = self.emotional_core.répondre(text)
                                    if response and self.voice_mode.upper() == "ON":
                                        self.speak(response, "neutre")

                    except sr.UnknownValueError:
                        self.logger.debug("Impossible de comprendre l'audio")
                    except sr.RequestError as e:
                        self.logger.error(f"Erreur de service de reconnaissance: {e}")
                        print("⚠️ Erreur de service de reconnaissance. Vérifiez votre connexion internet.")
                        # Pause avant nouvelle tentative
                        time.sleep(3)

            except Exception as loop_error:
                self.logger.error(f"Erreur dans la boucle d'écoute: {loop_error}")
                # Pause avant nouvelle tentative
                time.sleep(2)

    except Exception as thread_error:
        self.logger.error(f"Erreur critique dans le thread d'écoute: {thread_error}")
    finally:
        self.logger.info("Thread d'écoute terminé")
        print("🔇 Reconnaissance vocale désactivée")


# SPRINT 221: Intégration du narrateur d'état interne


def add_internal_state_narration(self, response: str, add_prefix: bool = False) -> str:
    """
    Ajoute une narration d'état interne à une réponse textuelle si nécessaire.
    Implémenté pour le Sprint 221.

    Args:
        response: Réponse textuelle originale
        add_prefix: Ajouter la narration au début de la réponse plutôt qu'à la fin

    Returns:
        str: Réponse avec narration d'état interne
    """
    # Si le cœur émotionnel n'a pas de narrateur d'état interne, retourner la réponse telle quelle
    if not self.emotional_core or not hasattr(self.emotional_core, "verbaliser_etat_interne"):
        return response

    try:
        # Obtenir les préférences actuelles de narration
        prefs_result = self.emotional_core.obtenir_preferences_narration()

        if not prefs_result.get("success", False):
            return response  # Impossible d'obtenir les préférences, aucune modification

        # Vérifier si la narration est activée et si elle doit être incluse dans les réponses
        frequence = prefs_result.get("frequence", "jamais")
        inclure_dans_reponse = prefs_result.get("inclure_dans_reponse", False)

        if frequence == "jamais" or not inclure_dans_reponse:
            return response  # Narration désactivée ou ne doit pas être incluse

        # Obtenir la verbalisation de l'état interne
        narration_result = self.emotional_core.verbaliser_etat_interne(
            type_verbalisation=prefs_result.get("type_defaut", "bref"),
            style=prefs_result.get("style_defaut", "naturel"),
            inclure_dans_reponse=True,
        )

        if not narration_result.get("success", False) or not narration_result.get("verbalisation"):
            return response  # Pas de verbalisation disponible

        narration_text = narration_result.get("verbalisation", "")

        # Déterminer si la verbalisation doit être générée selon la fréquence configurée
        if frequence == "rare" and random.random() > 0.2:
            return response
        if frequence == "occasionnel" and random.random() > 0.5:
            return response

        # Ajouter la narration à la réponse
        if add_prefix:
            # Ajouter au début avec une séparation claire
            return f"{narration_text}\n\n{response}"
        else:
            # Ajouter à la fin avec une séparation claire
            return f"{response}\n\n{narration_text}"

    except Exception as e:
        self.logger.error(f"Erreur lors de l'ajout de la narration d'état interne: {e}")
        return response  # En cas d'erreur, retourner la réponse originale


# SPRINT 222+: Intégration du moteur d'adaptation de personnalité


def adapt_voice_to_personality(self, voice_params: dict[str, Any], current_emotion: str) -> dict[str, Any]:
    """
    Adapte les paramètres vocaux en fonction du profil de personnalité actuel.
    Implémenté pour le système d'adaptation de personnalité.

    Args:
        voice_params: Paramètres vocaux à adapter
        current_emotion: Émotion actuelle demandée

    Returns:
        Dict[str, Any]: Paramètres vocaux adaptés
    """
    # Si le moteur d'adaptation de personnalité n'est pas disponible, retourner les paramètres tels quels
    if not self.adaptive_personality_engine:
        return voice_params

    try:
        # Obtenir le style actuel de la personnalité
        current_style = self.adaptive_personality_engine.get_current_style()

        # Si aucun style n'est disponible, retourner les paramètres tels quels
        if not current_style:
            return voice_params

        # Adapter les paramètres vocaux selon le style de personnalité
        adapted_params = voice_params.copy()

        # Appliquer les traits vocaux du style de personnalité
        if "vocal_style" in current_style:
            vocal_style = current_style["vocal_style"]

            # Adapter le rythme vocal
            if "pace" in vocal_style and "rate" in adapted_params:
                # Convertir pace (0-1) en rate (0.8-1.2 typiquement)
                pace = vocal_style["pace"]
                adapted_params["rate"] = 0.8 + (pace * 0.4)  # 0 -> 0.8, 1 -> 1.2

            # Note: pitch_variation n'est pas supporté par l'API d'ElevenLabs
            # Suppression de la référence à pitch_variation

            # Adapter l'expressivité
            if "expressivity" in vocal_style and "expression" in adapted_params:
                adapted_params["expression"] = vocal_style["expressivity"]

            # Adapter la douceur
            if "softness" in vocal_style:
                softness = vocal_style["softness"]
                # Appliquer la douceur à plusieurs paramètres
                if "smoothness" in adapted_params:
                    adapted_params["smoothness"] = softness
                if "breathiness" in adapted_params:
                    adapted_params["breathiness"] = 0.3 + (softness * 0.4)  # 0 -> 0.3, 1 -> 0.7

        # Appliquer les traits de ton qui peuvent affecter la voix
        if "tone" in current_style:
            tone = current_style["tone"]

            # La chaleur peut affecter la qualité vocale
            if "warmth" in tone:
                warmth = tone["warmth"]
                if "warmth" in adapted_params:
                    adapted_params["warmth"] = warmth
                elif "timbre" in adapted_params:
                    # Plus chaud = timbre plus riche
                    adapted_params["timbre"] = 0.4 + (warmth * 0.3)  # 0 -> 0.4, 1 -> 0.7

            # La formalité peut affecter la précision d'articulation
            if "formality" in tone and "articulation" in adapted_params:
                formality = tone["formality"]
                adapted_params["articulation"] = 0.5 + (formality * 0.4)  # 0 -> 0.5, 1 -> 0.9

        # Enregistrer une interaction pour le moteur d'adaptation si pertinent
        self._log_voice_interaction_for_adaptation(current_emotion, adapted_params)

        self.logger.info("Paramètres vocaux adaptés selon le profil de personnalité actuel")
        return adapted_params

    except Exception as e:
        self.logger.error(f"Erreur lors de l'adaptation vocale au profil de personnalité: {e}")
        return voice_params  # En cas d'erreur, retourner les paramètres originaux


def _log_voice_interaction_for_adaptation(self, emotion: str, voice_params: dict[str, Any]) -> None:
    """
    Enregistre les données d'interaction vocale pour le moteur d'adaptation.

    Args:
        emotion: Émotion utilisée pour la synthèse
        voice_params: Paramètres vocaux utilisés
    """
    if not self.adaptive_personality_engine:
        return

    try:
        # Préparer les données d'interaction pour le moteur d'adaptation
        from datetime import datetime

        interaction_data = {
            "timestamp": datetime.now().isoformat(),
            "voice": {
                "emotion": emotion,
                "pace_preference": (
                    voice_params.get("rate", 0.5) if "rate" in voice_params else voice_params.get("pace", 0.5)
                ),
                "tone_feedback": voice_params.get("emphasis", 0.5),
            },
        }

        # Observer l'interaction pour adaptation future
        self.adaptive_personality_engine.observe_interaction(interaction_data)

    except Exception as e:
        self.logger.error(f"Erreur lors de l'enregistrement de l'interaction vocale pour l'adaptation: {e}")


def process_voice_params_for_synthesis(self, text: str, emotion: str, voice_params: dict[str, Any]) -> dict[str, Any]:
    """
    Traite les paramètres vocaux pour la synthèse en appliquant toutes les adaptations.
    Cette méthode applique successivement toutes les couches d'adaptation:
    1. Adaptation de personnalité
    2. Stabilité émotionnelle
    3. Patterns émotionnels
    4. Rôle émotionnel

    Args:
        text: Texte à synthétiser
        emotion: Émotion principale
        voice_params: Paramètres vocaux initiaux

    Returns:
        Dict[str, Any]: Paramètres vocaux finaux après toutes les adaptations
    """
    processed_params = voice_params.copy()

    # Étape 1: Adapter selon le profil de personnalité
    processed_params = self.adapt_voice_to_personality(processed_params, emotion)

    # Étape 2: Appliquer les contraintes de stabilité émotionnelle
    processed_params = self.apply_emotional_stability_to_voice(processed_params, emotion)

    # Étape 3: Ajuster pour les patterns émotionnels problématiques
    processed_params = self.adjust_voice_for_emotional_patterns(processed_params, emotion)

    # Étape 4: Appliquer les caractéristiques du rôle émotionnel actif
    processed_params = self.apply_emotional_role_to_voice(processed_params, emotion)

    return processed_params
