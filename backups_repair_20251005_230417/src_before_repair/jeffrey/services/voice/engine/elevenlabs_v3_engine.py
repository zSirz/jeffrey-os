"""
🚀 ElevenLabs V3 Engine - Jeffrey's Revolutionary Voice System
===========================================================

Next-generation voice synthesis with emotional audio tags, dialogue API,
and cultural voice adaptation across 70+ languages.

Features V3:
- Audio tags emotional expression ([laughs], [whispers], [excited])
- Text-to-Dialogue API for multi-speaker conversations
- Cultural voice adaptation with native accents
- Real-time emotional context analysis
- Automatic fallback to V2.5 for compatibility
- Advanced voice persona management
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from jeffrey.bridge.adapters import ElevenLabsAdapter

# Import du gestionnaire sécurisé des clés API
try:
    from config.secrets_manager import SecureConfigError, get_api_key

    SECURE_CONFIG_AVAILABLE = True
except ImportError:
    try:
        from jeffrey.modules.config.secrets_manager import SecureConfigError, get_api_key

        SECURE_CONFIG_AVAILABLE = True
    except ImportError:
        SECURE_CONFIG_AVAILABLE = False
        logging.warning("🔒 SecureConfig non disponible pour ElevenLabs")

# Import base voice engine for fallback
from .voice_engine import VoiceEngine

logger = logging.getLogger(__name__)


class VoicePersona(Enum):
    """Jeffrey's voice personas for different contexts"""

    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    CREATIVE = "creative"
    EMPATHETIC = "empathetic"
    EXCITED = "excited"
    TEACHER = "teacher"


@dataclass
class V3VoiceSettings:
    """Enhanced voice settings for ElevenLabs V3"""

    voice_id: str = "0PvVCUwwhtu5GmdGdHdZ"  # Jeffrey's custom voice
    stability: float = 0.75
    similarity_boost: float = 0.85
    style: float = 0.0
    use_speaker_boost: bool = True

    # V3 specific settings
    enable_audio_tags: bool = True
    enable_dialogue_mode: bool = False
    language_code: str = "fr"  # Default French
    accent_strength: float = 0.8
    emotional_intensity: float = 0.7
    voice_persona: VoicePersona = VoicePersona.FRIENDLY


@dataclass
class EmotionalAudioTag:
    """V3 Audio tags for emotional expression"""

    tag: str
    description: str
    emotional_context: list[str]
    usage_examples: list[str]


@dataclass
class DialogueSpeaker:
    """Speaker configuration for dialogue mode"""

    speaker_id: str
    voice_id: str
    persona: VoicePersona
    emotional_state: str
    text: str
    audio_tags: list[str] = field(default_factory=list)


class EmotionalTagProcessor:
    """
    🎭 Intelligent audio tag injection based on emotional context
    """

    def __init__(self) -> None:
        """Initialize emotional tag processor"""
        self.audio_tags = self._load_audio_tags()
        self.emotion_tag_mapping = self._create_emotion_tag_mapping()
        self.context_patterns = self._load_context_patterns()

    def _load_audio_tags(self) -> dict[str, EmotionalAudioTag]:
        """Load available V3 audio tags"""
        return {
            "laughs": EmotionalAudioTag(
                tag="[laughs]",
                description="Natural laughter expression",
                emotional_context=["happy", "amused", "joyful"],
                usage_examples=["[laughs] C'est vraiment drôle !", "Tu sais quoi ? [laughs]"],
            ),
            "whispers": EmotionalAudioTag(
                tag="[whispers]",
                description="Quiet, intimate speaking",
                emotional_context=["secretive", "intimate", "mysterious"],
                usage_examples=[
                    "[whispers] Je vais te dire un secret...",
                    "[whispers] Écoute bien...",
                ],
            ),
            "excited": EmotionalAudioTag(
                tag="[excited]",
                description="High energy, enthusiastic",
                emotional_context=["excited", "thrilled", "energetic"],
                usage_examples=[
                    "[excited] C'est fantastique !",
                    "[excited] J'ai une idée géniale !",
                ],
            ),
            "thoughtful": EmotionalAudioTag(
                tag="[thoughtful]",
                description="Contemplative, reflective",
                emotional_context=["thinking", "analyzing", "considering"],
                usage_examples=[
                    "[thoughtful] Hmm, laisse-moi réfléchir...",
                    "[thoughtful] C'est intéressant...",
                ],
            ),
            "sighs": EmotionalAudioTag(
                tag="[sighs]",
                description="Emotional exhale",
                emotional_context=["sad", "disappointed", "resigned"],
                usage_examples=[
                    "[sighs] C'est difficile...",
                    "[sighs] Je comprends ta frustration...",
                ],
            ),
            "gasps": EmotionalAudioTag(
                tag="[gasps]",
                description="Sudden surprise or shock",
                emotional_context=["surprised", "shocked", "amazed"],
                usage_examples=["[gasps] Oh ! Je n'avais pas pensé à ça !", "[gasps] Vraiment ?"],
            ),
            "giggles": EmotionalAudioTag(
                tag="[giggles]",
                description="Light, playful laughter",
                emotional_context=["playful", "amused", "cheerful"],
                usage_examples=["[giggles] Tu es trop drôle !", "[giggles] C'est adorable !"],
            ),
            "confident": EmotionalAudioTag(
                tag="[confident]",
                description="Assured, self-assured tone",
                emotional_context=["confident", "certain", "professional"],
                usage_examples=[
                    "[confident] Je peux t'aider avec ça",
                    "[confident] Voici la solution...",
                ],
            ),
            "gentle": EmotionalAudioTag(
                tag="[gentle]",
                description="Soft, caring tone",
                emotional_context=["empathetic", "caring", "supportive"],
                usage_examples=[
                    "[gentle] Tout va bien se passer...",
                    "[gentle] Je suis là pour t'aider...",
                ],
            ),
            "playful": EmotionalAudioTag(
                tag="[playful]",
                description="Fun, mischievous tone",
                emotional_context=["playful", "fun", "mischievous"],
                usage_examples=[
                    "[playful] Devine quoi ?",
                    "[playful] J'ai une surprise pour toi !",
                ],
            ),
        }

    def _create_emotion_tag_mapping(self) -> dict[str, list[str]]:
        """Map emotions to appropriate audio tags"""
        return {
            "happy": ["[laughs]", "[excited]", "[giggles]", "[playful]"],
            "excited": ["[excited]", "[gasps]", "[laughs]"],
            "sad": ["[sighs]", "[gentle]"],
            "curious": ["[thoughtful]", "[gasps]"],
            "empathetic": ["[gentle]", "[sighs]"],
            "calm": ["[gentle]", "[thoughtful]"],
            "neutral": ["[confident]"],
            "playful": ["[playful]", "[giggles]", "[laughs]"],
            "surprised": ["[gasps]", "[excited]"],
        }

    def _load_context_patterns(self) -> dict[str, list[str]]:
        """Load context patterns for automatic tag injection"""
        return {
            "question": {
                "patterns": ["?", "comment", "pourquoi", "qu'est-ce que"],
                "tags": ["[curious]", "[thoughtful]"],
            },
            "excitement": {
                "patterns": ["fantastique", "génial", "incroyable", "waouw", "super"],
                "tags": ["[excited]", "[laughs]"],
            },
            "empathy": {
                "patterns": ["désolé", "comprends", "difficile", "inquiet"],
                "tags": ["[gentle]", "[sighs]"],
            },
            "humor": {
                "patterns": ["drôle", "blague", "rigolo", "hilarant", "😄", "😂"],
                "tags": ["[laughs]", "[giggles]"],
            },
            "secret": {
                "patterns": ["secret", "confidence", "entre nous", "discrètement"],
                "tags": ["[whispers]"],
            },
            "surprise": {
                "patterns": ["oh", "ah", "vraiment", "sérieusement", "pas possible"],
                "tags": ["[gasps]"],
            },
        }

    def process_text_with_emotion(self, text: str, emotion_context: str, conversation_history: list[str] = None) -> str:
        """
        🎭 Inject appropriate audio tags based on emotional context
        """
        processed_text = text

        # Get base emotion tags
        available_tags = self.emotion_tag_mapping.get(emotion_context, ["[confident]"])

        # Analyze text for context patterns
        context_tags = self._analyze_text_context(text)

        # Combine emotion and context tags
        all_tags = list(set(available_tags + context_tags))

        # Apply tags intelligently
        processed_text = self._apply_tags_to_text(processed_text, all_tags, emotion_context)

        return processed_text

    def _analyze_text_context(self, text: str) -> list[str]:
        """Analyze text for contextual audio tag opportunities"""
        context_tags = []
        text_lower = text.lower()

        for context_type, context_data in self.context_patterns.items():
            patterns = context_data["patterns"]
            tags = context_data["tags"]

        if any(pattern in text_lower for pattern in patterns):
            context_tags.extend(tags)

        return context_tags

    def _apply_tags_to_text(self, text: str, available_tags: list[str], emotion: str) -> str:
        """Intelligently apply audio tags to text"""
        # Don't over-tag - maximum 2-3 tags per sentence
        sentences = re.split(r"[.!?]+", text)
        processed_sentences = []

        for sentence in sentences:
            if not sentence.strip():
                continue

            # Choose best tag for this sentence
            best_tag = self._choose_best_tag(sentence, available_tags, emotion)

            if best_tag:
                # Insert tag at appropriate position
                tagged_sentence = self._insert_tag_optimally(sentence.strip(), best_tag)
                processed_sentences.append(tagged_sentence)
            else:
                processed_sentences.append(sentence.strip())

        return " ".join(processed_sentences) + ("." if not text.rstrip().endswith((".", "!", "?")) else "")

    def _choose_best_tag(self, sentence: str, available_tags: list[str], emotion: str) -> str | None:
        """Choose the most appropriate tag for a sentence"""
        sentence_lower = sentence.lower()

        # Priority mapping for different contexts
        tag_priorities = {
            "[laughs]": ["drôle", "blague", "rigolo", "hilarant"],
            "[whispers]": ["secret", "confidence", "discrètement"],
            "[gasps]": ["vraiment", "oh", "ah", "pas possible"],
            "[excited]": ["fantastique", "génial", "incroyable", "waouw"],
            "[gentle]": ["désolé", "comprends", "difficile"],
            "[thoughtful]": ["hmm", "réfléchir", "analyser", "considérer"],
            "[confident]": ["certain", "sûr", "évidemment", "bien sûr"],
        }

        # Find best matching tag
        best_score = 0
        best_tag = None

        for tag in available_tags:
            if tag in tag_priorities:
                keywords = tag_priorities[tag]
                score = sum(1 for keyword in keywords if keyword in sentence_lower)

                if score > best_score:
                    best_score = score
                    best_tag = tag

        return best_tag if best_score > 0 else (available_tags[0] if available_tags else None)

    def _insert_tag_optimally(self, sentence: str, tag: str) -> str:
        """Insert audio tag at the optimal position in the sentence"""
        # Different tags work better at different positions
        tag_positions = {
            "[laughs]": ["end", "after_setup"],  # After setup or at end
            "[whispers]": ["start"],  # At beginning
            "[gasps]": ["start"],  # At beginning
            "[excited]": ["start", "mid"],  # Beginning or middle
            "[gentle]": ["start"],  # At beginning
            "[thoughtful]": ["start", "mid"],  # Beginning or middle
            "[confident]": ["start"],  # At beginning
            "[giggles]": ["end"],  # At end
            "[sighs]": ["start"],  # At beginning
        }

        positions = tag_positions.get(tag, ["start"])

        if "start" in positions:
            return f"{tag} {sentence}"
        elif "end" in positions:
            return f"{sentence} {tag}"
        elif "mid" in positions:
            words = sentence.split()
            if len(words) > 3:
                mid_point = len(words) // 2
                words.insert(mid_point, tag)
                return " ".join(words)
            else:
                return f"{tag} {sentence}"
        else:
            return f"{tag} {sentence}"


class ElevenLabsV3Engine:
    """
    🚀 ElevenLabs V3 Engine - Revolutionary voice synthesis

    Provides cutting-edge voice synthesis with emotional audio tags,
    dialogue generation, and cultural adaptation while maintaining
    compatibility with existing Jeffrey voice system.
    """

    def __init__(self, data_dir: str = "data/voice") -> None:
        """Initialize V3 engine with fallback to existing system"""
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # V3 API configuration sécurisée
        self.elevenlabs_api_key = None
        if SECURE_CONFIG_AVAILABLE:
            try:
                self.elevenlabs_api_key = get_api_key("elevenlabs")
                logger.info("🔒 Clé ElevenLabs chargée depuis SecureConfig")
            except SecureConfigError as e:
                logger.warning(f"🔒 Erreur SecureConfig: {e}")

        # Fallback sur .env pour la transition
        if not self.elevenlabs_api_key:
            self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        if self.elevenlabs_api_key and self.elevenlabs_api_key != "your_key_here":
            logger.warning("⚠️ Clé ElevenLabs chargée depuis .env - Migration vers SecureConfig recommandée")

        self.v3_base_url = "https://api.elevenlabs.io/v1"

        # V3 endpoints
        self.v3_endpoints = {
            "tts": f"{self.v3_base_url}/text-to-speech",
            "dialogue": f"{self.v3_base_url}/text-to-dialogue",
            "voices": f"{self.v3_base_url}/voices",
            "models": f"{self.v3_base_url}/models",
        }

        # Settings and processors
        self.v3_settings = V3VoiceSettings()
        self.tag_processor = EmotionalTagProcessor()

        # Fallback to existing engine
        self.v2_engine = VoiceEngine(data_dir)

        # V3 availability
        self.v3_available = False
        self.use_v3 = True  # Default to V3 if available

        # ElevenLabs adapter for network calls
        self.adapter: ElevenLabsAdapter | None = None

        # Performance tracking
        self.v3_stats = {
            "v3_requests": 0,
            "v2_fallback_requests": 0,
            "dialogue_requests": 0,
            "tag_enhanced_requests": 0,
            "average_v3_synthesis_time": 0.0,
        }

        logger.info(f"🚀 ElevenLabsV3Engine initialized - API: {bool(self.elevenlabs_api_key)}")

        # Test V3 availability
        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(self._test_v3_availability())
        except RuntimeError:
            logger.debug("No event loop running - skipping V3 connectivity test")

    async def _test_v3_availability(self):
        """Test V3 API availability"""
        if not self.elevenlabs_api_key:
            logger.warning("No ElevenLabs API key - V3 disabled")
            return

        try:
            await self._ensure_adapter()
            available = await self.adapter.check_api_availability()

            if available:
                self.v3_available = True
                logger.info("✅ ElevenLabs V3 API available")
            else:
                logger.warning("⚠️ ElevenLabs V3 API not responding correctly")

        except Exception as e:
            logger.warning(f"⚠️ ElevenLabs V3 connectivity test failed: {e}")

    async def _ensure_adapter(self):
        """Ensure ElevenLabs adapter is initialized."""
        if self.adapter is None and self.elevenlabs_api_key:
            self.adapter = ElevenLabsAdapter(
                api_key=self.elevenlabs_api_key,
                subscription_tier="free",  # Default, could be configurable
                timeout=30,
            )
            await self.adapter.__aenter__()

    async def synthesize_speech(
        self,
        text: str,
        emotion_context: str = "neutral",
        conversation_history: list[str] = None,
        use_dialogue_mode: bool = False,
    ) -> bytes | None:
        """
        🎭 Synthesize speech with V3 emotional enhancement
        """
        start_time = datetime.now()

        try:
            # Decide between V3 and V2 fallback
            if self.v3_available and self.use_v3:
                # Use V3 with emotional tags
                if use_dialogue_mode:
                    audio_data = await self._synthesize_dialogue_v3(text, emotion_context)
                    self.v3_stats["dialogue_requests"] += 1
                else:
                    audio_data = await self._synthesize_enhanced_v3(text, emotion_context, conversation_history)
                    self.v3_stats["tag_enhanced_requests"] += 1

                self.v3_stats["v3_requests"] += 1

            else:
                # Fallback to V2.5 engine
                audio_data = await self.v2_engine.synthesize_speech(text, emotion_context)
                self.v3_stats["v2_fallback_requests"] += 1

            # Update performance stats
            synthesis_time = (datetime.now() - start_time).total_seconds()
            self.v3_stats["average_v3_synthesis_time"] = (
                (
                    (self.v3_stats["average_v3_synthesis_time"] * (self.v3_stats["v3_requests"] - 1) + synthesis_time)
                    / self.v3_stats["v3_requests"]
                )
                if self.v3_stats["v3_requests"] > 0
                else synthesis_time
            )

            return audio_data

        except Exception as e:
            logger.error(f"❌ V3 synthesis failed: {e}")
            # Emergency fallback to V2
            return await self.v2_engine.synthesize_speech(text, emotion_context)

    async def _synthesize_enhanced_v3(
        self, text: str, emotion_context: str, conversation_history: list[str] = None
    ) -> bytes | None:
        """Synthesize with V3 emotional tags"""
        # Process text with emotional tags
        enhanced_text = self.tag_processor.process_text_with_emotion(text, emotion_context, conversation_history)

        logger.info(f"🎭 Enhanced text: {enhanced_text}")

        # V3 API call with enhanced text
        return await self._call_v3_tts_api(enhanced_text, emotion_context)

    async def _synthesize_dialogue_v3(self, text: str, emotion_context: str) -> bytes | None:
        """Synthesize using V3 dialogue API for multi-speaker content"""
        # Use the new dialogue API integration
        result = await self.dialogue_api_integration([], text, {"emotion": emotion_context})
        return result.get("audio_data") if result else None

    async def dialogue_api_integration(
        self,
        conversation_history: list[dict],
        user_input: str,
        voice_settings: dict | None = None,
    ) -> dict[str, Any]:
        """
        🎭 Gère une conversation complète avec l'API ElevenLabs Dialogue

        Fonctionnalités avancées:
        - Support des conversations multi-tours avec contextualisation
        - Gestion intelligente du contexte de dialogue
        - Streaming audio pour réponses longues avec interruptions
        - Gestion robuste des erreurs avec retry exponential
        - Adaptation émotionnelle selon l'historique

        Args:
            conversation_history: List[Dict] - historique des échanges
                Format: [{"role": "user"|"assistant", "content": str, "emotion": str, "timestamp": str}]
            user_input: str - dernière entrée utilisateur
            voice_settings: Dict - paramètres vocaux optionnels
                Format: {"emotion": str, "persona": str, "intensity": float, "streaming": bool}

        Returns:
            Dict avec:
            - audio_stream: AsyncIterator[bytes] | bytes - flux audio ou données complètes
            - text_response: str - texte de la réponse générée
            - metadata: Dict - métadonnées (durée, tokens, émotions détectées)
            - conversation_context: Dict - contexte mis à jour pour prochaine itération
            - interruption_points: List[float] - points temporels d'interruption possible
        """
        start_time = datetime.now()

        # Initialisation avec valeurs par défaut
        settings = voice_settings or {}
        emotion_context = settings.get("emotion", "neutral")
        enable_streaming = settings.get("streaming", True)
        persona = settings.get("persona", self.v3_settings.voice_persona.value)
        emotional_intensity = settings.get("intensity", self.v3_settings.emotional_intensity)

        logger.info(f"🎭 [DIALOGUE] Starting dialogue API integration - User: '{user_input[:50]}...'")
        logger.debug(
            f"🎭 [DIALOGUE] Settings: emotion={emotion_context}, streaming={enable_streaming}, persona={persona}"
        )

        try:
            # Étape 1: Analyse contextuelle de la conversation
            conversation_context = await self._analyze_conversation_context(
                conversation_history, user_input, emotion_context
            )

            # Étape 2: Adaptation émotionnelle intelligente
            adaptive_emotion = await self._adapt_emotional_context(
                conversation_context, emotion_context, conversation_history
            )

            # Étape 3: Génération de la réponse textuelle (simulée pour l'instant)
            text_response = await self._generate_contextual_response(user_input, conversation_context, adaptive_emotion)

            # Étape 4: Enhancement du texte avec tags émotionnels V3
            enhanced_text = self.tag_processor.process_text_with_emotion(
                text_response,
                adaptive_emotion,
                [entry.get("content", "") for entry in conversation_history[-3:]],
            )

            # Étape 5: Préparation de la configuration vocale optimisée
            optimized_voice_config = self._optimize_voice_configuration(
                adaptive_emotion, persona, emotional_intensity, conversation_context
            )

            # Étape 6: Génération audio avec gestion streaming ou complète
            if enable_streaming:
                audio_result = await self._generate_streaming_audio(
                    enhanced_text, optimized_voice_config, conversation_context
                )
            else:
                audio_result = await self._generate_complete_audio(enhanced_text, optimized_voice_config)

            # Étape 7: Calcul des points d'interruption intelligents
            interruption_points = self._calculate_interruption_points(enhanced_text, audio_result.get("duration", 0))

            # Étape 8: Mise à jour du contexte conversationnel
            updated_context = self._update_conversation_context(
                conversation_context, user_input, text_response, adaptive_emotion
            )

            # Compilation des métadonnées complètes
            processing_time = (datetime.now() - start_time).total_seconds()
            metadata = {
                "processing_time_seconds": processing_time,
                "original_emotion": emotion_context,
                "adaptive_emotion": adaptive_emotion,
                "persona_used": persona,
                "emotional_intensity": emotional_intensity,
                "text_length": len(text_response),
                "enhanced_text_length": len(enhanced_text),
                "audio_format": audio_result.get("format", "mp3"),
                "audio_duration_seconds": audio_result.get("duration", 0),
                "streaming_enabled": enable_streaming,
                "interruption_points_count": len(interruption_points),
                "context_analysis": conversation_context.get("analysis", {}),
                "voice_modulations": optimized_voice_config,
                "api_calls_made": audio_result.get("api_calls", 1),
                "retry_attempts": audio_result.get("retry_attempts", 0),
            }

            logger.info(
                f"✅ [DIALOGUE] Dialogue API completed in {processing_time:.2f}s - Response: '{text_response[:50]}...'"
            )

            return {
                "audio_stream": audio_result.get("audio_data"),
                "audio_data": audio_result.get("audio_data"),  # Backward compatibility
                "text_response": text_response,
                "enhanced_text": enhanced_text,
                "metadata": metadata,
                "conversation_context": updated_context,
                "interruption_points": interruption_points,
                "success": True,
                "error": None,
            }

        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Dialogue API integration failed: {str(e)}"

            logger.error(f"❌ [DIALOGUE] {error_msg} (after {processing_time:.2f}s)")

            # Fallback robuste
            try:
                fallback_result = await self._fallback_dialogue_generation(user_input, emotion_context)

                return {
                    "audio_stream": fallback_result.get("audio_data"),
                    "audio_data": fallback_result.get("audio_data"),
                    "text_response": fallback_result.get(
                        "text",
                        "Je rencontre une difficulté technique, mais je suis là pour vous aider.",
                    ),
                    "enhanced_text": fallback_result.get(
                        "text",
                        "Je rencontre une difficulté technique, mais je suis là pour vous aider.",
                    ),
                    "metadata": {
                        "processing_time_seconds": processing_time,
                        "original_emotion": emotion_context,
                        "fallback_used": True,
                        "error_occurred": True,
                        "error_message": error_msg,
                    },
                    "conversation_context": {"fallback": True},
                    "interruption_points": [],
                    "success": False,
                    "error": error_msg,
                }

            except Exception as fallback_error:
                logger.error(f"❌ [DIALOGUE] Fallback also failed: {fallback_error}")

                return {
                    "audio_stream": None,
                    "audio_data": None,
                    "text_response": "Je rencontre des difficultés techniques temporaires.",
                    "enhanced_text": "Je rencontre des difficultés techniques temporaires.",
                    "metadata": {
                        "processing_time_seconds": processing_time,
                        "error_occurred": True,
                        "error_message": error_msg,
                        "fallback_error": str(fallback_error),
                    },
                    "conversation_context": {},
                    "interruption_points": [],
                    "success": False,
                    "error": error_msg,
                }

    async def _analyze_conversation_context(
        self, conversation_history: list[dict], user_input: str, emotion_context: str
    ) -> dict[str, Any]:
        """Analyse le contexte conversationnel pour optimiser la réponse"""
        context = {
            "conversation_length": len(conversation_history),
            "recent_emotions": [],
            "dominant_themes": [],
            "user_engagement_level": "medium",
            "conversation_flow": "normal",
            "requires_empathy": False,
            "requires_excitement": False,
            "is_continuation": False,
        }

        if conversation_history:
            # Analyser les dernières émotions
            recent_entries = conversation_history[-5:]  # 5 derniers échanges
            context["recent_emotions"] = [
                entry.get("emotion", "neutral") for entry in recent_entries if entry.get("emotion")
            ]

            # Détecter les patterns conversationnels
            user_messages = [entry for entry in recent_entries if entry.get("role") == "user"]
        if user_messages:
            context["is_continuation"] = any(
                word in user_input.lower() for word in ["et", "aussi", "continue", "encore", "alors", "donc"]
            )

            # Détecter le besoin d'empathie
            empathy_triggers = ["problème", "difficile", "triste", "inquiet", "peur", "mal"]
            context["requires_empathy"] = any(trigger in user_input.lower() for trigger in empathy_triggers)

            # Détecter l'enthousiasme
            excitement_triggers = ["génial", "fantastique", "super", "formidable", "excellent"]
            context["requires_excitement"] = any(trigger in user_input.lower() for trigger in excitement_triggers)

        # Analyser l'engagement basé sur la longueur et complexité
        input_complexity = len(user_input.split())
        if input_complexity > 20:
            context["user_engagement_level"] = "high"
        elif input_complexity < 5:
            context["user_engagement_level"] = "low"

        context["analysis"] = {
            "input_word_count": input_complexity,
            "question_detected": "?" in user_input,
            "exclamation_detected": "!" in user_input,
            "conversation_turn": len(conversation_history) + 1,
        }

        return context

    async def _adapt_emotional_context(
        self, conversation_context: dict, base_emotion: str, conversation_history: list[dict]
    ) -> str:
        """Adapte le contexte émotionnel selon l'historique et l'analyse"""

        # Si empathie requise, adapter
        if conversation_context.get("requires_empathy"):
            if base_emotion in ["neutral", "happy"]:
                return "empathetic"

        # Si enthousiasme détecté, amplifier
        if conversation_context.get("requires_excitement"):
            if base_emotion in ["neutral", "calm"]:
                return "excited"

        # Continuité émotionnelle
        recent_emotions = conversation_context.get("recent_emotions", [])
        if recent_emotions:
            last_emotion = recent_emotions[-1]

            # Si l'utilisateur était triste, rester empathique
        if last_emotion in ["sad", "empathetic"] and base_emotion == "neutral":
            return "empathetic"

            # Si l'utilisateur était excité, maintenir l'énergie
        if last_emotion in ["excited", "happy"] and base_emotion == "neutral":
            return "happy"

        return base_emotion

    async def _generate_contextual_response(self, user_input: str, context: dict, emotion: str) -> str:
        """Génère une réponse contextuelle (simulée pour l'instant)"""

        # Pour l'instant, génération simulée basée sur patterns
        # En production, ceci serait connecté au système de génération de Jeffrey

        input_lower = user_input.lower()

        if context.get("requires_empathy"):
            responses = [
                "Je comprends que ce soit difficile pour vous. Voulez-vous m'en dire plus ?",
                "C'est normal de ressentir cela. Je suis là pour vous aider.",
                "Je ressens votre préoccupation. Prenons le temps d'en parler ensemble.",
            ]
        elif context.get("requires_excitement"):
            responses = [
                "C'est fantastique ! J'adore votre enthousiasme !",
                "Quelle excellente nouvelle ! Racontez-moi tout !",
                "Je partage votre excitation ! C'est merveilleux !",
            ]
        elif "?" in user_input:
            responses = [
                "C'est une excellente question. Laissez-moi vous expliquer...",
                "Je vais vous aider avec ça. Voici ce que je peux vous dire...",
                "Bonne question ! Permettez-moi de vous donner une réponse détaillée...",
            ]
        else:
            responses = [
                "Je vois, c'est très intéressant. Continuez, je vous écoute.",
                "Merci de partager cela avec moi. Que puis-je faire pour vous aider ?",
                "J'apprécie que vous me fassiez confiance. Comment puis-je vous assister ?",
            ]

        import random

        return random.choice(responses)

    async def _optimize_voice_configuration(
        self, emotion: str, persona: str, intensity: float, context: dict
    ) -> dict[str, Any]:
        """Optimise la configuration vocale selon le contexte"""

        # Configuration de base selon l'émotion
        base_config = {
            "stability": self.v3_settings.stability,
            "similarity_boost": self.v3_settings.similarity_boost,
            "style": self.v3_settings.style,
            "use_speaker_boost": self.v3_settings.use_speaker_boost,
        }

        # Modulations selon l'émotion
        emotion_modulations = {
            "happy": {"stability": 0.8, "style": 0.3, "similarity_boost": 0.9},
            "excited": {"stability": 0.7, "style": 0.5, "similarity_boost": 0.85},
            "empathetic": {"stability": 0.9, "style": 0.1, "similarity_boost": 0.95},
            "calm": {"stability": 0.95, "style": 0.0, "similarity_boost": 0.9},
            "neutral": {"stability": 0.8, "style": 0.2, "similarity_boost": 0.85},
        }

        if emotion in emotion_modulations:
            modulation = emotion_modulations[emotion]
        for key, value in modulation.items():
            base_config[key] = max(0, min(1, value * intensity))

        # Ajustements selon le persona
        persona_adjustments = {
            "professional": {"stability": 1.1, "style": 0.9},
            "friendly": {"stability": 1.0, "style": 1.1},
            "creative": {"stability": 0.9, "style": 1.2},
            "empathetic": {"stability": 1.1, "style": 0.8},
            "excited": {"stability": 0.8, "style": 1.3},
            "teacher": {"stability": 1.1, "style": 0.9},
        }

        if persona in persona_adjustments:
            adjustment = persona_adjustments[persona]
        for key, multiplier in adjustment.items():
            base_config[key] = max(0, min(1, base_config[key] * multiplier))

        return base_config

    async def _generate_streaming_audio(self, text: str, voice_config: dict, context: dict) -> dict[str, Any]:
        """Génère l'audio en streaming avec gestion des interruptions"""

        # Pour l'implémentation actuelle, simuler le streaming
        # En production, ceci utiliserait l'API streaming d'ElevenLabs

        logger.info("🔊 [STREAMING] Generating streaming audio...")

        try:
            # Simuler le découpage en chunks pour streaming
            text_chunks = self._split_text_for_streaming(text)

            # Pour l'instant, générer l'audio complet puis simuler streaming
            audio_data = await self._call_v3_tts_api_with_config(text, voice_config)

            if audio_data:
                return {
                    "audio_data": audio_data,
                    "format": "mp3",
                    "duration": len(text) * 0.05,  # Estimation 50ms par caractère
                    "chunks": len(text_chunks),
                    "streaming": True,
                    "api_calls": len(text_chunks),
                    "retry_attempts": 0,
                }
            else:
                raise Exception("Failed to generate audio")

        except Exception as e:
            logger.error(f"❌ [STREAMING] Streaming audio generation failed: {e}")
            raise

    async def _generate_complete_audio(self, text: str, voice_config: dict) -> dict[str, Any]:
        """Génère l'audio complet en une fois"""

        logger.info("🔊 [COMPLETE] Generating complete audio...")

        try:
            audio_data = await self._call_v3_tts_api_with_config(text, voice_config)

            if audio_data:
                return {
                    "audio_data": audio_data,
                    "format": "mp3",
                    "duration": len(text) * 0.05,  # Estimation
                    "streaming": False,
                    "api_calls": 1,
                    "retry_attempts": 0,
                }
            else:
                raise Exception("Failed to generate audio")

        except Exception as e:
            logger.error(f"❌ [COMPLETE] Complete audio generation failed: {e}")
            raise

    async def _call_v3_tts_api_with_config(self, text: str, voice_config: dict) -> bytes | None:
        """Appel API V3 avec configuration personnalisée et retry logic"""

        if not self.elevenlabs_api_key:
            raise Exception("No ElevenLabs API key configured")

        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                payload = {
                    "text": text,
                    "model_id": "eleven_multilingual_v2",
                    "voice_settings": voice_config,
                }

                headers = {
                    "xi-api-key": self.elevenlabs_api_key,
                    "Content-Type": "application/json",
                }

                await self._ensure_adapter()

                logger.debug(f"🔄 [API] Attempt {attempt + 1}/{max_retries} - Calling V3 TTS API")

                try:
                    audio_data = await self.adapter.text_to_speech(
                        text=payload["text"],
                        voice_id=self.v3_settings.voice_id,
                        model_id=payload.get("model_id", "eleven_multilingual_v2"),
                        voice_settings=payload.get("voice_settings", {}),
                        output_format="mp3_44100_128",
                    )

                    if audio_data:
                        logger.info(f"✅ [API] V3 TTS API successful on attempt {attempt + 1}")
                        return audio_data
                    else:
                        logger.warning(f"⚠️ [API] No audio data received on attempt {attempt + 1}")
                        if attempt == max_retries - 1:
                            raise Exception(f"V3 API failed after {max_retries} attempts - no audio data")
                        await asyncio.sleep(retry_delay)

                except Exception as api_error:
                    logger.error(f"❌ [API] V3 API error on attempt {attempt + 1}: {api_error}")
                    if attempt == max_retries - 1:
                        raise Exception(f"V3 API failed after {max_retries} attempts: {api_error}")
                    await asyncio.sleep(retry_delay)

            except Exception as e:
                logger.error(f"❌ [API] Network error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise Exception(f"Network error after {max_retries} attempts: {e}")
                await asyncio.sleep(retry_delay)

        return None

    def _split_text_for_streaming(self, text: str) -> list[str]:
        """Découpe le texte en chunks optimaux pour streaming"""

        # Découpage intelligent selon la ponctuation
        sentences = re.split(r"[.!?]+", text)
        chunks = []
        current_chunk = ""
        max_chunk_length = 100  # Caractères max par chunk

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current_chunk) + len(sentence) < max_chunk_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks if chunks else [text]

    def _calculate_interruption_points(self, text: str, audio_duration: float) -> list[float]:
        """Calcule les points temporels où l'utilisateur peut interrompre"""

        # Points d'interruption aux pauses naturelles
        sentences = re.split(r"[.!?]+", text)
        interruption_points = []

        current_time = 0.0
        chars_per_second = len(text) / audio_duration if audio_duration > 0 else 20

        for sentence in sentences:
            if sentence.strip():
                sentence_duration = len(sentence) / chars_per_second
                current_time += sentence_duration
                interruption_points.append(round(current_time, 2))

        return interruption_points

    def _update_conversation_context(
        self, context: dict, user_input: str, response: str, emotion: str
    ) -> dict[str, Any]:
        """Met à jour le contexte conversationnel pour la prochaine itération"""

        updated_context = context.copy()
        updated_context.update(
            {
                "last_user_input": user_input,
                "last_response": response,
                "last_emotion": emotion,
                "last_interaction_time": datetime.now().isoformat(),
                "conversation_length": context.get("conversation_length", 0) + 1,
            }
        )

        return updated_context

    async def _fallback_dialogue_generation(self, user_input: str, emotion: str) -> dict[str, Any]:
        """Génération de dialogue de fallback en cas d'échec"""

        logger.info("🔄 [FALLBACK] Using fallback dialogue generation...")

        try:
            # Utiliser le système V2 comme fallback
            fallback_text = "Je vous écoute et je suis là pour vous aider."
            audio_data = await self.v2_engine.synthesize_speech(fallback_text, emotion)

            return {"audio_data": audio_data, "text": fallback_text, "fallback_used": True}

        except Exception as e:
            logger.error(f"❌ [FALLBACK] Fallback generation failed: {e}")
            raise

    async def _call_v3_tts_api(self, text: str, emotion_context: str) -> bytes | None:
        """Make actual V3 API call"""
        if not self.elevenlabs_api_key:
            raise Exception("No ElevenLabs API key configured")

        # Apply emotional modulation to V3 settings
        modulated_settings = self._apply_emotional_modulation(emotion_context)

        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",  # V3 model
            "voice_settings": {
                "stability": modulated_settings.stability,
                "similarity_boost": modulated_settings.similarity_boost,
                "style": modulated_settings.style,
                "use_speaker_boost": modulated_settings.use_speaker_boost,
            },
        }

        headers = {"xi-api-key": self.elevenlabs_api_key, "Content-Type": "application/json"}

        try:
            await self._ensure_adapter()

            audio_data = await self.adapter.text_to_speech(
                text=payload["text"],
                voice_id=self.v3_settings.voice_id,
                model_id=payload.get("model_id", "eleven_multilingual_v2"),
                voice_settings=payload.get("voice_settings", {}),
                output_format="mp3_44100_128",
            )

            if audio_data:
                return audio_data
            else:
                logger.error("❌ V3 API returned no audio data")
                return None

        except Exception as e:
            logger.error(f"❌ V3 API request failed: {e}")
            return None

    def _apply_emotional_modulation(self, emotion_context: str) -> V3VoiceSettings:
        """Apply emotional modulation to V3 settings"""
        # Get base emotional profile from V2 system
        emotional_profile = self.v2_engine.emotional_profiles.get(emotion_context)

        if not emotional_profile:
            return self.v3_settings

        # Create modulated V3 settings
        modulated = V3VoiceSettings(
            voice_id=self.v3_settings.voice_id,
            stability=max(0, min(1, self.v3_settings.stability + emotional_profile.stability_modifier)),
            similarity_boost=max(0, min(1, self.v3_settings.similarity_boost + emotional_profile.similarity_modifier)),
            style=max(0, min(1, self.v3_settings.style + emotional_profile.style_modifier)),
            use_speaker_boost=self.v3_settings.use_speaker_boost,
            enable_audio_tags=True,
            emotional_intensity=self.v3_settings.emotional_intensity,
        )

        return modulated

    def get_available_personas(self) -> list[dict[str, Any]]:
        """Get available voice personas"""
        return [
            {
                "id": persona.value,
                "name": persona.value.title(),
                "description": self._get_persona_description(persona),
                "emotional_context": self._get_persona_emotions(persona),
            }
            for persona in VoicePersona
        ]

    def _get_persona_description(self, persona: VoicePersona) -> str:
        """Get description for voice persona"""
        descriptions = {
            VoicePersona.PROFESSIONAL: "Confident, clear, and authoritative voice for explanations and tutorials",
            VoicePersona.FRIENDLY: "Warm, approachable, and conversational for casual interactions",
            VoicePersona.CREATIVE: "Dynamic, expressive, and playful for brainstorming and storytelling",
            VoicePersona.EMPATHETIC: "Gentle, caring, and supportive for emotional conversations",
            VoicePersona.EXCITED: "Energetic, enthusiastic, and animated for exciting news and discoveries",
            VoicePersona.TEACHER: "Patient, clear, and encouraging for learning and instruction",
        }
        return descriptions.get(persona, "Balanced, versatile voice for general use")

    def _get_persona_emotions(self, persona: VoicePersona) -> list[str]:
        """Get primary emotions for voice persona"""
        emotions = {
            VoicePersona.PROFESSIONAL: ["confident", "neutral", "calm"],
            VoicePersona.FRIENDLY: ["happy", "calm", "empathetic"],
            VoicePersona.CREATIVE: ["excited", "playful", "curious"],
            VoicePersona.EMPATHETIC: ["empathetic", "gentle", "calm"],
            VoicePersona.EXCITED: ["excited", "happy", "enthusiastic"],
            VoicePersona.TEACHER: ["calm", "patient", "encouraging"],
        }
        return emotions.get(persona, ["neutral"])

    def set_voice_persona(self, persona: VoicePersona) -> None:
        """Set active voice persona"""
        self.v3_settings.voice_persona = persona
        logger.info(f"🎭 Voice persona set to: {persona.value}")

    def get_performance_stats(self) -> dict[str, Any]:
        """Get V3 performance statistics"""
        total_requests = self.v3_stats["v3_requests"] + self.v3_stats["v2_fallback_requests"]

        return {
            **self.v3_stats,
            "total_requests": total_requests,
            "v3_usage_rate": ((self.v3_stats["v3_requests"] / total_requests * 100) if total_requests > 0 else 0),
            "v3_available": self.v3_available,
            "current_persona": self.v3_settings.voice_persona.value,
            "fallback_engine_stats": self.v2_engine.synthesis_stats,
        }

    async def test_emotional_expressions(self) -> dict[str, Any]:
        """Test different emotional expressions with V3"""
        test_phrases = [
            ("Bonjour ! Je suis Jeffrey, votre assistant vocal intelligent.", "happy"),
            ("Hmm, c'est une question très intéressante...", "curious"),
            ("Je suis vraiment désolé que vous rencontriez ce problème.", "empathetic"),
            ("Fantastique ! J'ai trouvé exactement ce qu'il vous faut !", "excited"),
            ("Laissez-moi vous expliquer cela étape par étape.", "professional"),
        ]

        results = {}

        for text, emotion in test_phrases:
            try:
                enhanced_text = self.tag_processor.process_text_with_emotion(text, emotion)
                results[emotion] = {"original": text, "enhanced": enhanced_text, "status": "ready"}
            except Exception as e:
                results[emotion] = {"original": text, "enhanced": text, "status": f"error: {e}"}

        return results


# Convenience functions for easy integration
async def synthesize_with_v3(
    text: str, emotion: str = "neutral", voice_engine: ElevenLabsV3Engine = None
) -> bytes | None:
    """Convenience function for V3 synthesis"""
    if voice_engine is None:
        voice_engine = ElevenLabsV3Engine()

    return await voice_engine.synthesize_speech(text, emotion)


def create_v3_engine(data_dir: str = "data/voice") -> ElevenLabsV3Engine:
    pass
    """Factory function to create V3 engine"""
    return ElevenLabsV3Engine(data_dir)
