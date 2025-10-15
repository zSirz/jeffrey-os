"""
# VOCAL RECOVERY - PROVENANCE HEADER
# Module: voice_controller_v3.py
# Source: Jeffrey_OS/src/storage/backups/pre_reorganization/old_versions/Jeffrey/Jeffrey_DEV_FIX/Jeffrey_LIVE/core/voice/voice_controller_v3.py
# Hash: 7a04bb9213d1c366
# Score: 2570
# Classes: EmotionalContextAnalyzer, VoiceControllerV3
# Recovered: 2025-08-08T11:33:24.799969
# Tier: TIER2_CORE
"""

from __future__ import annotations

"""
🚀 VoiceController V3 - Jeffrey's Enhanced Voice Pipeline with ElevenLabs V3
===========================================================================

Advanced voice pipeline orchestrator with V3 emotional intelligence,
dialogue capabilities, and seamless fallback to existing system.

Enhanced Features:
- ElevenLabs V3 emotional audio tags integration
- Intelligent emotion detection from conversation context
- Multi-speaker dialogue support
- Cultural voice adaptation
- Advanced performance monitoring
- 100% backward compatibility with existing VoiceController

Pipeline Flow V3:
Audio Input → Speech Recognition → Brain Processing → Emotional Analysis →
V3 Enhancement → Voice Synthesis → Audio Output
"""

import logging
import os

# Brain integration
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .elevenlabs_v3_engine import ElevenLabsV3Engine, VoicePersona

# Import existing voice infrastructure
from .voice_controller import VoiceController

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from agi_fusion.intelligent_learning_orchestrator import IntelligentLearningOrchestrator

    BRAIN_AVAILABLE = True
except ImportError:
    BRAIN_AVAILABLE = False
    logging.warning("Brain modules not available - using mock responses")

logger = logging.getLogger(__name__)


class EmotionalContextAnalyzer:
    """
    🧠 Analyzes conversation context to determine optimal emotional expression
    """

    def __init__(self) -> None:
        """Initialize emotional context analyzer"""
        self.conversation_history = []
        self.user_emotional_state = "neutral"
        self.conversation_type = "general"
        self.jeffrey_personality_state = {
            "mood": "balanced",
            "energy_level": 0.7,
            "empathy_level": 0.8,
            "playfulness": 0.6,
        }

    def analyze_conversation_context(
        self, user_message: str, jeffrey_response: str, brain_context: dict[str, Any] = None
    ) -> dict[str, Any]:
        """
        🔍 Analyze conversation to determine optimal emotional context
        """
        # Analyze user emotional state
        user_emotion = self._detect_user_emotion(user_message)

        # Analyze conversation type
        conversation_type = self._classify_conversation_type(user_message, brain_context)

        # Determine Jeffrey's optimal emotional response
        jeffrey_emotion = self._determine_jeffrey_emotion(
            user_emotion, conversation_type, jeffrey_response, brain_context
        )

        # Select optimal voice persona
        voice_persona = self._select_voice_persona(conversation_type, jeffrey_emotion)

        # Update conversation history
        self._update_conversation_history(user_message, jeffrey_response, user_emotion, jeffrey_emotion)

        return {
            "user_emotion": user_emotion,
            "jeffrey_emotion": jeffrey_emotion,
            "conversation_type": conversation_type,
            "voice_persona": voice_persona,
            "emotional_intensity": self._calculate_emotional_intensity(jeffrey_emotion),
            "conversation_context": self._get_conversation_context(),
        }

    def _detect_user_emotion(self, message: str) -> str:
        """Detect user's emotional state from message"""
        message_lower = message.lower()

        # Emotion indicators
        emotion_patterns = {
            "excited": ["fantastique", "génial", "incroyable", "waouw", "super", "!!", "😄", "🎉"],
            "happy": ["content", "heureux", "joie", "sourire", "drôle", "😊", "😃"],
            "sad": ["triste", "déçu", "difficile", "problème", "inquiet", "😢", "😞"],
            "angry": ["énervé", "frustré", "agacé", "colère", "ridicule", "😠", "😡"],
            "curious": ["pourquoi", "comment", "qu'est-ce que", "intéressant", "?", "🤔"],
            "surprised": ["vraiment", "sérieusement", "pas possible", "oh", "ah", "😮", "😲"],
            "confused": ["comprends pas", "perdu", "confus", "bizarre", "strange", "🤷"],
            "grateful": ["merci", "thanks", "reconnaissance", "aidé", "🙏", "❤️"],
        }

        # Count emotion indicators
        emotion_scores = {}
        for emotion, patterns in emotion_patterns.items():
            score = sum(1 for pattern in patterns if pattern in message_lower)
            if score > 0:
                emotion_scores[emotion] = score

        # Return strongest emotion or neutral
        if emotion_scores:
            return max(emotion_scores, key=emotion_scores.get)
        else:
            return "neutral"

    def _classify_conversation_type(self, message: str, brain_context: dict[str, Any] = None) -> str:
        """Classify the type of conversation"""
        message_lower = message.lower()

        # Conversation type patterns
        type_patterns = {
            "learning": ["apprendre", "expliquer", "comment", "pourquoi", "enseigner", "leçon"],
            "creative": ["créer", "imaginer", "idée", "brainstorm", "invention", "créatif"],
            "problem_solving": ["problème", "solution", "aide", "réparer", "résoudre", "bug"],
            "casual": ["salut", "bonjour", "comment ça va", "parler", "discussion"],
            "emotional_support": [
                "inquiet",
                "stressé",
                "difficile",
                "soutien",
                "conseil",
                "écouter",
            ],
            "humor": ["blague", "drôle", "rire", "joke", "amusant", "rigolo"],
            "professional": [
                "travail",
                "projet",
                "business",
                "professionnel",
                "réunion",
                "présentation",
            ],
        }

        # Check brain context for additional clues
        if brain_context:
            task_type = brain_context.get("task_type", "")
            if task_type in ["analysis", "explanation"]:
                return "learning"
            elif task_type in ["creative_writing", "brainstorming"]:
                return "creative"

        # Score conversation types
        type_scores = {}
        for conv_type, patterns in type_patterns.items():
            score = sum(1 for pattern in patterns if pattern in message_lower)
            if score > 0:
                type_scores[conv_type] = score

        return max(type_scores, key=type_scores.get) if type_scores else "general"

    def _determine_jeffrey_emotion(
        self,
        user_emotion: str,
        conversation_type: str,
        jeffrey_response: str,
        brain_context: dict[str, Any] = None,
    ) -> str:
        """Determine Jeffrey's optimal emotional response"""

        # Empathetic response mapping
        empathy_mapping = {
            "sad": "empathetic",
            "angry": "calm",
            "confused": "patient",
            "excited": "excited",
            "happy": "happy",
            "curious": "curious",
            "surprised": "curious",
            "grateful": "warm",
        }

        # Base emotion from empathy
        base_emotion = empathy_mapping.get(user_emotion, "neutral")

        # Adjust based on conversation type
        type_adjustments = {
            "learning": {"patient": 0.3, "confident": 0.2},
            "creative": {"excited": 0.4, "playful": 0.3},
            "problem_solving": {"confident": 0.3, "helpful": 0.2},
            "emotional_support": {"empathetic": 0.5, "gentle": 0.3},
            "humor": {"playful": 0.4, "amused": 0.3},
            "professional": {"confident": 0.3, "professional": 0.4},
        }

        # Analyze Jeffrey's response content for emotional cues
        response_emotion = self._analyze_response_emotion(jeffrey_response)

        # Combine factors to determine final emotion
        if conversation_type in type_adjustments:
            # Use conversation type primary emotion
            primary_emotions = list(type_adjustments[conversation_type].keys())
            if primary_emotions:
                return primary_emotions[0]

        return response_emotion or base_emotion

    def _analyze_response_emotion(self, response: str) -> str | None:
        """Analyze Jeffrey's response to detect intended emotion"""
        response_lower = response.lower()

        response_patterns = {
            "excited": ["fantastique", "génial", "incroyable", "découvert", "trouvé"],
            "confident": ["certain", "sûr", "évidemment", "bien sûr", "précisément"],
            "curious": ["intéressant", "explorer", "découvrir", "analyser", "hmm"],
            "empathetic": ["comprends", "désolé", "difficile", "soutien", "aide"],
            "playful": ["amusant", "drôle", "jeu", "blague", "rigolo"],
            "professional": ["analysons", "procédure", "étape", "méthode", "système"],
        }

        for emotion, patterns in response_patterns.items():
            if any(pattern in response_lower for pattern in patterns):
                return emotion

        return None

    def _select_voice_persona(self, conversation_type: str, jeffrey_emotion: str) -> VoicePersona:
        """Select optimal voice persona"""

        # Primary persona mapping
        persona_mapping = {
            "learning": VoicePersona.TEACHER,
            "creative": VoicePersona.CREATIVE,
            "problem_solving": VoicePersona.PROFESSIONAL,
            "emotional_support": VoicePersona.EMPATHETIC,
            "humor": VoicePersona.FRIENDLY,
            "professional": VoicePersona.PROFESSIONAL,
            "casual": VoicePersona.FRIENDLY,
            "general": VoicePersona.FRIENDLY,
        }

        # Emotion overrides
        emotion_overrides = {
            "excited": VoicePersona.EXCITED,
            "empathetic": VoicePersona.EMPATHETIC,
            "confident": VoicePersona.PROFESSIONAL,
            "playful": VoicePersona.CREATIVE,
        }

        # Check emotion override first
        if jeffrey_emotion in emotion_overrides:
            return emotion_overrides[jeffrey_emotion]

        # Use conversation type mapping
        return persona_mapping.get(conversation_type, VoicePersona.FRIENDLY)

    def _calculate_emotional_intensity(self, emotion: str) -> float:
        """Calculate emotional intensity level"""
        intensity_mapping = {
            "excited": 0.9,
            "empathetic": 0.7,
            "confident": 0.6,
            "curious": 0.5,
            "playful": 0.8,
            "calm": 0.4,
            "neutral": 0.5,
        }

        return intensity_mapping.get(emotion, 0.5)

    def _update_conversation_history(
        self, user_message: str, jeffrey_response: str, user_emotion: str, jeffrey_emotion: str
    ):
        """Update conversation history for context"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "jeffrey_response": jeffrey_response,
            "user_emotion": user_emotion,
            "jeffrey_emotion": jeffrey_emotion,
        }

        self.conversation_history.append(entry)

        # Keep only last 10 exchanges for performance
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

    def _get_conversation_context(self) -> list[str]:
        """Get conversation context for V3 processing"""
        return [entry["jeffrey_response"] for entry in self.conversation_history[-3:]]


class VoiceControllerV3:
    """
    🚀 Enhanced voice controller with ElevenLabs V3 integration

    Provides seamless upgrade to V3 capabilities while maintaining
    100% compatibility with existing voice pipeline.
    """

    def __init__(self, data_dir: str = "data/voice") -> None:
        """Initialize V3 voice controller"""
        self.data_dir = Path(data_dir)

        # Initialize V3 engine and fallback controller
        self.v3_engine = ElevenLabsV3Engine(data_dir)
        self.v2_controller = VoiceController(data_dir)  # Fallback controller

        # Enhanced components
        self.emotional_analyzer = EmotionalContextAnalyzer()

        # Configuration
        self.use_v3_enhancement = True
        self.auto_persona_selection = True
        self.emotional_continuity = True

        # Performance tracking
        self.v3_performance = {
            "total_interactions": 0,
            "v3_enhanced_interactions": 0,
            "emotional_accuracy_score": 0.0,
            "user_satisfaction_indicators": [],
            "average_response_time": 0.0,
        }

        logger.info("🚀 VoiceControllerV3 initialized with emotional intelligence")

    async def process_voice_interaction(
        self,
        audio_input: bytes = None,
        text_input: str = None,
        brain_context: dict[str, Any] = None,
    ) -> dict[str, Any]:
        """
        🎭 Process complete voice interaction with V3 enhancement
        """
        start_time = datetime.now()

        try:
            # Phase 1: Handle input (audio or text)
            if audio_input:
                # Speech recognition
                speech_result = await self._process_speech_input(audio_input)
                user_message = speech_result.get("text", "")
                confidence = speech_result.get("confidence", 0.0)
            else:
                user_message = text_input or ""
                confidence = 1.0

            if not user_message:
                return {"error": "No input received", "success": False}

            # Phase 2: Brain processing
            brain_response = await self._process_with_brain(user_message, brain_context)
            jeffrey_response = brain_response.get("response", "Je n'ai pas compris votre demande.")

            # Phase 3: Enhanced emotional analysis for V3
            if self.use_v3_enhancement:
                emotional_context = self.emotional_analyzer.analyze_conversation_context(
                    user_message, jeffrey_response, brain_context
                )
            else:
                emotional_context = {
                    "jeffrey_emotion": "neutral",
                    "voice_persona": VoicePersona.FRIENDLY,
                }

            # Phase 4: V3 voice synthesis
            audio_result = await self._synthesize_response_v3(jeffrey_response, emotional_context)

            # Phase 5: Package results
            result = {
                "success": True,
                "user_input": {
                    "text": user_message,
                    "confidence": confidence,
                    "detected_emotion": emotional_context.get("user_emotion", "neutral"),
                },
                "brain_response": brain_response,
                "emotional_context": emotional_context,
                "voice_output": audio_result,
                "performance": {
                    "total_time": (datetime.now() - start_time).total_seconds(),
                    "v3_enhanced": self.use_v3_enhancement,
                    "persona_used": emotional_context.get("voice_persona", VoicePersona.FRIENDLY).value,
                },
            }

            # Update performance tracking
            self._update_performance_stats(result)

            return result

        except Exception as e:
            logger.error(f"❌ Voice interaction failed: {e}")

            # Emergency fallback to V2 controller
            return await self._emergency_fallback(audio_input, text_input, brain_context)

    async def _process_speech_input(self, audio_input: bytes) -> dict[str, Any]:
        """Process speech recognition"""
        # Use existing speech recognition from V2 controller
        return await self.v2_controller._process_audio_input(audio_input)

    async def _process_with_brain(self, message: str, context: dict[str, Any] = None) -> dict[str, Any]:
        """Process message with Jeffrey's brain"""
        if BRAIN_AVAILABLE:
            try:
                # Use existing brain integration
                brain = IntelligentLearningOrchestrator()
                response = await brain.process_message(message, context)
                return {"response": response, "context": context}
            except Exception as e:
                logger.error(f"Brain processing failed: {e}")

        # Fallback response
        return {"response": f"J'ai bien reçu votre message: {message}", "context": context or {}}

    async def _synthesize_response_v3(self, text: str, emotional_context: dict[str, Any]) -> dict[str, Any]:
        """Synthesize response with V3 enhancement"""
        try:
            # Set persona if auto-selection enabled
            if self.auto_persona_selection:
                persona = emotional_context.get("voice_persona", VoicePersona.FRIENDLY)
                self.v3_engine.set_voice_persona(persona)

            # Get emotion and conversation history
            emotion = emotional_context.get("jeffrey_emotion", "neutral")
            conversation_history = emotional_context.get("conversation_context", [])

            # V3 synthesis with emotional enhancement
            audio_data = await self.v3_engine.synthesize_speech(text, emotion, conversation_history)

            if audio_data:
                return {
                    "audio_data": audio_data,
                    "text": text,
                    "emotion_used": emotion,
                    "persona_used": emotional_context.get("voice_persona", VoicePersona.FRIENDLY).value,
                    "v3_enhanced": True,
                    "synthesis_method": "elevenlabs_v3",
                }
            else:
                # Fallback to V2 if V3 fails
                return await self._fallback_synthesis(text, emotion)

        except Exception as e:
            logger.error(f"V3 synthesis failed: {e}")
            return await self._fallback_synthesis(text, emotional_context.get("jeffrey_emotion", "neutral"))

    async def _fallback_synthesis(self, text: str, emotion: str) -> dict[str, Any]:
        """Fallback to V2 synthesis"""
        try:
            audio_data = await self.v2_controller.voice_engine.synthesize_speech(text, emotion)
            return {
                "audio_data": audio_data,
                "text": text,
                "emotion_used": emotion,
                "v3_enhanced": False,
                "synthesis_method": "elevenlabs_v2_fallback",
            }
        except Exception as e:
            logger.error(f"Fallback synthesis failed: {e}")
            return {"error": str(e), "text": text, "synthesis_method": "failed"}

    async def _emergency_fallback(
        self,
        audio_input: bytes = None,
        text_input: str = None,
        brain_context: dict[str, Any] = None,
    ) -> dict[str, Any]:
        """Emergency fallback to V2 controller"""
        logger.warning("🚨 Using emergency fallback to V2 controller")

        try:
            if hasattr(self.v2_controller, "process_voice_interaction"):
                return await self.v2_controller.process_voice_interaction(audio_input, text_input, brain_context)
            else:
                # Basic fallback response
                return {
                    "success": False,
                    "error": "Voice processing temporarily unavailable",
                    "fallback_response": "Je rencontre des difficultés techniques. Veuillez réessayer.",
                }
        except Exception as e:
            logger.error(f"Emergency fallback failed: {e}")
            return {
                "success": False,
                "error": "Complete voice system failure",
                "fallback_response": "Système vocal temporairement indisponible.",
            }

    def _update_performance_stats(self, result: dict[str, Any]):
        """Update performance statistics"""
        self.v3_performance["total_interactions"] += 1

        if result.get("voice_output", {}).get("v3_enhanced", False):
            self.v3_performance["v3_enhanced_interactions"] += 1

        # Update average response time
        response_time = result.get("performance", {}).get("total_time", 0.0)
        total = self.v3_performance["total_interactions"]
        current_avg = self.v3_performance["average_response_time"]

        self.v3_performance["average_response_time"] = (current_avg * (total - 1) + response_time) / total

    def get_v3_capabilities(self) -> dict[str, Any]:
        """Get V3 capabilities and status"""
        return {
            "v3_available": self.v3_engine.v3_available,
            "available_personas": self.v3_engine.get_available_personas(),
            "audio_tags_enabled": True,
            "emotional_analysis_enabled": self.use_v3_enhancement,
            "performance_stats": {
                **self.v3_performance,
                "v3_engine_stats": self.v3_engine.get_performance_stats(),
            },
        }

    async def test_v3_emotional_range(self) -> dict[str, Any]:
        """Test V3 emotional expression capabilities"""
        return await self.v3_engine.test_emotional_expressions()

    def configure_v3_settings(self, settings: dict[str, Any]):
        """Configure V3 settings"""
        if "use_v3_enhancement" in settings:
            self.use_v3_enhancement = settings["use_v3_enhancement"]

        if "auto_persona_selection" in settings:
            self.auto_persona_selection = settings["auto_persona_selection"]

        if "emotional_continuity" in settings:
            self.emotional_continuity = settings["emotional_continuity"]

        logger.info(f"V3 settings updated: {settings}")

    async def switch_persona(self, persona: str | VoicePersona):
        """Manually switch voice persona"""
        if isinstance(persona, str):
            persona = VoicePersona(persona)

        self.v3_engine.set_voice_persona(persona)
        logger.info(f"Voice persona switched to: {persona.value}")


# Convenience functions for easy integration
async def create_v3_voice_interaction(
    text: str, emotion: str = "neutral", persona: VoicePersona = VoicePersona.FRIENDLY
) -> bytes:
    """Quick V3 voice synthesis"""
    controller = VoiceControllerV3()
    await controller.switch_persona(persona)

    result = await controller.process_voice_interaction(text_input=text)
    return result.get("voice_output", {}).get("audio_data", b"")


def get_v3_controller(data_dir: str = "data/voice") -> VoiceControllerV3:
    """Factory function for V3 controller"""
    return VoiceControllerV3(data_dir)
