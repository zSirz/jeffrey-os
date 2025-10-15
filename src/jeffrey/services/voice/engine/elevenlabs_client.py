"""
ElevenLabs API Client for Jeffrey V3.

Clean, async wrapper around ElevenLabs API with proper error handling,
streaming support, and modern Python patterns.

Now uses ElevenLabsAdapter for all network communications.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from jeffrey.bridge.adapters import ElevenLabsAdapter

logger = logging.getLogger(__name__)


class ElevenLabsError(Exception):
    """Base exception for ElevenLabs API errors."""


class VoiceNotFoundError(ElevenLabsError):
    """Raised when a requested voice is not found."""


class APIQuotaError(ElevenLabsError):
    """Raised when API quota is exceeded."""


@dataclass
class VoiceSettings:
    """Voice settings for ElevenLabs synthesis."""

    stability: float = 0.5
    similarity_boost: float = 0.8
    style: float = 0.0
    use_speaker_boost: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API requests."""
        return {
            "stability": self.stability,
            "similarity_boost": self.similarity_boost,
            "style": self.style,
            "use_speaker_boost": self.use_speaker_boost,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VoiceSettings:
        """Create from dictionary."""
        return cls(
            stability=data.get("stability", 0.5),
            similarity_boost=data.get("similarity_boost", 0.8),
            style=data.get("style", 0.0),
            use_speaker_boost=data.get("use_speaker_boost", True),
        )


@dataclass
class Voice:
    """Represents an ElevenLabs voice."""

    voice_id: str
    name: str
    category: str = "premade"
    description: str = ""
    preview_url: str | None = None
    available_for_tiers: list[str] = None

    def __post_init__(self):
        if self.available_for_tiers is None:
            self.available_for_tiers = ["free", "starter", "creator", "pro"]


class ElevenLabsClient:
    """
    Async client for ElevenLabs API.

    Provides clean interface for text-to-speech generation with streaming
    support, voice management, and proper error handling.

    Now uses ElevenLabsAdapter for all network communications.
    """

    MAX_TEXT_LENGTH = 5000  # ElevenLabs limit

    def __init__(self, api_key: str, timeout: float = 30.0, subscription_tier: str = "free") -> None:
        """
        Initialize ElevenLabs client.

        Args:
            api_key: ElevenLabs API key
            timeout: Request timeout in seconds
            subscription_tier: Subscription tier for rate limiting
        """
        self.api_key = api_key
        self.timeout = timeout
        self.subscription_tier = subscription_tier
        self.adapter: ElevenLabsAdapter | None = None
        self._voices_cache: list[Voice] | None = None
        self._cache_ttl = 3600  # 1 hour cache for voices
        self._cache_timestamp = 0

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_adapter()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def _ensure_adapter(self):
        """Ensure ElevenLabs adapter is created."""
        if self.adapter is None:
            self.adapter = ElevenLabsAdapter(
                api_key=self.api_key,
                subscription_tier=self.subscription_tier,
                timeout=int(self.timeout),
            )
            await self.adapter.__aenter__()

    async def close(self):
        """Close the adapter."""
        if self.adapter:
            await self.adapter.__aexit__(None, None, None)
            self.adapter = None

    def _validate_text(self, text: str) -> str:
        """Validate and clean text for synthesis."""
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        # Clean text
        text = text.strip()

        # Check length
        if len(text) > self.MAX_TEXT_LENGTH:
            logger.warning(f"Text length {len(text)} exceeds limit {self.MAX_TEXT_LENGTH}, truncating")
            text = text[: self.MAX_TEXT_LENGTH]

        return text

    async def generate_speech_stream(
        self,
        text: str,
        voice_id: str,
        voice_settings: VoiceSettings | None = None,
        model_id: str = "eleven_multilingual_v2",
    ) -> AsyncIterator[bytes]:
        """
        Generate speech from text, yielding audio chunks as they arrive.

        Args:
            text: Text to synthesize
            voice_id: ElevenLabs voice ID
            voice_settings: Voice configuration
            model_id: Model to use for synthesis

        Yields:
            Audio chunks as bytes

        Raises:
            ElevenLabsError: If API request fails
        """
        await self._ensure_adapter()

        # Validate input
        text = self._validate_text(text)

        if voice_settings is None:
            voice_settings = VoiceSettings()

        try:
            async for chunk in self.adapter.text_to_speech_stream(
                text=text,
                voice_id=voice_id,
                model_id=model_id,
                voice_settings=voice_settings.to_dict(),
            ):
                yield chunk

        except Exception as e:
            logger.error(f"ElevenLabs streaming failed: {e}")
            raise ElevenLabsError(f"Streaming failed: {e}")

    async def generate_speech(
        self,
        text: str,
        voice_id: str,
        voice_settings: VoiceSettings | None = None,
        model_id: str = "eleven_multilingual_v2",
    ) -> bytes:
        """
        Generate speech from text, returning complete audio data.

        Args:
            text: Text to synthesize
            voice_id: ElevenLabs voice ID
            voice_settings: Voice configuration
            model_id: Model to use for synthesis

        Returns:
            Complete audio data as bytes

        Raises:
            ElevenLabsError: If API request fails
        """
        await self._ensure_adapter()

        # Validate input
        text = self._validate_text(text)

        if voice_settings is None:
            voice_settings = VoiceSettings()

        try:
            audio_data = await self.adapter.text_to_speech(
                text=text,
                voice_id=voice_id,
                model_id=model_id,
                voice_settings=voice_settings.to_dict(),
            )

            if audio_data is None:
                raise ElevenLabsError("No audio data received from API")

            return audio_data

        except Exception as e:
            logger.error(f"ElevenLabs synthesis failed: {e}")
            raise ElevenLabsError(f"Synthesis failed: {e}")

    async def get_voices(self, force_refresh: bool = False) -> list[Voice]:
        """
        Get available voices from ElevenLabs.

        Args:
            force_refresh: Force refresh from API instead of cache

        Returns:
            List of available voices

        Raises:
            ElevenLabsError: If API request fails
        """
        await self._ensure_adapter()

        # Check cache first
        current_time = time.time()
        if not force_refresh and self._voices_cache and current_time - self._cache_timestamp < self._cache_ttl:
            logger.debug("Returning cached voices")
            return self._voices_cache

        try:
            voice_data = await self.adapter.get_voices(force_refresh=force_refresh)

            voices = []
            for voice_info in voice_data:
                voice = Voice(
                    voice_id=voice_info["voice_id"],
                    name=voice_info["name"],
                    category=voice_info.get("category", "premade"),
                    description=voice_info.get("description", ""),
                    preview_url=voice_info.get("preview_url"),
                    available_for_tiers=voice_info.get("available_for_tiers", ["free", "starter", "creator", "pro"]),
                )
                voices.append(voice)

            # Update cache
            self._voices_cache = voices
            self._cache_timestamp = current_time

            logger.info(f"Retrieved {len(voices)} voices from ElevenLabs")
            return voices

        except Exception as e:
            logger.error(f"Failed to get voices: {e}")
            # Return cache if available
            if self._voices_cache:
                logger.warning("Using cached voices due to API error")
                return self._voices_cache
            raise ElevenLabsError(f"Failed to get voices: {e}")

    async def get_voice_by_name(self, name: str) -> Voice | None:
        """
        Get voice by name.

        Args:
            name: Voice name to search for

        Returns:
            Voice object if found, None otherwise
        """
        voices = await self.get_voices()

        for voice in voices:
            if voice.name.lower() == name.lower():
                return voice

        return None

    async def get_voice_by_id(self, voice_id: str) -> Voice | None:
        """
        Get voice by ID.

        Args:
            voice_id: Voice ID to search for

        Returns:
            Voice object if found, None otherwise
        """
        voices = await self.get_voices()

        for voice in voices:
            if voice.voice_id == voice_id:
                return voice

        return None

    async def get_user_info(self) -> dict[str, Any] | None:
        """
        Get user information and subscription details.

        Returns:
            User info dict or None if error
        """
        await self._ensure_adapter()

        try:
            return await self.adapter.get_user_info()
        except Exception as e:
            logger.error(f"Failed to get user info: {e}")
            return None

    async def get_subscription_info(self) -> dict[str, Any] | None:
        """
        Get subscription information including character limits.

        Returns:
            Subscription info dict or None if error
        """
        await self._ensure_adapter()

        try:
            return await self.adapter.get_subscription_info()
        except Exception as e:
            logger.error(f"Failed to get subscription info: {e}")
            return None

    async def check_api_availability(self) -> bool:
        """
        Check if ElevenLabs API is available.

        Returns:
            True if API is accessible
        """
        await self._ensure_adapter()

        try:
            return await self.adapter.check_api_availability()
        except Exception as e:
            logger.error(f"API availability check failed: {e}")
            return False
