"""
# VOCAL RECOVERY - PROVENANCE HEADER
# Module: streaming_audio_pipeline.py
# Source: Jeffrey_OS/src/storage/backups/pre_reorganization/old_versions/Jeffrey/Jeffrey_DEV_FIX/Jeffrey_LIVE/core/voice/streaming_audio_pipeline.py
# Hash: 21c58518ec2726eb
# Score: 2590
# Classes: AudioFormat, StreamingConfig, AudioChunk, StreamingAudioProcessor, IntelligentAudioCache, StreamingAudioPipeline
# Recovered: 2025-08-08T11:33:29.803610
# Tier: TIER2_CORE
"""

from __future__ import annotations

#!/usr/bin/env python3
"""
🚀 Streaming Audio Pipeline - Jeffrey's Ultra-Fast Audio System
==============================================================

Pipeline audio streaming ultra-rapide avec FFmpeg optimisé, élimination des
fichiers temporaires et buffering intelligent pour atteindre <500ms de latence.

Optimisations:
- Pipeline FFmpeg direct avec pipes (pas de fichiers temporaires)
- Streaming audio par chunks avec buffering minimal
- Conversion et lecture parallélisées
- Cache intelligent intégré
- Monitoring des performances temps réel
"""

import asyncio
import hashlib
import json
import logging
import subprocess
import time
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AudioFormat(Enum):
    """Formats audio supportés"""

    MP3 = "mp3"
    WAV = "wav"
    PCM = "pcm_s16le"
    AAC = "aac"


@dataclass
class StreamingConfig:
    """Configuration du pipeline streaming"""

    target_format: AudioFormat = AudioFormat.WAV
    sample_rate: int = 22050  # Optimisé pour voix
    channels: int = 1  # Mono pour réduire la taille
    bitrate: str = "64k"  # Qualité suffisante pour voix
    buffer_size: int = 4096  # Buffer minimal pour faible latence
    chunk_size: int = 1024  # Taille des chunks pour streaming
    max_latency_ms: int = 500  # Latence cible
    enable_compression: bool = True
    enable_normalization: bool = True


@dataclass
class AudioChunk:
    """Chunk audio pour streaming"""

    data: bytes
    timestamp: float
    chunk_id: int
    is_final: bool = False
    metadata: dict[str, Any] = None


class StreamingAudioProcessor:
    """
    🎵 Processeur audio streaming avec FFmpeg optimisé
    """

    def __init__(self, config: StreamingConfig = None) -> None:
        self.config = config or StreamingConfig()
        self.performance_stats = {
            "total_processed": 0,
            "average_latency_ms": 0.0,
            "cache_hits": 0,
            "chunks_processed": 0,
            "ffmpeg_calls": 0,
            "errors": 0,
        }

        # Vérifier la disponibilité de FFmpeg
        self.ffmpeg_available = self._check_ffmpeg_availability()

    def _check_ffmpeg_availability(self) -> bool:
        """Vérifier si FFmpeg est disponible"""
        try:
            result = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
            available = result.returncode == 0
            logger.info(f"🎵 FFmpeg available: {available}")
            return available
        except Exception as e:
            logger.warning(f"FFmpeg not available: {e}")
            return False

    async def convert_audio_streaming(
        self,
        input_data: bytes,
        input_format: AudioFormat,
        output_format: AudioFormat = None,
        on_chunk: Callable[[AudioChunk], None] = None,
    ) -> AsyncGenerator[AudioChunk, None]:
        """
        🚀 Conversion audio streaming ultra-rapide avec FFmpeg

        Args:
            input_data: Données audio d'entrée
            input_format: Format d'entrée
            output_format: Format de sortie (optionnel)
            on_chunk: Callback appelé pour chaque chunk

        Yields:
            AudioChunk: Chunks audio convertis en streaming
        """
        start_time = time.time()
        output_format = output_format or self.config.target_format

        if not self.ffmpeg_available:
            # Fallback sans conversion
            yield AudioChunk(
                data=input_data,
                timestamp=time.time(),
                chunk_id=0,
                is_final=True,
                metadata={"conversion": "bypass", "reason": "ffmpeg_unavailable"},
            )
            return

        try:
            # Commande FFmpeg optimisée pour streaming
            ffmpeg_cmd = self._build_ffmpeg_command(input_format, output_format)

            # Démarrer FFmpeg avec pipes
            process = await asyncio.create_subprocess_exec(
                *ffmpeg_cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Créer les tâches pour l'input et l'output
            input_task = asyncio.create_task(self._write_input_data(process, input_data))
            output_task = asyncio.create_task(self._read_output_chunks(process, on_chunk))

            # Attendre les résultats
            chunk_count = 0
            async for chunk in output_task:
                chunk.chunk_id = chunk_count
                chunk_count += 1
                yield chunk

            # Attendre la fin de l'input
            await input_task

            # Attendre la fin du processus
            await process.wait()

            # Statistiques
            processing_time = (time.time() - start_time) * 1000
            self.performance_stats["total_processed"] += 1
            self.performance_stats["ffmpeg_calls"] += 1
            self._update_latency_stats(processing_time)

            logger.debug(f"🎵 Streaming conversion completed in {processing_time:.1f}ms")

        except Exception as e:
            logger.error(f"❌ Streaming conversion failed: {e}")
            self.performance_stats["errors"] += 1

            # Fallback sans conversion
            yield AudioChunk(
                data=input_data,
                timestamp=time.time(),
                chunk_id=0,
                is_final=True,
                metadata={"conversion": "fallback", "error": str(e)},
            )

    def _build_ffmpeg_command(self, input_format: AudioFormat, output_format: AudioFormat) -> list[str]:
        """Construire la commande FFmpeg optimisée"""

        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]

        # Input configuration (from pipe)
        cmd.extend(["-f", input_format.value, "-i", "pipe:0"])

        # Audio filters pour optimisation
        filters = []

        if self.config.enable_normalization:
            filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")

        if self.config.enable_compression:
            filters.append("acompressor=threshold=0.1:ratio=3:attack=5:release=50")

        # Optimizations pour faible latence
        filters.append(f"aresample={self.config.sample_rate}")

        if filters:
            cmd.extend(["-af", ",".join(filters)])

        # Output configuration
        cmd.extend(
            [
                "-acodec",
                output_format.value,
                "-ar",
                str(self.config.sample_rate),
                "-ac",
                str(self.config.channels),
            ]
        )

        if output_format == AudioFormat.MP3:
            cmd.extend(["-b:a", self.config.bitrate])

        # Output format et optimisations streaming
        cmd.extend(
            [
                "-f",
                output_format.value,
                "-fflags",
                "+flush_packets",  # Force flushing pour streaming
                "-deadline",
                "realtime",  # Mode temps réel
                "-threads",
                "2",  # Limiter les threads pour latence
                "pipe:1",  # Output vers pipe
            ]
        )

        return cmd

    async def _write_input_data(self, process: asyncio.subprocess.Process, data: bytes):
        """Écrire les données d'entrée vers FFmpeg"""
        try:
            process.stdin.write(data)
            await process.stdin.drain()
            process.stdin.close()
            await process.stdin.wait_closed()
        except Exception as e:
            logger.error(f"❌ Error writing to FFmpeg: {e}")

    async def _read_output_chunks(
        self, process: asyncio.subprocess.Process, on_chunk: Callable[[AudioChunk], None] = None
    ) -> AsyncGenerator[AudioChunk, None]:
        """Lire les chunks de sortie de FFmpeg"""
        chunk_id = 0

        try:
            while True:
                chunk_data = await process.stdout.read(self.config.chunk_size)

                if not chunk_data:
                    break

                chunk = AudioChunk(data=chunk_data, timestamp=time.time(), chunk_id=chunk_id, is_final=False)

                if on_chunk:
                    on_chunk(chunk)

                yield chunk
                chunk_id += 1
                self.performance_stats["chunks_processed"] += 1

        except Exception as e:
            logger.error(f"❌ Error reading from FFmpeg: {e}")

        # Chunk final
        if chunk_id > 0:
            final_chunk = AudioChunk(data=b"", timestamp=time.time(), chunk_id=chunk_id, is_final=True)

            if on_chunk:
                on_chunk(final_chunk)

            yield final_chunk

    def _update_latency_stats(self, latency_ms: float):
        """Mettre à jour les statistiques de latence"""
        current_avg = self.performance_stats["average_latency_ms"]
        total_processed = self.performance_stats["total_processed"]

        new_avg = ((current_avg * (total_processed - 1)) + latency_ms) / total_processed
        self.performance_stats["average_latency_ms"] = new_avg

    def get_performance_stats(self) -> dict[str, Any]:
        """Obtenir les statistiques de performance"""
        return {
            **self.performance_stats,
            "config": {
                "target_format": self.config.target_format.value,
                "sample_rate": self.config.sample_rate,
                "channels": self.config.channels,
                "max_latency_ms": self.config.max_latency_ms,
            },
            "ffmpeg_available": self.ffmpeg_available,
        }


class IntelligentAudioCache:
    """
    🧠 Cache audio intelligent avec prédiction et pré-génération
    """

    def __init__(self, cache_dir: str = "data/voice/cache_v2") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Cache en mémoire pour accès ultra-rapide
        self.memory_cache: dict[str, bytes] = {}
        self.memory_cache_metadata: dict[str, dict] = {}

        # Configuration cache
        self.max_memory_cache_size = 50 * 1024 * 1024  # 50MB en RAM
        self.max_disk_cache_size = 500 * 1024 * 1024  # 500MB sur disque

        # Phrases communes à pré-cacher
        self.common_phrases = [
            "Bonjour David",
            "Comment puis-je t'aider ?",
            "D'accord",
            "Je comprends",
            "Bien sûr",
            "Laisse-moi réfléchir",
            "Un instant",
            "C'est une excellente question",
            "Je vais t'aider avec ça",
            "Voici ce que je pense",
        ]

        # Statistiques
        self.cache_stats = {
            "memory_hits": 0,
            "disk_hits": 0,
            "misses": 0,
            "pre_cached_items": 0,
            "total_size_mb": 0,
        }

        # Charger le cache existant
        self._load_existing_cache()

    def _generate_cache_key(self, text: str, voice_params: dict[str, Any]) -> str:
        """Générer une clé de cache unique"""
        content = f"{text}|{json.dumps(voice_params, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _load_existing_cache(self):
        """Charger le cache existant depuis le disque"""
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_key = cache_file.stem

                # Charger métadonnées
                metadata_file = self.cache_dir / f"{cache_key}.meta"
                if metadata_file.exists():
                    with open(metadata_file) as f:
                        metadata = json.load(f)

                    # Charger en mémoire si la phrase est commune
                    if metadata.get("text", "") in self.common_phrases:
                        with open(cache_file, "rb") as f:
                            audio_data = f.read()

                        self.memory_cache[cache_key] = audio_data
                        self.memory_cache_metadata[cache_key] = metadata

                        logger.debug(f"🔥 Pre-loaded to memory: {metadata.get('text', '')}")

            logger.info(f"💾 Loaded {len(self.memory_cache)} items to memory cache")

        except Exception as e:
            logger.warning(f"⚠️ Error loading cache: {e}")

    async def get_cached_audio(self, text: str, voice_params: dict[str, Any]) -> bytes | None:
        """Récupérer audio du cache (mémoire puis disque)"""
        cache_key = self._generate_cache_key(text, voice_params)

        # 1. Vérifier cache mémoire (ultra rapide)
        if cache_key in self.memory_cache:
            self.cache_stats["memory_hits"] += 1
            logger.debug(f"🔥 Memory cache hit: {text[:30]}...")
            return self.memory_cache[cache_key]

        # 2. Vérifier cache disque
        cache_file = self.cache_dir / f"{cache_key}.cache"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    audio_data = f.read()

                # Charger en mémoire si c'est une phrase fréquente
                if len(self.memory_cache) * 1024 < self.max_memory_cache_size:
                    self.memory_cache[cache_key] = audio_data

                self.cache_stats["disk_hits"] += 1
                logger.debug(f"💿 Disk cache hit: {text[:30]}...")
                return audio_data

            except Exception as e:
                logger.warning(f"⚠️ Error reading cache: {e}")

        # 3. Cache miss
        self.cache_stats["misses"] += 1
        return None

    async def cache_audio(self, text: str, voice_params: dict[str, Any], audio_data: bytes):
        """Mettre en cache l'audio généré"""
        cache_key = self._generate_cache_key(text, voice_params)

        # Métadonnées
        metadata = {
            "text": text,
            "voice_params": voice_params,
            "cached_at": time.time(),
            "size_bytes": len(audio_data),
            "is_common_phrase": text in self.common_phrases,
        }

        # Sauvegarder sur disque
        cache_file = self.cache_dir / f"{cache_key}.cache"
        metadata_file = self.cache_dir / f"{cache_key}.meta"

        try:
            with open(cache_file, "wb") as f:
                f.write(audio_data)

            with open(metadata_file, "w") as f:
                json.dump(metadata, f)

            # Ajouter en mémoire si c'est une phrase commune
            if text in self.common_phrases and len(self.memory_cache) * 1024 < self.max_memory_cache_size:
                self.memory_cache[cache_key] = audio_data
                self.memory_cache_metadata[cache_key] = metadata

            logger.debug(f"💾 Cached: {text[:30]}...")

        except Exception as e:
            logger.warning(f"⚠️ Error caching audio: {e}")

    async def pre_generate_common_phrases(self, voice_synthesizer: Callable):
        """Pré-générer les phrases communes en arrière-plan"""
        logger.info("🔥 Pre-generating common phrases...")

        for phrase in self.common_phrases:
            try:
                # Vérifier si déjà en cache
                cached = await self.get_cached_audio(phrase, {"emotion": "neutral"})
                if cached:
                    continue

                # Générer et cacher
                audio_data = await voice_synthesizer(phrase, "neutral")
                if audio_data:
                    await self.cache_audio(phrase, {"emotion": "neutral"}, audio_data)
                    self.cache_stats["pre_cached_items"] += 1

                # Délai pour ne pas surcharger l'API
                await asyncio.sleep(1)

            except Exception as e:
                logger.warning(f"⚠️ Error pre-generating '{phrase}': {e}")

        logger.info(f"🔥 Pre-generation completed: {self.cache_stats['pre_cached_items']} items")

    def get_cache_stats(self) -> dict[str, Any]:
        """Obtenir les statistiques du cache"""
        total_requests = self.cache_stats["memory_hits"] + self.cache_stats["disk_hits"] + self.cache_stats["misses"]

        hit_rate = 0
        if total_requests > 0:
            hit_rate = (self.cache_stats["memory_hits"] + self.cache_stats["disk_hits"]) / total_requests * 100

        return {
            **self.cache_stats,
            "total_requests": total_requests,
            "hit_rate_percent": hit_rate,
            "memory_cache_size": len(self.memory_cache),
            "disk_cache_files": len(list(self.cache_dir.glob("*.cache"))),
        }


class StreamingAudioPipeline:
    """
    🚀 Pipeline audio streaming complet - Ultra Performance
    """

    def __init__(self, config: StreamingConfig = None) -> None:
        self.config = config or StreamingConfig()
        self.processor = StreamingAudioProcessor(self.config)
        self.cache = IntelligentAudioCache()

        # Monitoring performance
        self.pipeline_stats = {
            "total_processed": 0,
            "average_total_latency_ms": 0.0,
            "cache_enabled": True,
            "streaming_enabled": True,
        }

    async def process_audio(
        self,
        audio_data: bytes,
        input_format: AudioFormat,
        output_format: AudioFormat = None,
        cache_key_params: dict[str, Any] = None,
    ) -> AsyncGenerator[AudioChunk, None]:
        """
        🎵 Traiter l'audio avec cache et streaming optimisés
        """
        start_time = time.time()

        # Vérifier le cache d'abord
        if cache_key_params and self.pipeline_stats["cache_enabled"]:
            cached_audio = await self.cache.get_cached_audio("", cache_key_params)
            if cached_audio:
                yield AudioChunk(
                    data=cached_audio,
                    timestamp=time.time(),
                    chunk_id=0,
                    is_final=True,
                    metadata={"source": "cache", "latency_ms": 0},
                )
                return

        # Traitement streaming
        output_format = output_format or self.config.target_format

        async for chunk in self.processor.convert_audio_streaming(audio_data, input_format, output_format):
            yield chunk

        # Statistiques
        total_latency = (time.time() - start_time) * 1000
        self._update_pipeline_stats(total_latency)

    def _update_pipeline_stats(self, latency_ms: float):
        """Mettre à jour les statistiques du pipeline"""
        self.pipeline_stats["total_processed"] += 1
        current_avg = self.pipeline_stats["average_total_latency_ms"]
        total_processed = self.pipeline_stats["total_processed"]

        new_avg = ((current_avg * (total_processed - 1)) + latency_ms) / total_processed
        self.pipeline_stats["average_total_latency_ms"] = new_avg

    def get_comprehensive_stats(self) -> dict[str, Any]:
        """Obtenir toutes les statistiques du pipeline"""
        return {
            "pipeline": self.pipeline_stats,
            "processor": self.processor.get_performance_stats(),
            "cache": self.cache.get_cache_stats(),
            "config": {
                "target_latency_ms": self.config.max_latency_ms,
                "streaming_enabled": True,
                "cache_enabled": True,
            },
        }


# Factory function pour faciliter l'usage
def create_streaming_pipeline(max_latency_ms: int = 500) -> StreamingAudioPipeline:
    """Créer un pipeline streaming optimisé"""
    config = StreamingConfig(max_latency_ms=max_latency_ms)
    return StreamingAudioPipeline(config)


# Fonction de test
async def test_streaming_pipeline():
    """Tester le pipeline streaming"""
    pipeline = create_streaming_pipeline()

    # Test data (simulé)
    test_audio = b"fake_audio_data" * 1000

    logger.info("🚀 Testing Streaming Audio Pipeline...")

    async for chunk in pipeline.process_audio(test_audio, AudioFormat.MP3, AudioFormat.WAV):
        logger.info(f"📦 Received chunk {chunk.chunk_id}: {len(chunk.data)} bytes")

    stats = pipeline.get_comprehensive_stats()
    logger.info(f"📊 Pipeline Stats: {stats}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(test_streaming_pipeline())
