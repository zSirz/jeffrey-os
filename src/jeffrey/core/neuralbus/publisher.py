"""
Event publisher with deduplication and retry logic
Supports Redis distributed dedup with LRU fallback
"""

import asyncio
import json
import logging
from collections import OrderedDict
from datetime import datetime, timedelta

import nats
from nats.errors import TimeoutError

from .config import HAS_MSGPACK, HAS_REDIS, config
from .contracts import CloudEvent, EventPriority
from .exceptions import PublishError

if HAS_MSGPACK:
    import msgpack

if HAS_REDIS:
    import redis.asyncio as aioredis

# Try to import LZ4 for compression
HAS_LZ4 = False
try:
    import lz4.frame

    HAS_LZ4 = True
except ImportError:
    pass

# Try orjson for faster serialization
HAS_ORJSON = False
try:
    import orjson

    HAS_ORJSON = True
except ImportError:
    pass

logger = logging.getLogger(__name__)


class NeuralPublisher:
    """High-performance event publisher"""

    def __init__(self):
        self.nc: nats.NATS | None = None
        self.js: nats.JetStreamContext | None = None

        # Local LRU cache for deduplication
        self._dedup_cache = OrderedDict()
        self._dedup_window = timedelta(minutes=config.DEDUP_WINDOW_MINUTES)

        # Redis for distributed deduplication
        self.redis: aioredis.Redis | None = None

        # Metrics for compression
        self._metrics = {"compressed_count": 0, "compression_saved_bytes": 0}

    async def connect(self):
        """Connect to NATS and optional Redis"""
        # Connect to NATS
        self.nc = await nats.connect(
            servers=[config.NATS_URL],
            user=config.NATS_USER,
            password=config.NATS_PASSWORD,
            max_reconnect_attempts=config.NATS_MAX_RECONNECT,
            reconnect_time_wait=config.NATS_RECONNECT_WAIT,
            connect_timeout=config.NATS_CONNECT_TIMEOUT,
            name="jeffrey-publisher",
            error_cb=self._error_cb,
            disconnected_cb=self._disconnected_cb,
            reconnected_cb=self._reconnected_cb,
        )
        self.js = self.nc.jetstream()
        logger.info(f"Publisher connected to NATS at {config.NATS_URL}")

        # Optional Redis connection
        if HAS_REDIS and config.USE_REDIS_DEDUP:
            try:
                self.redis = aioredis.from_url(config.REDIS_URL, decode_responses=True)
                await self.redis.ping()
                logger.info("Redis connected for distributed deduplication")
            except Exception as e:
                logger.warning(f"Redis unavailable, using local dedup: {e}")
                self.redis = None

    async def publish(self, event: CloudEvent, dedup: bool = True) -> str:
        """
        Publish event with deduplication and retry
        Returns: Event ID
        """
        if not self.js:
            raise PublishError("Publisher not connected")

        # Check deduplication
        if dedup and await self._is_duplicate(event.meta.id):
            logger.warning(f"Duplicate event ignored: {event.meta.id}")
            return event.meta.id

        # Serialize payload with compression if needed
        payload, headers = self._serialize_with_compression(event)

        # Determine subject based on priority
        subject = self._get_subject(event)

        # Build headers (merge with compression headers if any)
        base_headers = self._build_headers(event)
        headers.update(base_headers)

        # Publish with retry
        for attempt in range(config.MAX_DELIVER):
            try:
                ack = await self.js.publish(
                    subject=subject,
                    payload=payload,
                    headers=headers,
                    timeout=5.0,
                )

                # Mark as sent for deduplication
                await self._mark_as_sent(event.meta.id)

                logger.info(f"Event published: {event.meta.id} to {subject} (seq={ack.seq})")
                return event.meta.id

            except TimeoutError as e:
                wait_time = 2**attempt  # Exponential backoff
                logger.warning(
                    f"Publish timeout attempt {attempt + 1}/{config.MAX_DELIVER}, retrying in {wait_time}s: {e}"
                )
                await asyncio.sleep(wait_time)

        raise PublishError(f"Failed to publish after {config.MAX_DELIVER} attempts")

    def _serialize_with_compression(self, event: CloudEvent) -> tuple[bytes, dict[str, str]]:
        """Serialize event to bytes with intelligent compression"""
        headers = {}

        # Choose fastest serialization method
        if HAS_ORJSON:
            # orjson is ~3x faster than json
            payload_bytes = orjson.dumps(event.model_dump())
        elif HAS_MSGPACK and config.USE_MSGPACK:
            # Convert to dict with JSON-serializable datetime strings
            data = json.loads(event.model_dump_json())
            payload_bytes = msgpack.packb(data, use_bin_type=True)
        else:
            payload_bytes = event.model_dump_json().encode("utf-8")

        # COMPRESSION INTELLIGENTE (seulement si > 4KB)
        if HAS_LZ4 and len(payload_bytes) > 4096:
            try:
                compressed_bytes = lz4.frame.compress(payload_bytes)

                # Ne garder compression que si gain > 20%
                compression_ratio = len(compressed_bytes) / len(payload_bytes)
                if compression_ratio < 0.8:
                    saved_bytes = len(payload_bytes) - len(compressed_bytes)
                    self._metrics["compression_saved_bytes"] += saved_bytes
                    self._metrics["compressed_count"] += 1

                    # Use compressed version
                    payload_bytes = compressed_bytes
                    headers["content-encoding"] = "lz4"

                    # Log compression stats periodically
                    if self._metrics["compressed_count"] % 100 == 0:
                        logger.info(
                            f"Compression stats: {self._metrics['compressed_count']} messages, "
                            f"saved {self._metrics['compression_saved_bytes'] / 1024 / 1024:.2f} MB"
                        )

            except Exception as e:
                logger.warning(f"Compression failed: {e}")

        return payload_bytes, headers

    def _serialize(self, event: CloudEvent) -> bytes:
        """Legacy serialize method for compatibility"""
        payload, _ = self._serialize_with_compression(event)
        return payload

    def _get_subject(self, event: CloudEvent) -> str:
        """Determine subject based on priority"""
        if event.meta.priority == EventPriority.CRITICAL:
            return f"priority.critical.{event.meta.type}"
        elif event.meta.priority == EventPriority.HIGH:
            return f"priority.high.{event.meta.type}"
        else:
            return f"events.{event.meta.type}"

    def _build_headers(self, event: CloudEvent) -> dict[str, str]:
        """Build NATS headers"""
        content_type = "application/msgpack" if (HAS_MSGPACK and config.USE_MSGPACK) else "application/json"

        # Handle priority as string or enum
        priority_value = event.meta.priority.value if hasattr(event.meta.priority, "value") else event.meta.priority

        return {
            "Nats-Msg-Id": event.meta.id,  # NATS native deduplication
            "Content-Type": content_type,
            "X-Event-Type": event.meta.type,
            "X-Priority": priority_value,
            "X-Tenant-Id": event.meta.tenant_id,
            "X-Source": event.meta.source,
            "X-Jeffrey-Copyright": event.meta.jeffrey_copyright,
        }

    async def _is_duplicate(self, event_id: str) -> bool:
        """Check for duplicate using Redis first, then LRU"""
        # Try Redis first (distributed)
        if self.redis:
            try:
                ttl = config.DEDUP_WINDOW_MINUTES * 60
                key = f"neuralbus:dedup:{event_id}"

                # SETNX with expiry (atomic operation)
                result = await self.redis.set(
                    key,
                    "1",
                    nx=True,
                    ex=ttl,  # Only set if not exists  # Expire after TTL
                )

                if result is None:  # Key already exists
                    return True
                return False

            except Exception as e:
                logger.warning(f"Redis dedup check failed, using LRU: {e}")

        # Fallback to local LRU
        return self._check_lru(event_id)

    def _check_lru(self, event_id: str) -> bool:
        """Check local LRU cache"""
        now = datetime.utcnow()

        # Check if duplicate
        if event_id in self._dedup_cache:
            age = now - self._dedup_cache[event_id]
            if age < self._dedup_window:
                return True

        # Purge old entries if cache too large
        if len(self._dedup_cache) > config.DEDUP_CACHE_MAX:
            self._purge_dedup_cache()

        return False

    async def _mark_as_sent(self, event_id: str):
        """Mark event as sent in dedup cache"""
        now = datetime.utcnow()
        self._dedup_cache[event_id] = now
        self._dedup_cache.move_to_end(event_id)  # LRU order

    def _purge_dedup_cache(self):
        """Purge expired entries from LRU cache"""
        now = datetime.utcnow()
        cutoff = now - self._dedup_window

        # Keep only recent entries
        self._dedup_cache = OrderedDict((k, v) for k, v in self._dedup_cache.items() if v > cutoff)

        # Additional cap at max size
        while len(self._dedup_cache) > config.DEDUP_CACHE_MAX:
            self._dedup_cache.popitem(last=False)  # Remove oldest

    async def _error_cb(self, e):
        """NATS error callback"""
        logger.error(f"NATS error: {e}")

    async def _disconnected_cb(self):
        """NATS disconnected callback"""
        logger.warning("NATS disconnected")

    async def _reconnected_cb(self):
        """NATS reconnected callback"""
        logger.info("NATS reconnected")

    async def close(self):
        """Clean shutdown"""
        if self.redis:
            await self.redis.close()
        if self.nc:
            await self.nc.close()
        logger.info("Publisher closed")
