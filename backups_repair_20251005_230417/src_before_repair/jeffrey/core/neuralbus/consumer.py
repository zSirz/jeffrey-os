"""
Event consumer with batching, circuit breaker, and DLQ
Supports dynamic batch sizing and per-handler circuit breaking
"""

import asyncio
import json
import logging
import statistics
import uuid
from collections import deque
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

import nats
from nats.js.api import AckPolicy, ConsumerConfig

from .config import HAS_MSGPACK, config
from .contracts import CloudEvent
from .exceptions import CircuitOpen, ConsumerError

if HAS_MSGPACK:
    import msgpack

# Try to import LZ4 for decompression
HAS_LZ4 = False
try:
    import lz4.frame

    HAS_LZ4 = True
except ImportError:
    pass

logger = logging.getLogger(__name__)


class HandlerCircuitBreaker:
    """Async circuit breaker for handler protection"""

    def __init__(self, max_failures: int = 5, timeout: int = 60):
        self.max_failures = max_failures
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = 0.0
        self.state = "closed"  # closed, open, half-open

    async def call(self, coro_func: Callable[..., Awaitable[Any]], *args, **kwargs):
        """Execute async function with circuit breaker protection"""
        loop = asyncio.get_event_loop()
        now = loop.time()

        # Check circuit state
        if self.state == "open":
            if now - self.last_failure_time > self.timeout:
                # Try half-open
                self.state = "half-open"
                logger.info(f"Circuit half-open for {coro_func.__name__}")
            else:
                raise CircuitOpen(f"Circuit open for {coro_func.__name__}")

        try:
            # Execute the coroutine
            result = await coro_func(*args, **kwargs)

            # Success - maybe close circuit
            if self.state == "half-open":
                self.state = "closed"
                self.failures = 0
                logger.info(f"Circuit closed for {coro_func.__name__}")

            return result

        except Exception:
            self.failures += 1
            self.last_failure_time = now

            if self.failures >= self.max_failures:
                self.state = "open"
                logger.error(f"Circuit opened for {coro_func.__name__} after {self.failures} failures")

            raise


class DynamicBatchSizer:
    """Auto-adjust batch size based on throughput"""

    def __init__(self):
        self.batch_size = config.BATCH_SIZE
        self.throughput_history = deque(maxlen=100)
        self.last_adjustment = 0.0

    def record(self, batch_size: int, duration: float):
        """Record batch processing metrics"""
        if duration > 0:
            throughput = batch_size / duration
            self.throughput_history.append(throughput)

    def get_optimal_size(self) -> int:
        """Calculate optimal batch size using EMA"""
        if not config.BATCH_DYNAMIC:
            return config.BATCH_SIZE

        if len(self.throughput_history) < 10:
            return self.batch_size

        loop = asyncio.get_event_loop()
        now = loop.time()

        # Adjust every 10 seconds
        if now - self.last_adjustment < 10:
            return self.batch_size

        # Calculate average throughput
        avg_throughput = statistics.mean(self.throughput_history)

        # Adjust batch size based on throughput
        if avg_throughput > 100:  # High throughput
            self.batch_size = min(self.batch_size + 10, config.BATCH_SIZE_MAX)
        elif avg_throughput < 20:  # Low throughput
            self.batch_size = max(self.batch_size - 10, config.BATCH_SIZE_MIN)

        self.last_adjustment = now
        logger.debug(f"Batch size adjusted to {self.batch_size} (throughput: {avg_throughput:.1f}/s)")

        return self.batch_size


class NeuralConsumer:
    """High-performance event consumer with resilience features"""

    def __init__(self, consumer_name: str, subject_filter: str):
        self.consumer_name = consumer_name
        self.subject_filter = subject_filter
        self.nc: nats.NATS | None = None
        self.js: nats.JetStreamContext | None = None

        # Handlers and circuit breakers
        self._handlers: dict[str, list[Callable]] = {}
        self._circuit_breakers: dict[str, HandlerCircuitBreaker] = {}

        # Dynamic batching
        self.batch_sizer = DynamicBatchSizer()

        # Control
        self._running = False

    async def connect(self):
        """Connect to NATS and ensure consumer exists"""
        self.nc = await nats.connect(
            servers=[config.NATS_URL],
            user=config.NATS_USER,
            password=config.NATS_PASSWORD,
            name=f"jeffrey-consumer-{self.consumer_name}",
        )
        self.js = self.nc.jetstream()

        # Ensure consumer exists
        await self._ensure_consumer()

        logger.info(f"Consumer '{self.consumer_name}' connected")

    async def _ensure_consumer(self):
        """Create consumer if it doesn't exist"""
        try:
            # Check if exists
            await self.js.consumer_info(config.STREAM_NAME, self.consumer_name)
            logger.debug(f"Consumer {self.consumer_name} already exists")
        except:
            # Create new consumer
            cfg = ConsumerConfig(
                durable_name=self.consumer_name,
                filter_subject=self.subject_filter,
                ack_policy=AckPolicy.EXPLICIT,
                ack_wait=config.ACK_WAIT,  # Already in seconds
                max_deliver=config.MAX_DELIVER,
                max_ack_pending=config.MAX_ACK_PENDING,
            )

            await self.js.add_consumer(stream=config.STREAM_NAME, config=cfg)
            logger.info(f"Consumer {self.consumer_name} created")

    def register_handler(self, event_type: str, handler: Callable):
        """Register event handler with circuit breaker"""
        if event_type not in self._handlers:
            self._handlers[event_type] = []

        self._handlers[event_type].append(handler)

        # Create circuit breaker for this handler
        if config.CIRCUIT_BREAKER_ENABLED:
            breaker_key = f"{event_type}:{handler.__name__}"
            self._circuit_breakers[breaker_key] = HandlerCircuitBreaker(
                max_failures=config.CIRCUIT_BREAKER_MAX_FAILURES,
                timeout=config.CIRCUIT_BREAKER_TIMEOUT,
            )

        logger.info(f"Handler {handler.__name__} registered for {event_type}")

    async def _process_message(self, msg) -> bool:
        """Process single message with error handling and decompression"""
        try:
            # Check if message is compressed and decompress if needed
            msg = await self._decompress_if_needed(msg)

            # Deserialize based on content type
            event = await self._deserialize_message(msg)

            # Security check
            if config.REQUIRE_TENANT_ID:
                if not self._validate_tenant(event, msg):
                    await msg.term()  # Terminate invalid message
                    return False

            # Find handlers
            handlers = self._get_handlers(event.meta.type)

            if not handlers:
                logger.warning(f"No handler for {event.meta.type}")
                await msg.ack()  # Ack to avoid redelivery
                return True

            # Get headers (protect against None)
            headers = msg.headers or {}

            # Execute handlers with circuit breaker
            for handler in handlers:
                await self._execute_handler(handler, event, headers)

            await msg.ack()
            return True

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)

            # Check if should go to DLQ
            if msg.metadata.num_delivered >= config.MAX_DELIVER:
                await self._send_to_dlq(msg, str(e))
                await msg.term()
            else:
                await msg.nak(delay=5)  # Retry after 5 seconds

            return False

    async def _decompress_if_needed(self, msg):
        """Decompress message if LZ4 compression is detected"""
        # Check compression header
        if msg.headers and msg.headers.get("content-encoding") == "lz4":
            if HAS_LZ4:
                try:
                    # Decompress the message data
                    decompressed_data = lz4.frame.decompress(msg.data)

                    # Create a new message-like object with decompressed data
                    # Keep all other properties the same
                    class DecompressedMsg:
                        def __init__(self, original_msg, new_data):
                            self.data = new_data
                            self.headers = original_msg.headers
                            self.subject = original_msg.subject
                            self.metadata = original_msg.metadata
                            self.ack = original_msg.ack
                            self.nak = original_msg.nak
                            self.term = original_msg.term

                    msg = DecompressedMsg(msg, decompressed_data)
                    logger.debug(f"Decompressed message from {len(msg.data)} bytes")
                except Exception as e:
                    logger.error(f"Decompression failed: {e}")
                    # Continue with compressed data if decompression fails
            else:
                logger.warning("Message is LZ4 compressed but lz4 not available")

        return msg

    async def _deserialize_message(self, msg) -> CloudEvent:
        """Deserialize message based on content type"""
        # Protect against None headers
        headers = msg.headers or {}
        content_type = headers.get("Content-Type", "application/json")

        if "msgpack" in content_type and HAS_MSGPACK:
            data = msgpack.unpackb(msg.data, raw=False)
        else:
            data = json.loads(msg.data.decode("utf-8"))

        return CloudEvent(**data)

    def _validate_tenant(self, event: CloudEvent, msg) -> bool:
        """Validate tenant isolation"""
        # Protect against None headers
        headers = msg.headers or {}
        expected_tenant = headers.get("X-Tenant-Id")
        if expected_tenant and event.meta.tenant_id != expected_tenant:
            logger.error(f"Tenant mismatch: {event.meta.tenant_id} != {expected_tenant}")
            return False
        return True

    def _get_handlers(self, event_type: str) -> list[Callable]:
        """Get handlers for event type, including wildcards"""
        handlers = []

        # Exact match
        handlers.extend(self._handlers.get(event_type, []))

        # Wildcard match
        handlers.extend(self._handlers.get("*", []))

        # Pattern match (e.g., "user.*" matches "user.created")
        for pattern, pattern_handlers in self._handlers.items():
            if pattern.endswith("*") and event_type.startswith(pattern[:-1]):
                handlers.extend(pattern_handlers)

        return handlers

    async def _execute_handler(self, handler: Callable, event: CloudEvent, headers: dict):
        """Execute handler with circuit breaker"""
        breaker_key = f"{event.meta.type}:{handler.__name__}"
        breaker = self._circuit_breakers.get(breaker_key)

        if breaker and config.CIRCUIT_BREAKER_ENABLED:
            try:
                await breaker.call(handler, event, headers)
            except CircuitOpen:
                logger.warning(f"Circuit open, skipping {handler.__name__}")
        else:
            await handler(event, headers)

    async def _send_to_dlq(self, msg, error: str):
        """Send failed message to DLQ as CloudEvent"""
        try:
            # Parse original event
            try:
                # Protect against None headers
                headers = msg.headers or {}
                if "msgpack" in headers.get("Content-Type", ""):
                    original_data = msgpack.unpackb(msg.data, raw=False)
                else:
                    original_data = json.loads(msg.data.decode())
            except:
                original_data = {
                    "error": "Could not parse original",
                    "raw": msg.data.hex() if isinstance(msg.data, bytes) else str(msg.data),
                }

            # Create DLQ CloudEvent
            dlq_event = {
                "meta": {
                    "id": str(uuid.uuid4()),
                    "source": "jeffrey.neuralbus.dlq",
                    "spec_version": "1.0",
                    "type": f"dlq.{original_data.get('meta', {}).get('type', 'unknown')}",
                    "time": datetime.utcnow().isoformat() + "Z",
                    "tenant_id": original_data.get("meta", {}).get("tenant_id", "unknown"),
                    "priority": "low",
                    "jeffrey_copyright": "proprietary-jeffrey-os",
                },
                "data": {
                    "original": original_data,
                    "error": error,
                    "original_subject": msg.subject,
                    "attempts": msg.metadata.num_delivered,
                    "consumer": self.consumer_name,
                    "failed_at": datetime.utcnow().isoformat() + "Z",
                },
            }

            # Publish to DLQ
            dlq_subject = f"dlq.{msg.subject}"
            await self.js.publish(
                subject=dlq_subject,
                payload=json.dumps(dlq_event).encode(),
                headers={
                    "Content-Type": "application/json",
                    "X-Original-Subject": msg.subject,
                    "X-Failed-Attempts": str(msg.metadata.num_delivered),
                    "X-Consumer": self.consumer_name,
                },
            )

            logger.warning(f"Message sent to DLQ: {dlq_subject}")

        except Exception as e:
            logger.error(f"Failed to send to DLQ: {e}", exc_info=True)

    async def run(self):
        """Main consumer loop with dynamic batching"""
        if not self.js:
            raise ConsumerError("Consumer not connected")

        self._running = True

        # Create pull subscription
        psub = await self.js.pull_subscribe(self.subject_filter, durable=self.consumer_name)

        logger.info(f"Consumer {self.consumer_name} running on {self.subject_filter}")

        while self._running:
            try:
                # Get dynamic batch size
                batch_size = self.batch_sizer.get_optimal_size()

                # Fetch batch
                start_time = asyncio.get_event_loop().time()
                messages = await psub.fetch(batch=batch_size, timeout=config.BATCH_TIMEOUT)

                if messages:
                    # Process in parallel
                    tasks = [self._process_message(msg) for msg in messages]
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    # Record metrics
                    duration = asyncio.get_event_loop().time() - start_time
                    self.batch_sizer.record(len(messages), duration)

                    # Log stats
                    success_count = sum(1 for r in results if r is True)
                    logger.debug(f"Batch processed: {success_count}/{len(messages)} OK in {duration:.2f}s")

            except TimeoutError:
                # Normal - no messages available
                continue

            except Exception as e:
                logger.error(f"Consumer loop error: {e}", exc_info=True)
                await asyncio.sleep(1)

    async def stop(self):
        """Stop consumer"""
        self._running = False
        if self.nc:
            await self.nc.close()
        logger.info(f"Consumer {self.consumer_name} stopped")
