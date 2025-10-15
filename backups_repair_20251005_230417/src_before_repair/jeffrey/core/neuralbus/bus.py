"""
NeuralBus main orchestrator
Manages stream, publishers, consumers, and self-optimization
"""

import asyncio
import logging
from typing import Any

import nats
from nats.js.api import DiscardPolicy, RetentionPolicy, StorageType, StreamConfig

from .config import config
from .consumer import NeuralConsumer
from .contracts import CloudEvent
from .publisher import NeuralPublisher

logger = logging.getLogger(__name__)


class NeuralBus:
    """
    Main event bus orchestrator for Jeffrey OS
    Manages infrastructure, routing, and optimization
    """

    def __init__(self):
        self.publisher = NeuralPublisher()
        self.consumers: dict[str, NeuralConsumer] = {}
        self.nc: nats.NATS | None = None
        self.js: nats.JetStreamContext | None = None

        # Background tasks
        self._purge_task: asyncio.Task | None = None
        self._optimize_task: asyncio.Task | None = None

    async def initialize(self):
        """Initialize bus infrastructure"""
        logger.info("Initializing NeuralBus...")

        # Connect to NATS
        self.nc = await nats.connect(
            servers=[config.NATS_URL],
            user=config.NATS_USER,
            password=config.NATS_PASSWORD,
            name="jeffrey-neuralbus",
        )
        self.js = self.nc.jetstream()

        # Ensure stream exists with all subjects
        await self._ensure_stream()

        # Connect publisher
        await self.publisher.connect()

        # Start background tasks
        if config.ENABLE_SELF_OPTIMIZE:
            self._purge_task = asyncio.create_task(self._purge_loop())
            self._optimize_task = asyncio.create_task(self._optimize_loop())

        logger.info("NeuralBus initialized successfully")

    async def _ensure_stream(self):
        """Create or update stream with all required subjects"""
        cfg = StreamConfig(
            name=config.STREAM_NAME,
            subjects=[
                "events.>",  # Normal events
                "priority.>",  # Priority lanes
                "dlq.>",  # Dead letter queue
            ],
            storage=StorageType.FILE,
            retention=RetentionPolicy.LIMITS,
            max_age=config.STREAM_MAX_AGE_DAYS * 24 * 60 * 60,  # Convert days to seconds
            discard=DiscardPolicy.OLD,
            num_replicas=config.STREAM_REPLICAS,
            duplicate_window=config.STREAM_DUPLICATE_WINDOW,  # Already in seconds
        )

        try:
            # Check if stream exists
            info = await self.js.stream_info(config.STREAM_NAME)
            logger.info(f"Stream '{config.STREAM_NAME}' exists with {info.state.messages} messages")

            # Optionally update configuration
            try:
                await self.js.update_stream(config=cfg)
                logger.info(f"Stream '{config.STREAM_NAME}' configuration updated")
            except Exception as e:
                logger.debug(f"Stream update skipped: {e}")

        except:
            # Create new stream
            await self.js.add_stream(config=cfg)
            logger.info(f"Stream '{config.STREAM_NAME}' created")

    async def publish(self, event: CloudEvent, dedup: bool = True) -> str:
        """
        Publish an event to the bus

        Args:
            event: CloudEvent to publish
            dedup: Whether to check for duplicates

        Returns:
            Event ID
        """
        return await self.publisher.publish(event, dedup=dedup)

    def create_consumer(self, name: str, subject_filter: str) -> NeuralConsumer:
        """
        Create and register a new consumer

        Args:
            name: Consumer name (must be unique)
            subject_filter: NATS subject pattern to consume

        Returns:
            NeuralConsumer instance
        """
        if name in self.consumers:
            raise ValueError(f"Consumer '{name}' already exists")

        consumer = NeuralConsumer(name, subject_filter)
        self.consumers[name] = consumer

        logger.info(f"Consumer '{name}' created for {subject_filter}")
        return consumer

    def create_priority_consumers(self) -> tuple[NeuralConsumer, NeuralConsumer]:
        """
        Create standard consumers for normal and priority events

        Returns:
            Tuple of (normal_consumer, priority_consumer)
        """
        # Normal events consumer
        normal = self.create_consumer(config.CONSUMER_NAME, "events.>")

        # Priority events consumer
        priority = self.create_consumer(config.PRIORITY_CONSUMER_NAME, "priority.>")

        logger.info("Created standard consumers for normal and priority events")
        return normal, priority

    async def start_consumers(self):
        """Start all registered consumers"""
        for name, consumer in self.consumers.items():
            await consumer.connect()
            asyncio.create_task(consumer.run())
            logger.info(f"Consumer '{name}' started")

    async def _purge_loop(self):
        """Periodically purge dedup cache"""
        while True:
            try:
                await asyncio.sleep(config.PURGE_INTERVAL)

                # Purge publisher cache
                if hasattr(self.publisher, "_purge_dedup_cache"):
                    self.publisher._purge_dedup_cache()
                    logger.debug("Publisher dedup cache purged")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Purge loop error: {e}")

    async def _optimize_loop(self):
        """Self-optimization loop for performance tuning"""
        while True:
            try:
                await asyncio.sleep(config.OPTIMIZE_INTERVAL)

                # Collect metrics from consumers
                metrics = {}
                for name, consumer in self.consumers.items():
                    if hasattr(consumer.batch_sizer, "batch_size"):
                        metrics[name] = {
                            "batch_size": consumer.batch_sizer.batch_size,
                            "throughput": len(consumer.batch_sizer.throughput_history),
                        }

                if metrics:
                    logger.info(f"Optimization metrics: {metrics}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Optimize loop error: {e}")

    async def get_stream_info(self) -> dict[str, Any]:
        """Get stream statistics"""
        if not self.js:
            return {}

        try:
            info = await self.js.stream_info(config.STREAM_NAME)
            return {
                "messages": info.state.messages,
                "bytes": info.state.bytes,
                "first_seq": info.state.first_seq,
                "last_seq": info.state.last_seq,
                "consumer_count": info.state.consumer_count,
            }
        except Exception as e:
            logger.error(f"Failed to get stream info: {e}")
            return {}

    async def health_check(self) -> dict[str, Any]:
        """Health check for monitoring"""
        health = {
            "status": "healthy",
            "publisher_connected": self.publisher.nc is not None and not self.publisher.nc.is_closed,
            "consumers": {},
            "stream_info": {},
        }

        # Check consumers
        for name, consumer in self.consumers.items():
            health["consumers"][name] = {
                "connected": consumer.nc is not None and not consumer.nc.is_closed,
                "running": consumer._running,
            }

        # Get stream info
        health["stream_info"] = await self.get_stream_info()

        # Overall status
        if not health["publisher_connected"]:
            health["status"] = "degraded"

        return health

    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down NeuralBus...")

        # Cancel background tasks
        if self._purge_task:
            self._purge_task.cancel()
            try:
                await self._purge_task
            except asyncio.CancelledError:
                pass

        if self._optimize_task:
            self._optimize_task.cancel()
            try:
                await self._optimize_task
            except asyncio.CancelledError:
                pass

        # Stop consumers
        for name, consumer in self.consumers.items():
            await consumer.stop()
            logger.info(f"Consumer '{name}' stopped")

        # Close publisher
        await self.publisher.close()

        # Close NATS connection
        if self.nc:
            await self.nc.close()

        logger.info("NeuralBus shutdown complete")


# Singleton instance
neural_bus = NeuralBus()
