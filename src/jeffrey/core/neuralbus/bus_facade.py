"""
BusFacade - Progressive async migration with internal queue
Combines GPT's queue design with Grok's auto-switch on load
"""
import asyncio
import logging
import os
from typing import Callable, Dict, List, Awaitable, Optional, Any
from collections import defaultdict, deque
from datetime import datetime
import time

# TWEAK 1: psutil optionnel
try:
    import psutil  # type: ignore
except Exception:
    psutil = None

logger = logging.getLogger(__name__)

class BusFacade:
    """
    Facade for event bus with internal queue and progressive async
    Implements Strangler Fig pattern as Gemini suggests
    """

    def __init__(
        self,
        max_queue: int = 1000,
        async_threshold_cpu: float = 70.0,
        pruning_enabled: bool = True
    ):
        # Queue for async processing
        self._queue: asyncio.Queue = None  # Created when async enabled
        self._sync_queue: deque = deque(maxlen=max_queue)  # Sync fallback

        # Handlers registry
        self._handlers: Dict[str, List[Callable]] = defaultdict(list)

        # Worker task
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False

        # TWEAK 2: Mode async explicite en tests/dev
        self.async_enabled = os.getenv("JEFFREY_BUS_ASYNC", "0") == "1"
        self.async_threshold_cpu = async_threshold_cpu

        # Stats for monitoring with saturation tracking - GPT ENHANCEMENT
        self.stats = {
            "events_published": 0,
            "events_processed": 0,
            "events_dropped": 0,
            "async_switches": 0,
            "current_mode": "async" if self.async_enabled else "sync",
            "queue_size": 0,
            "queue_max_size": max_queue,
            "queue_saturation_pct": 0.0,
            "queue_high_water_mark": 0,
            "backpressure_activations": 0,
            "last_prune": None,
            "processing_rate_events_per_sec": 0.0,
            "avg_processing_latency_ms": 0.0
        }

        # Latency tracking for performance monitoring
        self._processing_times = deque(maxlen=100)
        self._last_stats_update = time.time()

        # Pruning settings (Grok's idea)
        self.pruning_enabled = pruning_enabled
        self.max_queue_size = max_queue

    def check_system_load(self) -> bool:
        """
        Auto-switch to async based on CPU load (Grok's optimization)
        Returns True if switched
        """
        # TWEAK 1: Fallback si psutil n'est pas disponible
        if not psutil:
            return False

        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent

            should_be_async = (cpu_percent > self.async_threshold_cpu or
                              memory_percent > 80)

            if should_be_async and not self.async_enabled:
                logger.warning(f"🚀 Auto-switching to ASYNC mode (CPU: {cpu_percent:.1f}%, MEM: {memory_percent:.1f}%)")
                self.async_enabled = True
                self.stats["async_switches"] += 1
                self.stats["current_mode"] = "async"

                # Initialize async queue if needed
                if not self._queue:
                    self._queue = asyncio.Queue(maxsize=self.max_queue_size)

                # Transfer sync queue to async
                while self._sync_queue:
                    try:
                        event = self._sync_queue.popleft()
                        self._queue.put_nowait(event)
                    except asyncio.QueueFull:
                        break

                return True

            elif not should_be_async and self.async_enabled and os.getenv("JEFFREY_BUS_ASYNC", "0") != "1":
                # Ne pas revenir en sync si forcé par env var
                logger.info(f"📉 Switching back to SYNC mode (CPU: {cpu_percent:.1f}%)")
                self.async_enabled = False
                self.stats["current_mode"] = "sync"

        except Exception as e:
            logger.warning(f"Load check failed: {e}")

        return False

    def is_saturated(self, threshold: float = 0.8) -> bool:
        """Check if queue is saturated and should trigger backpressure - GPT ENHANCEMENT"""
        if self.async_enabled and self._queue:
            current_size = self._queue.qsize()
            saturation = current_size / self.max_queue_size
            self.stats["queue_saturation_pct"] = saturation * 100
            self.stats["queue_high_water_mark"] = max(
                self.stats["queue_high_water_mark"], current_size
            )
            return saturation >= threshold
        return False

    def get_backpressure_delay(self) -> float:
        """Calculate adaptive backpressure delay based on queue saturation - GPT ENHANCEMENT"""
        if not self.is_saturated():
            return 0.0

        saturation = self.stats["queue_saturation_pct"] / 100
        # Exponential backpressure: 50ms at 80%, 200ms at 90%, 500ms at 95%
        if saturation >= 0.95:
            return 0.5
        elif saturation >= 0.9:
            return 0.2
        elif saturation >= 0.8:
            return 0.05
        return 0.0

    def subscribe(
        self,
        topic: str,
        handler: Callable[[dict], Awaitable[None] | None]
    ) -> Callable[[], None]:
        """
        Subscribe to an event topic
        Returns unsubscribe function
        """
        self._handlers[topic].append(handler)
        logger.debug(f"📥 Subscribed handler to {topic}")

        def unsubscribe():
            if handler in self._handlers[topic]:
                self._handlers[topic].remove(handler)
                logger.debug(f"📤 Unsubscribed handler from {topic}")

        return unsubscribe

    async def publish(self, event: dict):
        """
        Publish an event (sync or async based on mode)
        """
        self.stats["events_published"] += 1

        # Check if we should switch modes
        self.check_system_load()

        if self.async_enabled and self._queue:
            # Async mode with queue + backpressure tracking - GPT ENHANCEMENT
            try:
                self._queue.put_nowait(event)
                self.stats["queue_size"] = self._queue.qsize()

                # Update saturation metrics
                self.is_saturated()  # Updates saturation stats as side effect

            except asyncio.QueueFull:
                self.stats["events_dropped"] += 1
                self.stats["backpressure_activations"] += 1
                logger.warning(f"🚫 Queue full - backpressure activated for event type={event.get('type')}")

                # Try pruning if enabled
                if self.pruning_enabled:
                    await self._prune_queue()
        else:
            # Sync mode - process immediately
            await self._process_event(event)

    async def _process_event(self, event: dict):
        """Process a single event by calling handlers - GPT ENHANCEMENT with latency tracking"""
        start_time = time.perf_counter()

        try:
            topic = event.get("type")
            handlers = self._handlers.get(topic, [])

            for handler in handlers:
                try:
                    result = handler(event)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(f"Handler error for {topic}: {e}")

            # Track processing latency
            latency_ms = (time.perf_counter() - start_time) * 1000
            self._processing_times.append(latency_ms)

            # Update stats
            self.stats["events_processed"] += 1
            if self._processing_times:
                self.stats["avg_processing_latency_ms"] = sum(self._processing_times) / len(self._processing_times)

            # Calculate processing rate every few seconds
            now = time.time()
            if now - self._last_stats_update > 5.0:  # Update every 5 seconds
                time_diff = now - self._last_stats_update
                events_diff = self.stats["events_processed"]
                if time_diff > 0:
                    self.stats["processing_rate_events_per_sec"] = events_diff / time_diff
                self._last_stats_update = now

        except Exception as e:
            logger.exception(f"Event processing error: {e}")

    async def _worker(self):
        """Async worker that processes events from queue"""
        self._running = True
        logger.info("🔄 Event worker started")

        while self._running:
            try:
                # Wait for event with timeout to check running status
                event = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=1.0
                )

                await self._process_event(event)
                self._queue.task_done()

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")

    async def _prune_queue(self):
        """
        Prune oldest events if queue too full (Grok's idea)
        Only for async queue
        """
        if not self._queue or self._queue.qsize() < self.max_queue_size * 0.9:
            return

        logger.warning(f"🧹 Pruning queue (size: {self._queue.qsize()})")

        # Move to temp list
        temp = []
        while not self._queue.empty() and len(temp) < self.max_queue_size:
            try:
                temp.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        # Keep only newest 80%
        keep_count = int(len(temp) * 0.8)
        temp = temp[-keep_count:]

        # Put back
        for event in temp:
            try:
                self._queue.put_nowait(event)
            except asyncio.QueueFull:
                break

        self.stats["last_prune"] = datetime.now().isoformat()
        self.stats["events_dropped"] += len(temp) - keep_count

    def start(self):
        """Start the event worker if in async mode"""
        self._running = True
        if self.async_enabled and not self._worker_task:
            if not self._queue:
                self._queue = asyncio.Queue(maxsize=self.max_queue_size)
            self._worker_task = asyncio.create_task(self._worker())
            logger.info("✅ BusFacade started in ASYNC mode")
        else:
            logger.info("✅ BusFacade started in SYNC mode")

    async def stop(self, drain: bool = True):
        """Stop the event worker gracefully"""
        if drain and self._queue:
            await self._queue.join()

        self._running = False

        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

        logger.info("🛑 BusFacade stopped")

    def get_stats(self) -> Dict[str, Any]:
        """Get current stats for monitoring"""
        return {
            **self.stats,
            "running": self._running,
            "async_enabled": self.async_enabled,
            "handlers_registered": sum(len(h) for h in self._handlers.values()),
            "topics_active": list(self._handlers.keys()),
            "psutil_available": psutil is not None
        }