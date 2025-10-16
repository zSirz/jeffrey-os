"""
ThoughtPipeline - Production-ready cognitive processing
Combines GPT's robustness with Grok's adaptivity
"""
import asyncio
import logging
import time
import hashlib
from collections import deque, defaultdict
from typing import Any, Dict, Optional, Set
from enum import Enum

from jeffrey.core.ports.memory_port import MemoryPort
from jeffrey.core.contracts.thoughts import create_thought, ThoughtState
from jeffrey.core.neuralbus.events import (
    make_event, EMOTION_DETECTED, MEMORY_STORED, THOUGHT_GENERATED
)

logger = logging.getLogger(__name__)

class CircuitBreakerState(Enum):
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery

class ThoughtPipeline:
    """
    Ultra-robust pipeline with production features:
    - Idempotence with hash-based deduplication
    - Exponential backoff retries
    - Circuit breaker pattern
    - Dead Letter Queue (DLQ)
    - Semaphore concurrency control
    - Metrics and observability
    """

    def __init__(
        self,
        bus,
        memory: Optional[MemoryPort],
        consciousness: Optional[Any],
        orchestrator=None,  # Will be set by orchestrator
        max_concurrency: int = 8,
        max_retries: int = 3,
        breaker_threshold: int = 5,
        breaker_timeout: float = 30.0,
        dlq_size: int = 1000
    ):
        self.bus = bus
        self.memory = memory
        self.consciousness = consciousness
        self.orchestrator = orchestrator

        # Concurrency control
        self.max_concurrency = max_concurrency
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.max_retries = max_retries

        # Idempotence tracking (per-event-type for efficiency) - GPT CORRECTION
        self._processed_hashes: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._hash_sets: Dict[str, Set[str]] = defaultdict(set)

        # Circuit breaker per component
        self.breakers = {
            "memory": self._create_breaker(breaker_threshold, breaker_timeout),
            "consciousness": self._create_breaker(breaker_threshold, breaker_timeout)
        }

        # Dead Letter Queue with metadata
        self._dlq: deque = deque(maxlen=dlq_size)

        # Metrics
        self.metrics = {
            "events_processed": 0,
            "events_skipped": 0,
            "events_failed": 0,
            "memories_stored": 0,
            "thoughts_generated": 0,
            "retries_total": 0,
            "circuit_opens": 0,
            "dlq_size": 0,
            "avg_latency_ms": 0,
            "p95_latency_ms": 0,
            "p99_latency_ms": 0
        }

        # Latency tracking
        self._latencies: deque = deque(maxlen=1000)

    def _create_breaker(self, threshold: int, timeout: float) -> Dict:
        """Create a circuit breaker configuration"""
        return {
            "state": CircuitBreakerState.CLOSED,
            "failure_count": 0,
            "success_count": 0,
            "threshold": threshold,
            "timeout": timeout,
            "last_failure": 0,
            "half_open_tests": 0
        }

    def _hash_event(self, event: Dict) -> str:
        """Generate deterministic hash for idempotence - FIXED VERSION"""
        # Priority: Use event ID if available
        if event.get("id"):
            return f"id:{event['id']}"

        # Fallback: Use content-based hash with request_id for uniqueness
        data = event.get("data", {})
        key_parts = [
            event.get("type", ""),
            data.get("request_id", ""),  # Critical for uniqueness under load
            data.get("text", ""),
            data.get("emotion", ""),
            str(data.get("timestamp", ""))  # Keep full timestamp precision
        ]
        key = "|".join(key_parts)
        return hashlib.sha256(key.encode()).hexdigest()[:32]  # Longer hash for better collision resistance

    def _is_duplicate(self, event: Dict) -> bool:
        """Check if event was already processed (per-event-type idempotence) - OPTIMIZED VERSION"""
        event_type = event.get("type", "unknown")
        event_hash = self._hash_event(event)

        # Check in type-specific hash set
        if event_hash in self._hash_sets[event_type]:
            self.metrics["events_skipped"] += 1
            return True

        # Add to type-specific tracking
        self._processed_hashes[event_type].append(event_hash)
        self._hash_sets[event_type].add(event_hash)

        # Clean eviction per event type - GPT CORRECTION
        while len(self._hash_sets[event_type]) > len(self._processed_hashes[event_type]):
            oldest = self._processed_hashes[event_type].popleft()
            self._hash_sets[event_type].discard(oldest)

        return False

    async def _call_with_breaker(
        self,
        breaker_name: str,
        func,
        *args,
        **kwargs
    ):
        """Execute function with circuit breaker protection"""
        breaker = self.breakers[breaker_name]

        # Check breaker state
        if breaker["state"] == CircuitBreakerState.OPEN:
            # Check if timeout has passed
            if time.time() - breaker["last_failure"] > breaker["timeout"]:
                breaker["state"] = CircuitBreakerState.HALF_OPEN
                breaker["half_open_tests"] = 0
                logger.info(f"🔌 Circuit breaker '{breaker_name}' entering HALF_OPEN")
            else:
                # GPT CORRECTION: court-circuit explicite
                raise Exception(f"Circuit breaker '{breaker_name}' is OPEN")

        try:
            # Execute function
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)

            # Record success
            breaker["success_count"] += 1
            if breaker["state"] == CircuitBreakerState.HALF_OPEN:
                breaker["half_open_tests"] += 1
                if breaker["half_open_tests"] >= 3:  # Require 3 successes to close
                    breaker["state"] = CircuitBreakerState.CLOSED
                    breaker["failure_count"] = 0
                    logger.info(f"✅ Circuit breaker '{breaker_name}' CLOSED")

            return result

        except Exception as e:
            # Record failure
            breaker["failure_count"] += 1
            breaker["last_failure"] = time.time()

            if breaker["failure_count"] >= breaker["threshold"]:
                breaker["state"] = CircuitBreakerState.OPEN
                self.metrics["circuit_opens"] += 1
                logger.error(f"⚡ Circuit breaker '{breaker_name}' OPENED after {breaker['failure_count']} failures")

            raise e

    async def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute with exponential backoff retry"""
        delay = 0.1  # Start with 100ms
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                self.metrics["retries_total"] += 1

                if attempt < self.max_retries - 1:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 10)  # Cap at 10 seconds
                    logger.debug(f"Retry {attempt + 1}/{self.max_retries} after {delay}s")

        raise last_exception

    def _send_to_dlq(self, event: Dict, error: str, exception: Optional[Exception] = None):
        """Send failed event to Dead Letter Queue"""
        dlq_entry = {
            "event": event,
            "error": error,
            "exception": str(exception) if exception else None,
            "timestamp": time.time(),
            "retry_count": event.get("_retry_count", 0)
        }

        self._dlq.append(dlq_entry)
        self.metrics["dlq_size"] = len(self._dlq)
        self.metrics["events_failed"] += 1

        logger.error(f"💀 Event sent to DLQ: {error}")

        # Notify orchestrator if available
        if self.orchestrator:
            asyncio.create_task(
                self.orchestrator.handle_dlq_event(dlq_entry)
            )

    async def on_emotion_detected(self, event: Dict):
        """
        Process emotion event with full production robustness
        """
        if self._is_duplicate(event):
            logger.debug(f"Skipping duplicate event: {event.get('id')}")
            return

        # GPT CORRECTION: éviter la "tempête" avec circuit breaker OPEN
        if self.breakers["memory"]["state"] == CircuitBreakerState.OPEN:
            self._send_to_dlq(event, "memory_breaker_open")
            return

        start_time = time.perf_counter()

        async with self.semaphore:
            try:
                self.metrics["events_processed"] += 1
                data = event.get("data", {})

                # Store in memory with circuit breaker
                if self.memory:
                    memory_entry = {
                        "text": data.get("text"),
                        "emotion": data.get("emotion"),
                        "confidence": data.get("confidence"),
                        "timestamp": data.get("timestamp"),
                        "tags": [data.get("emotion"), "emotion_event"],
                        "meta": {
                            "source": event.get("source"),
                            "request_id": data.get("request_id"),
                            "circadian_phase": await self._get_circadian_context()
                        }
                    }

                    # Store with retry and circuit breaker
                    async def store_memory():
                        return await self._call_with_breaker(
                            "memory",
                            self.memory.store,
                            memory_entry
                        )

                    success = await self._retry_with_backoff(store_memory)

                    if success:
                        self.metrics["memories_stored"] += 1

                        # Publish memory stored event
                        memory_event = make_event(
                            MEMORY_STORED,
                            {"entry": memory_entry, "success": success},
                            source="jeffrey.pipeline.memory"
                        )
                        await self.bus.publish(memory_event)

                        logger.debug(f"📝 Memory stored for emotion: {data.get('emotion')}")

                # Track latency
                latency = (time.perf_counter() - start_time) * 1000
                self._latencies.append(latency)
                self._update_latency_metrics()

            except Exception as e:
                self._send_to_dlq(event, f"emotion_processing_failed", e)

    async def on_memory_stored(self, event: Dict):
        """
        Generate thought from memory with full robustness
        """
        if self._is_duplicate(event):
            return

        # GPT CORRECTION: éviter tempête avec consciousness breaker
        if self.breakers["consciousness"]["state"] == CircuitBreakerState.OPEN:
            # génère une fallback-thought légère plutôt que DLQ
            thought = create_thought(
                state=ThoughtState.AWARE,
                summary="Consciousness paused (breaker open) - light fallback",
                mode="fallback_cb_open",
            )
            await self.bus.publish(make_event(THOUGHT_GENERATED, thought, source="jeffrey.pipeline.thought"))
            return

        start_time = time.perf_counter()

        async with self.semaphore:
            try:
                entry = event.get("data", {}).get("entry", {})

                # Generate thought with circuit breaker
                thought = None

                if self.consciousness:
                    # Retrieve recent memories for context
                    memories = []
                    if self.memory:
                        memories = self.memory.search("", limit=5)

                    async def generate_thought():
                        return await self._call_with_breaker(
                            "consciousness",
                            self._process_consciousness,
                            memories, entry
                        )

                    thought = await self._retry_with_backoff(generate_thought)
                else:
                    # Fallback thought generation
                    thought = create_thought(
                        state=ThoughtState.AWARE,
                        summary=f"Basic processing of {entry.get('emotion', 'emotion')}",
                        mode="fallback_no_consciousness"
                    )

                if thought:
                    self.metrics["thoughts_generated"] += 1

                    # Enrich with metadata
                    thought["pipeline_metadata"] = {
                        "latency_ms": (time.perf_counter() - start_time) * 1000,
                        "circuit_breaker_state": self.breakers["consciousness"]["state"].value
                    }

                    # Publish thought event
                    thought_event = make_event(
                        THOUGHT_GENERATED,
                        thought,
                        source="jeffrey.pipeline.thought"
                    )
                    await self.bus.publish(thought_event)

                    logger.info(f"💭 Thought generated: {thought.get('summary', '')[:50]}...")

                # Update latency
                latency = (time.perf_counter() - start_time) * 1000
                self._latencies.append(latency)
                self._update_latency_metrics()

            except Exception as e:
                self._send_to_dlq(event, "thought_generation_failed", e)

    async def _process_consciousness(self, memories, entry):
        """Process with consciousness, handling sync/async"""
        if hasattr(self.consciousness, 'process'):
            result = self.consciousness.process(memories)
            if asyncio.iscoroutine(result):
                return await result
            return result
        else:
            # Fallback
            return create_thought(
                state=ThoughtState.AWARE,
                summary=f"Processing {entry.get('emotion', 'unknown')} with {len(memories)} memories",
                emotion_context=entry.get("emotion")
            )

    async def _get_circadian_context(self) -> Optional[Dict]:
        """Get current circadian context if available"""
        if self.orchestrator and hasattr(self.orchestrator, 'get_circadian_state'):
            return await self.orchestrator.get_circadian_state()
        return None

    def _update_latency_metrics(self):
        """Calculate latency percentiles"""
        if not self._latencies:
            return

        sorted_latencies = sorted(self._latencies)
        n = len(sorted_latencies)

        self.metrics["avg_latency_ms"] = sum(sorted_latencies) / n
        self.metrics["p95_latency_ms"] = sorted_latencies[int(n * 0.95)]
        self.metrics["p99_latency_ms"] = sorted_latencies[int(n * 0.99)]

    def get_metrics(self) -> Dict:
        """Get current metrics for monitoring"""
        return {
            **self.metrics,
            "breakers": {
                name: {
                    "state": breaker["state"].value,
                    "failure_count": breaker["failure_count"]
                }
                for name, breaker in self.breakers.items()
            },
            "concurrency": {
                "max": self.max_concurrency,
                "available": self.semaphore._value
            }
        }

    def get_dlq(self, limit: int = 10) -> list:
        """Retrieve recent DLQ entries for inspection"""
        return list(self._dlq)[-limit:]