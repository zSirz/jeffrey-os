# Code migré vers EventBus V2 - Auto-généré
# Vérifier les handlers et la logique async

from jeffrey.core.event_bus import EventBus
from jeffrey.core.event_bus_helpers import event_handler

"\nEvent Bus System for Jeffrey\nCentral communication hub for all modules\n"
import asyncio
import inspect
import logging
import statistics
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class EventPriority(Enum):
    """Event priority levels for EventBus processing"""

    CRITICAL = 0
    NORMAL = 1


@dataclass
class Event:
    """Event data structure with correlation tracking"""

    name: str
    data: dict[str, Any]
    source: str
    timestamp: datetime = None
    priority: EventPriority = EventPriority.NORMAL
    correlation_id: str | None = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.correlation_id is None:
            self.correlation_id = str(uuid.uuid4())


class EventBusModule(ABC):
    """Interface for optional EventBus modules"""

    @abstractmethod
    @event_handler("generic_event")
    def on_event_published(self, event: Event):
        """Called when an event is published to the bus"""
        pass

    @abstractmethod
    @event_handler("generic_event")
    def on_event_processed(self, event: Event, duration: float):
        """Called when an event has been processed by all subscribers"""
        pass


class EventBus:
    """Production-ready central event bus for plugin communication"""

    def __init__(self):
        self._subscribers: dict[str, list[Callable]] = {}
        self._async_subscribers: dict[str, list[Callable]] = {}
        self._event_history: list[Event] = []
        self._max_history = 1000
        self._lock = asyncio.Lock()
        self._critical_queue: deque = deque()
        self._normal_queue: deque = deque()
        self._events_published = 0
        self._events_processed = 0
        self._events_per_type = defaultdict(int)
        self._processing_times = deque(maxlen=1000)
        self._modules: list[EventBusModule] = []
        logger.info("EventBus initialized with production features")

    async def subscribe(self, event_name: str, handler: Callable, is_async: bool = None):
        """Subscribe to an event with automatic detection of async handlers"""
        if is_async is None:
            is_async = inspect.iscoroutinefunction(handler)
            logger.debug(f"Auto-detected handler {handler.__name__} as {('async' if is_async else 'sync')}")
        async with self._lock:
            if is_async:
                if event_name not in self._async_subscribers:
                    self._async_subscribers[event_name] = []
                self._async_subscribers[event_name].append(handler)
                logger.debug(f"Async handler {handler.__name__} subscribed to event: {event_name}")
            else:
                if event_name not in self._subscribers:
                    self._subscribers[event_name] = []
                self._subscribers[event_name].append(handler)
                logger.debug(f"Sync handler {handler.__name__} subscribed to event: {event_name}")

    async def unsubscribe(self, event_name: str, handler: Callable):
        """Unsubscribe from an event"""
        async with self._lock:
            if event_name in self._subscribers and handler in self._subscribers[event_name]:
                self._subscribers[event_name].remove(handler)
            if event_name in self._async_subscribers and handler in self._async_subscribers[event_name]:
                self._async_subscribers[event_name].remove(handler)
            logger.debug(f"Handler unsubscribed from event: {event_name}")

    async def publish(self, event: Event):
        """Publish an event to the appropriate priority queue"""
        start_time = time.time()
        logger.debug(f"Publishing event: {event.name} from {event.source} [correlation_id: {event.correlation_id}]")
        async with self._lock:
            self._events_published += 1
            self._events_per_type[event.name] += 1
            if event.priority == EventPriority.CRITICAL:
                self._critical_queue.append(event)
            else:
                self._normal_queue.append(event)
        for module in self._modules:
            try:
                module.on_event_published(event)
            except Exception as e:
                logger.error(f"Error in module.on_event_published: {e}")
        await self._process_event(event, start_time)

    async def emit(self, event: Event):
        """Emit an event (alias for publish for backward compatibility)"""
        await self.publish_emotion_event(event, event_bus_instance=self.event_bus)

    async def _process_event(self, event: Event, start_time: float):
        """Process an event with all subscribers and track metrics"""
        logger.debug(f"Processing event: {event.name} [correlation_id: {event.correlation_id}]")
        async with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)
        if event.name in self._subscribers:
            for handler in self._subscribers[event.name]:
                try:
                    logger.debug(
                        f"Calling sync handler {handler.__name__} for {event.name} [correlation_id: {event.correlation_id}]"
                    )
                    result = handler(event)
                    if inspect.isawaitable(result):
                        logger.warning(
                            f"Sync handler {handler.__name__} returned awaitable, but was called synchronously"
                        )
                except Exception as e:
                    logger.error(
                        f"Error in sync handler {handler.__name__} for {event.name} [correlation_id: {event.correlation_id}]: {e}"
                    )
                    logger.error(traceback.format_exc())
        if event.name in self._async_subscribers:
            tasks = []
            for handler in self._async_subscribers[event.name]:
                try:
                    logger.debug(
                        f"Creating task for async handler {handler.__name__} for {event.name} [correlation_id: {event.correlation_id}]"
                    )
                    task = asyncio.create_task(handler(event))
                    tasks.append(task)
                except Exception as e:
                    logger.error(
                        f"Error creating task for async handler {handler.__name__} for {event.name} [correlation_id: {event.correlation_id}]: {e}"
                    )
                    logger.error(traceback.format_exc())
            if tasks:
                try:
                    logger.debug(
                        f"Waiting for {len(tasks)} async handlers to complete for {event.name} [correlation_id: {event.correlation_id}]"
                    )
                    await asyncio.gather(*tasks, return_exceptions=True)
                except Exception as e:
                    logger.error(
                        f"Error in async handler execution for {event.name} [correlation_id: {event.correlation_id}]: {e}"
                    )
                    logger.error(traceback.format_exc())
        await self._emit_wildcard(event)
        end_time = time.time()
        processing_duration = end_time - start_time
        async with self._lock:
            self._events_processed += 1
            self._processing_times.append(processing_duration)
        for module in self._modules:
            try:
                module.on_event_processed(event, processing_duration)
            except Exception as e:
                logger.error(f"Error in module.on_event_processed: {e}")

    async def _emit_wildcard(self, event: Event):
        """Emit event to wildcard subscribers with correlation tracking"""
        wildcard_event = "*"
        if wildcard_event in self._subscribers:
            for handler in self._subscribers[wildcard_event]:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Error in wildcard sync handler [correlation_id: {event.correlation_id}]: {e}")
        if wildcard_event in self._async_subscribers:
            tasks = []
            for handler in self._async_subscribers[wildcard_event]:
                try:
                    task = asyncio.create_task(handler(event))
                    tasks.append(task)
                except Exception as e:
                    logger.error(f"Error in wildcard async handler [correlation_id: {event.correlation_id}]: {e}")
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    async def get_history(self, event_name: str | None = None, limit: int = 100) -> list[Event]:
        """Get event history"""
        async with self._lock:
            if event_name:
                filtered = [e for e in self._event_history if e.name == event_name]
                return filtered[-limit:]
            else:
                return self._event_history[-limit:]

    async def clear_history(self):
        """Clear event history"""
        async with self._lock:
            self._event_history.clear()

    def get_subscribers_count(self, event_name: str) -> int:
        """Get number of subscribers for an event"""
        count = 0
        if event_name in self._subscribers:
            count += len(self._subscribers[event_name])
        if event_name in self._async_subscribers:
            count += len(self._async_subscribers[event_name])
        return count

    def get_metrics(self) -> dict[str, Any]:
        """Get comprehensive EventBus metrics"""
        processing_time_avg = 0.0
        if self._processing_times:
            processing_time_avg = statistics.mean(self._processing_times)
        active_subscribers = 0
        for handlers in self._subscribers.values():
            active_subscribers += len(handlers)
        for handlers in self._async_subscribers.values():
            active_subscribers += len(handlers)
        queue_size = len(self._critical_queue) + len(self._normal_queue)
        return {
            "events_published": self._events_published,
            "events_processed": self._events_processed,
            "active_subscribers": active_subscribers,
            "queue_size": queue_size,
            "events_per_type": dict(self._events_per_type),
            "processing_time_avg": processing_time_avg,
        }

    async def process_priority_queue(self):
        """Process events from priority queues (CRITICAL first, then NORMAL)"""
        while True:
            event = None
            start_time = time.time()
            async with self._lock:
                if self._critical_queue:
                    event = self._critical_queue.popleft()
                elif self._normal_queue:
                    event = self._normal_queue.popleft()
            if event:
                await self._process_event(event, start_time)
            else:
                await asyncio.sleep(0.01)

    def register_module(self, module: EventBusModule):
        """Register an optional module for event lifecycle callbacks"""
        if not isinstance(module, EventBusModule):
            raise TypeError("Module must implement EventBusModule interface")
        self._modules.append(module)
        logger.info(f"EventBusModule {module.__class__.__name__} registered")

    def unregister_module(self, module: EventBusModule):
        """Unregister an optional module"""
        if module in self._modules:
            self._modules.remove(module)
            logger.info(f"EventBusModule {module.__class__.__name__} unregistered")


event_bus = EventBus()


class Events:
    """Common event names used across the system"""

    SYSTEM_READY = "system.ready"
    SYSTEM_SHUTDOWN = "system.shutdown"
    PLUGIN_LOADED = "plugin.loaded"
    PLUGIN_UNLOADED = "plugin.unloaded"
    PLUGIN_ERROR = "plugin.error"
    EMOTION_CHANGED = "emotion.changed"
    EMOTION_INTENSITY_UPDATE = "emotion.intensity_update"
    EMOTION_STATE_CHANGE = "emotion.state_change"
    MEMORY_STORED = "memory.stored"
    MEMORY_RECALLED = "memory.recalled"
    MEMORY_CONSOLIDATED = "memory.consolidated"
    USER_MESSAGE = "conversation.user_message"
    ASSISTANT_RESPONSE = "conversation.assistant_response"
    CONVERSATION_CONTEXT_UPDATE = "conversation.context_update"
    RESPONSE_GENERATION = "conversation.response_generation"
    VOICE_STARTED = "voice.started"
    VOICE_COMPLETED = "voice.completed"
    VOICE_ERROR = "voice.error"
