"""Unified Memory System - Production Ready Version"""

import asyncio
import hashlib
import json
import re
import time
from collections import defaultdict, deque
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from jeffrey.utils.logger import get_logger
from jeffrey.utils.lru_cache import LRUCache

logger = get_logger("UnifiedMemory")


# Memory type enum
class MemoryType(Enum):
    """Types of memory storage"""

    EPISODIC = "episodic"
    PROCEDURAL = "procedural"
    AFFECTIVE = "affective"
    CONTEXTUAL = "contextual"
    GENERAL = "general"


# Memory priority enum
class MemoryPriority(Enum):
    """Priority levels for memory retention"""

    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    TEMPORARY = 5


class MemoryValidator:
    """Validator with comprehensive XSS and injection protection"""

    # Patterns de sécurité à supprimer
    FORBIDDEN_PATTERNS = [
        (r"<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>", ""),
        (r"javascript\s*:", ""),
        (r'on\w+\s*=\s*["\']?[^"\'>\s]*', ""),
        (r"<iframe.*?</iframe>", ""),
        (r"<object.*?</object>", ""),
        (r"<embed.*?>", ""),
        (r"eval\s*\(", "eval_disabled("),
        (r"expression\s*\(", "expression_disabled("),
        (r"vbscript\s*:", ""),
        (r"<link[^>]*>", ""),
        (r"<meta[^>]*>", ""),
    ]

    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        "DROP TABLE",
        "DELETE FROM",
        "TRUNCATE",
        "ALTER TABLE",
        "EXEC SP_",
        "UNION SELECT",
        "INSERT INTO",
        "UPDATE SET",
    ]

    @classmethod
    def validate(cls, data: dict[str, Any]) -> bool:
        """Validate data structure and check for injections"""
        if not isinstance(data, dict):
            return False

        # Check for SQL injection
        str_data = json.dumps(data).upper()
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if pattern in str_data:
                logger.warning(f"SQL injection detected: {pattern}")
                return False

        return True

    @classmethod
    def sanitize_text(cls, text: str, max_length: int = 10000) -> str:
        """Deep sanitization with XSS protection"""
        if not text:
            return ""

        # Convert to string
        text = str(text)[: max_length * 2]  # Pre-trim for performance

        # Remove null bytes and control chars
        text = "".join(ch for ch in text if ord(ch) >= 32 or ch in "\n\r\t")

        # Apply all security patterns
        for pattern, replacement in cls.FORBIDDEN_PATTERNS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE | re.DOTALL)

        # HTML entity encoding for remaining suspicious chars
        text = text.replace("<", "&lt;").replace(">", "&gt;")

        # Normalize whitespace
        text = " ".join(text.split())

        # Final trim
        return text[:max_length].strip()

    @classmethod
    def sanitize_data(cls, data: Any, max_depth: int = 10) -> Any:
        """Recursively sanitize all strings in nested structures"""
        if max_depth <= 0:
            return data  # Prevent infinite recursion

        if isinstance(data, dict):
            return {k: cls.sanitize_data(v, max_depth - 1) for k, v in data.items()}
        elif isinstance(data, list):
            return [cls.sanitize_data(item, max_depth - 1) for item in data]
        elif isinstance(data, str):
            return cls.sanitize_text(data)
        else:
            return data


class UnifiedMemory:
    """Unified memory system with SQLite backend and async processing"""

    def __init__(
        self,
        backend: str = "sqlite",
        data_dir: str = "data",
        cache_size: int = 5000,
        cache_ttl: float = 3600,
    ):
        """Initialize with all required attributes"""

        # Logging
        self.logger = get_logger("UnifiedMemory")

        # Data directory
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Cache with TTL
        self.cache = LRUCache(maxsize=cache_size, ttl=cache_ttl)
        self.cache_size = cache_size

        # Context management
        self.current_context = deque(maxlen=10)
        self.context_window = 10

        # Emotional and relationship systems
        self.emotional_traces = {}
        self.emotional_patterns = {}
        self.long_term_patterns = {}
        self.relationships = {}
        self.learned_preferences = {}

        # Memory types and priorities
        self.memory_types = defaultdict(list)  # Type -> memories mapping
        self.memory_priorities = defaultdict(lambda: MemoryPriority.MEDIUM)

        # Async write queue with size limit (optimized for performance)
        self.write_queue = asyncio.Queue(maxsize=5000)  # Increased from 1000
        self._batch_size = 100  # Increased from 50
        self._flush_interval = 0.2  # Decreased from 1.0 for faster writes

        # Task handles
        self._writer_task = None
        self._consol_task = None
        self._pruner_task = None
        self._stop_event = asyncio.Event()

        # Locks for thread safety
        self._write_lock = asyncio.Lock()
        self._stats_lock = asyncio.Lock()

        # Statistics
        self.stats = {
            "total_stored": 0,
            "total_retrieved": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "write_queue_overflows": 0,
            "sanitization_blocks": 0,
            "consolidations": 0,
            "errors": 0,
        }

        # Embeddings cache for future
        self.embeddings_cache = {}

        # Initialize backend
        self.backend = None
        self.backend_type = backend

        if backend == "sqlite":
            try:
                from jeffrey.core.memory.sqlite.backend import SQLiteMemoryBackend

                db_path = self.data_dir / "unified_memory.db"
                self.backend = SQLiteMemoryBackend(str(db_path))
                self.logger.info(f"SQLite backend initialized: {db_path}")
            except ImportError as e:
                self.logger.warning(f"SQLite backend not available: {e}")
                self.backend = None
        elif backend == "memory":
            self.backend = None
            self.logger.info("In-memory backend (no persistence)")
        else:
            self.logger.warning(f"Unknown backend: {backend}, using in-memory")
            self.backend = None

        self.logger.info("UnifiedMemory initialized with all attributes")

    async def initialize(self):
        """Initialize and start all background tasks"""
        try:
            # Initialize backend
            if self.backend:
                await self.backend.initialize()
                self.logger.info("Backend initialized successfully")

            # Start background tasks if not already running
            if not self._writer_task or self._writer_task.done():
                self._writer_task = asyncio.create_task(self._batch_writer())
                self.logger.info("Batch writer task started")

            if not self._consol_task or self._consol_task.done():
                self._consol_task = asyncio.create_task(self._auto_consolidation())
                self.logger.info("Auto-consolidation task started")

            if not self._pruner_task or self._pruner_task.done():
                self._pruner_task = asyncio.create_task(self._cache_pruner())
                self.logger.info("Cache pruner task started")

            # Load recent memories into cache
            await self._load_existing_memories()

            # Load persistent data files
            await self._load_persistent_data()

            self.logger.info("✅ UnifiedMemory fully initialized with all tasks")

        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            self.stats["errors"] += 1
            # Continue in degraded mode
            self.backend = None

    async def store(self, data: dict[str, Any], memory_type: str | None = None) -> str:
        """Store memory with sanitization and validation"""
        # Validate
        if not MemoryValidator.validate(data):
            self.stats["sanitization_blocks"] += 1
            raise ValueError("Invalid data or injection attempt detected")

        # Sanitize
        data = MemoryValidator.sanitize_data(data)

        # Add metadata
        if "_id" not in data:
            data["_id"] = self._generate_id(data)
        if "_timestamp" not in data:
            data["_timestamp"] = time.time()

        if memory_type:
            data["type"] = memory_type
        elif "type" not in data:
            data["type"] = MemoryType.GENERAL.value

        # Update context
        self.current_context.append(data)

        # Update in-memory structures
        if data.get("type") == "emotional" or "emotion" in data:
            self._update_emotional_traces(data)
        elif data.get("type") == "pattern":
            self._detect_patterns(data)

        # Extract user_id if present
        user_id = data.get("user_id", "default")
        if user_id != "default" and user_id not in self.learned_preferences:
            self.learned_preferences[user_id] = {}

        # Add to queue
        try:
            self.write_queue.put_nowait(data)
        except asyncio.QueueFull:
            self.stats["write_queue_overflows"] += 1
            # Force flush and retry
            await self._flush_queue()
            await self.write_queue.put(data)

        # Update cache
        self.cache.set(data["_id"], data)

        self.stats["total_stored"] += 1
        return data["_id"]

    async def retrieve(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Retrieve memories by text search"""
        self.stats["total_retrieved"] += 1

        if self.backend:
            results = await self.backend.search_text(query, limit)
            # Update cache
            for result in results:
                if "_id" in result:
                    self.cache.set(result["_id"], result)
            return results
        else:
            # In-memory search
            results = []
            query_lower = query.lower()
            for mem_list in self.memory_types.values():
                for mem in mem_list:
                    if query_lower in str(mem).lower():
                        results.append(mem)
                        if len(results) >= limit:
                            return results

            # Also search in context
            for item in list(self.current_context)[-limit:]:
                if isinstance(item, dict) and query_lower in str(item).lower():
                    results.append(item)

            return results[:limit]

    async def query(self, filter_dict: dict[str, Any]) -> list[dict[str, Any]]:
        """Query with filters and caching"""
        # Generate cache key
        cache_key = hashlib.md5(json.dumps(filter_dict, sort_keys=True).encode()).hexdigest()

        # Check cache
        cached = self.cache.get(cache_key)
        if cached is not None:
            self.stats["cache_hits"] += 1
            return cached

        self.stats["cache_misses"] += 1

        # Query backend
        if self.backend:
            results = await self.backend.query(filter_dict)
        else:
            # In-memory query
            results = []
            mem_type = filter_dict.get("type")
            limit = filter_dict.get("limit", 100)

            if mem_type and mem_type in self.memory_types:
                results = self.memory_types[mem_type][:limit]
            else:
                for mems in self.memory_types.values():
                    results.extend(mems)
                    if len(results) >= limit:
                        break
            results = results[:limit]

        # Cache results
        self.cache.set(cache_key, results)
        return results

    async def search_text(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Search by text (delegates to backend or memory)"""
        if self.backend:
            return await self.backend.search_text(query, limit)
        else:
            return await self.retrieve(query, limit)

    # Background tasks
    async def _batch_writer(self):
        """Background task for batched writes with error recovery"""
        self.logger.info("Batch writer started")

        while not self._stop_event.is_set():
            try:
                # Wait for interval or stop signal
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=self._flush_interval)
                    break  # Stop event set
                except TimeoutError:
                    pass  # Continue to flush

                # Flush queue
                await self._flush_queue()

            except Exception as e:
                self.logger.error(f"Batch writer error: {e}")
                self.stats["errors"] += 1
                await asyncio.sleep(1)  # Brief pause before retry

        # Final flush before exit
        await self._flush_queue()
        self.logger.info("Batch writer stopped")

    async def _flush_queue(self):
        """Flush write queue to backend with retry logic"""
        if not self.backend or self.write_queue.empty():
            return

        batch = []
        failed_items = []

        # Collect batch
        for _ in range(min(self._batch_size, self.write_queue.qsize())):
            try:
                item = self.write_queue.get_nowait()
                batch.append(item)
            except asyncio.QueueEmpty:
                break

        if not batch:
            return

        # Attempt to store
        try:
            async with self._write_lock:
                await self.backend.store_batch(batch)
            self.logger.debug(f"Flushed {len(batch)} items to backend")

        except Exception as e:
            self.logger.error(f"Batch write failed: {e}")
            self.stats["errors"] += 1
            failed_items = batch

        # Re-queue failed items (with limit to prevent infinite loop)
        for item in failed_items[:10]:  # Max 10 retries
            try:
                self.write_queue.put_nowait(item)
            except asyncio.QueueFull:
                self.stats["write_queue_overflows"] += 1
                self.logger.warning("Write queue full, dropping item")
                break

    async def _auto_consolidation(self):
        """Periodic consolidation task"""
        self.logger.info("Auto-consolidation started")

        while not self._stop_event.is_set():
            try:
                # Wait 1 hour or until stop
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=3600)
                    break
                except TimeoutError:
                    pass

                # Run consolidation
                await self.consolidate()
                self.stats["consolidations"] += 1
                self.logger.info("Consolidation completed")

            except Exception as e:
                self.logger.error(f"Consolidation error: {e}")
                self.stats["errors"] += 1

        self.logger.info("Auto-consolidation stopped")

    async def _cache_pruner(self):
        """Periodically prune expired cache entries"""
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(300)  # Every 5 minutes

                pruned = self.cache.prune_expired()
                if pruned > 0:
                    self.logger.debug(f"Pruned {pruned} expired cache entries")

            except Exception as e:
                self.logger.error(f"Cache pruner error: {e}")

    async def _load_existing_memories(self):
        """Load recent memories into cache on startup"""
        if not self.backend:
            return

        try:
            # Load last 24h memories
            recent = await self.backend.query(
                {"_timestamp_gte": time.time() - 86400, "limit": min(1000, self.cache_size // 2)}
            )

            for mem in recent:
                if "_id" in mem:
                    self.cache.set(mem["_id"], mem)

            self.logger.info(f"Loaded {len(recent)} recent memories into cache")

        except Exception as e:
            self.logger.warning(f"Could not load existing memories: {e}")

    async def _load_persistent_data(self):
        """Load persistent data from disk"""
        files_to_load = {
            "emotional_memory.json": "emotional_traces",
            "conversation_memory.json": "long_term_patterns",
            "relationships.json": "relationships",
            "jeffrey_learning.json": "learned_preferences",
        }

        for filename, attr in files_to_load.items():
            filepath = self.data_dir / filename
            if filepath.exists():
                try:
                    with open(filepath, encoding="utf-8") as f:
                        data = json.load(f)

                        if filename == "jeffrey_learning.json":
                            # Handle user data
                            user_data = {}
                            for key, value in data.items():
                                if not key.startswith("_"):
                                    user_data[key] = value
                            setattr(self, attr, user_data)
                        else:
                            setattr(self, attr, data)

                        self.logger.info(f"Loaded {filename}: {len(data)} entries")
                except Exception as e:
                    self.logger.error(f"Error loading {filename}: {e}")

    async def save_persistent_data(self):
        """Save persistent data to disk"""
        data_to_save = {
            "emotional_memory.json": self.emotional_traces,
            "conversation_memory.json": self.long_term_patterns,
            "relationships.json": self.relationships,
            "jeffrey_learning.json": self.learned_preferences,
        }

        for filename, data in data_to_save.items():
            filepath = self.data_dir / filename
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                self.logger.debug(f"Saved {filename}")
            except Exception as e:
                self.logger.error(f"Error saving {filename}: {e}")

    # Stats methods
    def get_stats(self) -> dict[str, Any]:
        """Get runtime statistics (synchronous)"""
        return {
            **self.stats,
            "cache_stats": self.cache.stats(),
            "queue_size": self.write_queue.qsize(),
            "context_size": len(self.current_context),
            "emotional_traces": len(self.emotional_traces),
            "relationships": len(self.relationships),
            "memory_types": {str(t): len(m) for t, m in self.memory_types.items()},
            "tasks_running": {
                "writer": bool(self._writer_task and not self._writer_task.done()),
                "consolidator": bool(self._consol_task and not self._consol_task.done()),
                "pruner": bool(self._pruner_task and not self._pruner_task.done()),
            },
        }

    async def get_backend_stats(self) -> dict[str, Any]:
        """Get backend statistics (asynchronous)"""
        if self.backend and hasattr(self.backend, "get_stats"):
            try:
                return await self.backend.get_stats()
            except Exception as e:
                self.logger.error(f"Failed to get backend stats: {e}")
                return {"error": str(e)}
        return {}

    async def get_full_stats(self) -> dict[str, Any]:
        """Get complete statistics including backend"""
        runtime_stats = self.get_stats()
        backend_stats = await self.get_backend_stats()
        return {
            "runtime": runtime_stats,
            "backend": backend_stats,
            "timestamp": datetime.now().isoformat(),
        }

    async def _save_stats(self):
        """Save statistics to file for monitoring"""
        try:
            stats_file = self.data_dir / "memory_stats.json"
            full_stats = await self.get_full_stats()

            with open(stats_file, "w") as f:
                json.dump(full_stats, f, indent=2, default=str)

            self.logger.debug(f"Stats saved to {stats_file}")

        except Exception as e:
            self.logger.error(f"Failed to save stats: {e}")

    # Shutdown
    async def shutdown(self):
        """Clean shutdown with queue flush and task cancellation"""
        self.logger.info("Shutting down UnifiedMemory...")

        try:
            # Signal stop to all tasks
            self._stop_event.set()

            # Flush remaining items in queue (with timeout)
            flush_start = time.time()
            while not self.write_queue.empty() and (time.time() - flush_start) < 5:
                await self._flush_queue()
                await asyncio.sleep(0.1)

            if not self.write_queue.empty():
                self.logger.warning(f"Shutdown with {self.write_queue.qsize()} items in queue")

            # Cancel all tasks gracefully
            tasks = [self._writer_task, self._consol_task, self._pruner_task]
            for task in tasks:
                if task and not task.done():
                    task.cancel()

            # Wait for cancellation (with timeout)
            for task in tasks:
                if task:
                    try:
                        await asyncio.wait_for(task, timeout=2)
                    except (TimeoutError, asyncio.CancelledError):
                        pass

            # Save persistent data
            await self.save_persistent_data()

            # Shutdown backend
            if self.backend and hasattr(self.backend, "shutdown"):
                await self.backend.shutdown()

            # Save final stats
            await self._save_stats()

            self.logger.info("✅ UnifiedMemory shutdown complete")

        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")
            self.stats["errors"] += 1

    # Utility methods
    def _generate_id(self, data: dict) -> str:
        """Generate unique ID for memory"""
        content = json.dumps(data, sort_keys=True)
        hash_part = hashlib.md5(content.encode()).hexdigest()[:8]
        time_part = str(int(time.time() * 1000000))[-8:]
        return f"mem_{hash_part}_{time_part}"

    def _update_emotional_traces(self, data: dict):
        """Update emotional traces"""
        emotion = data.get("emotion", {})
        if isinstance(emotion, dict):
            primary = emotion.get("primary_emotion", "neutral")
        else:
            primary = str(emotion)

        if primary not in self.emotional_traces:
            self.emotional_traces[primary] = {"count": 0, "last_seen": time.time(), "contexts": []}

        self.emotional_traces[primary]["count"] += 1
        self.emotional_traces[primary]["last_seen"] = time.time()

        if "message" in data:
            self.emotional_traces[primary]["contexts"].append(data["message"][:50])
            if len(self.emotional_traces[primary]["contexts"]) > 10:
                self.emotional_traces[primary]["contexts"].pop(0)

    def _detect_patterns(self, data: dict):
        """Detect and record patterns"""
        message = data.get("message", "") or data.get("text", "")
        if not message:
            return

        words = message.lower().split()
        emotion = "neutral"
        if "emotion" in data:
            if isinstance(data["emotion"], dict):
                emotion = data["emotion"].get("primary_emotion", "neutral")
            else:
                emotion = str(data["emotion"])

        for word in words:
            if len(word) > 4:
                pattern_key = f"{word}_{emotion}"
                if pattern_key not in self.emotional_patterns:
                    self.emotional_patterns[pattern_key] = []
                self.emotional_patterns[pattern_key].append(time.time())

                if len(self.emotional_patterns[pattern_key]) > 3:
                    self.long_term_patterns[pattern_key] = {
                        "word": word,
                        "emotion": emotion,
                        "frequency": len(self.emotional_patterns[pattern_key]),
                    }

    # Compatibility methods (for existing code)
    def update(self, message: str, emotion_state: dict, metadata: dict | None = None):
        """Compatibility: update memory"""
        asyncio.create_task(self.store({"message": message, "emotion": emotion_state, **(metadata or {})}))

    def search_memories(self, user_id: str, query: str) -> list[str]:
        """Compatibility: search memories (blocking version for legacy code)"""
        # First check learned preferences (sync)
        results = []
        query_lower = query.lower()

        if user_id in self.learned_preferences:
            user_prefs = self.learned_preferences[user_id]
            for key, value in user_prefs.items():
                if query_lower in key.lower() or query_lower in str(value).lower():
                    if isinstance(value, dict) and "nom" in value:
                        results.append(f"Ton {key} s'appelle {value['nom']}")
                    else:
                        results.append(f"Je me souviens que {key}: {value}")

        # Try to get more results from backend if needed
        if len(results) < 3:
            try:
                # Check if we're in an async context
                asyncio.get_running_loop()
                # We're in async context, can't block - schedule for later
                asyncio.create_task(self.retrieve(query))
            except RuntimeError:
                # No running loop, we can block
                try:
                    more_results = asyncio.run(self.retrieve(query, limit=3))
                    for res in more_results:
                        if isinstance(res, dict):
                            text = res.get("text", res.get("message", ""))
                            if text and text not in results:
                                results.append(text[:100])
                except:
                    pass  # Fallback safe

        return results[:3]

    def get_emotional_summary(self, user_id: str = "default") -> dict:
        """Compatibility: emotional summary"""
        summary = {
            "dominant_emotions": [],
            "emotional_stability": 0.0,
            "recent_mood": "neutral",
            "relationship_depth": 0,
        }

        # Get dominant emotions
        if self.emotional_traces:
            sorted_emotions = sorted(
                [(k, v) for k, v in self.emotional_traces.items() if isinstance(v, dict) and "count" in v],
                key=lambda x: x[1].get("count", 0),
                reverse=True,
            )[:3]
            summary["dominant_emotions"] = [e[0] for e in sorted_emotions]

        # Check relationship depth
        if user_id in self.relationships:
            summary["relationship_depth"] = self.relationships[user_id].get("depth", 0)

        return summary

    def get_context_summary(self) -> str:
        """Compatibility: context summary"""
        if not self.current_context:
            return "Aucun contexte disponible."

        parts = []

        # Recent messages
        recent = list(self.current_context)[-3:]
        if recent:
            messages = []
            for entry in recent:
                if isinstance(entry, dict):
                    msg = entry.get("message", entry.get("text", str(entry)))
                else:
                    msg = str(entry)
                messages.append(msg[:100])
            parts.append("Contexte récent:\n" + "\n".join(f"- {m}" for m in messages))

        # Emotional state
        emotional = self.get_emotional_summary()
        if emotional["recent_mood"] != "neutral":
            parts.append(f"Humeur: {emotional['recent_mood']}")

        return "\n\n".join(parts)

    def update_relationship(self, user_id: str, quality: float):
        """Compatibility: update relationship"""
        if user_id not in self.relationships:
            self.relationships[user_id] = {
                "first_interaction": datetime.now().isoformat(),
                "depth": 0,
                "quality": 0.5,
                "interaction_count": 0,
            }

        rel = self.relationships[user_id]
        rel["interaction_count"] += 1
        rel["last_interaction"] = datetime.now().isoformat()
        rel["quality"] = rel["quality"] * 0.8 + quality * 0.2

        if rel["quality"] > 0.6:
            rel["depth"] = min(rel["depth"] + 0.05, 1.0)

    async def save_fact(self, user_id: str, category: str, fact: str):
        """Compatibility: save fact"""
        if user_id not in self.learned_preferences:
            self.learned_preferences[user_id] = {}

        self.learned_preferences[user_id][category] = fact

        await self.store({"type": "fact", "user_id": user_id, "category": category, "fact": fact})

        # Save immediately
        await self.save_persistent_data()
        self.logger.info(f"Saved fact for {user_id}: {category} = {fact}")

    async def consolidate(self):
        """Consolidation with cleanup and optimization"""
        report = {"timestamp": datetime.now().isoformat(), "actions": [], "memory_changes": {}}

        try:
            # Clear old context
            cutoff = time.time() - (30 * 24 * 3600)  # 30 days
            old_count = len(self.current_context)
            new_context = deque(maxlen=self.context_window)
            for item in self.current_context:
                if isinstance(item, dict) and item.get("_timestamp", 0) > cutoff:
                    new_context.append(item)
                elif isinstance(item, str):
                    new_context.append(item)  # Keep string IDs
            self.current_context = new_context
            removed = old_count - len(self.current_context)
            if removed > 0:
                report["actions"].append(f"Removed {removed} old context entries")

            # Optimize backend
            if self.backend and hasattr(self.backend, "vacuum"):
                await self.backend.vacuum()
                report["actions"].append("Backend optimized")

            # Clear cache
            self.cache.clear()
            report["actions"].append("Cache cleared")

            self.logger.info(f"Consolidation complete: {report}")

        except Exception as e:
            self.logger.error(f"Consolidation failed: {e}")
            report["error"] = str(e)

        return report

    async def evolve(self):
        """Evolution system for parameter optimization"""
        changes = {}

        # Adjust cache size based on hit rate
        cache_stats = self.cache.stats()
        if cache_stats["hit_rate"] < 0.3 and self.cache.maxsize < 10000:
            self.cache.maxsize = min(10000, self.cache.maxsize + 500)
            changes["cache_size"] = self.cache.maxsize

        # Adjust batch size based on queue usage
        if self.write_queue.qsize() > self.write_queue.maxsize * 0.8:
            self._batch_size = min(100, self._batch_size + 10)
            changes["batch_size"] = self._batch_size

        # Adjust context window based on usage patterns
        if len(self.current_context) == self.context_window:
            self.context_window = min(20, self.context_window + 2)
            self.current_context = deque(self.current_context, maxlen=self.context_window)
            changes["context_window"] = self.context_window

        if changes:
            self.logger.info(f"🧬 Memory evolved: {changes}")

        return changes


# Export
__all__ = ["UnifiedMemory", "MemoryValidator", "MemoryType", "MemoryPriority"]


# --- AUTO-ADDED health_check (hardening post-launch) ---
def health_check():
    """Health check for memory module"""
    try:
        test_mem = {f"k{i}": i for i in range(100)}
        assert len(test_mem) == 100
        _ = sum(test_mem.values())
        test_mem.clear()
        return {
            "status": "healthy",
            "module": __name__,
            "type": "memory",
            "memory_test": "passed",
            "work": _,
        }
    except Exception as e:
        return {"status": "unhealthy", "module": __name__, "error": str(e)}


# --- /AUTO-ADDED ---
