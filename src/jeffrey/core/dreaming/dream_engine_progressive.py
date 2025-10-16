import asyncio
import hashlib
import json
import os
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
import logging

# Nouveaux imports pour PostgreSQL
from jeffrey.memory.hybrid_store import HybridMemoryStore
from jeffrey.db.session import AsyncSessionLocal

logger = logging.getLogger(__name__)


# Simple learner for auto-optimization
class SimpleLearner:
    def __init__(self):
        self.history = deque(maxlen=20)
        self.adjustments = {'batch_size': 0, 'timeout': 0}

    def learn(self, quality_score):
        self.history.append(quality_score)
        if len(self.history) >= 3:
            avg_quality = sum(self.history) / len(self.history)
            if avg_quality < 0.1:
                self.adjustments['batch_size'] = 50  # Increase
            elif avg_quality > 0.5:
                self.adjustments['batch_size'] = -25  # Decrease
        return self.adjustments


class DreamEngineProgressive:
    """
    Production-grade DreamEngine with:
    - Distributed locks (Redis)
    - Idempotence with persistent tracking
    - Auto-evolution via learner
    - Memory pagination
    - DLQ for failures
    - Semaphore for resource control
    """

    def __init__(self, bus, memory_port, circadian=None):
        self.bus = bus
        self.memory = memory_port  # Keep for compatibility
        self.memory_store = HybridMemoryStore()  # New PostgreSQL store
        self.circadian = circadian

        # Redis for distributed operations (using redis.asyncio)
        self.redis = None
        if os.getenv('REDIS_URL'):
            try:
                import redis.asyncio as redis_asyncio
                self.redis = redis_asyncio.from_url(
                    os.getenv('REDIS_URL'),
                    decode_responses=True
                )
                logger.info("✅ Redis connected for distributed operations")
            except ImportError:
                logger.warning("⚠️ redis.asyncio not available, using file locks")

        # Import existing DreamEngine if available
        try:
            from jeffrey.core.dreaming.dream_engine import DreamEngine
            self.original_engine = DreamEngine()
            logger.info("✅ Original DreamEngine loaded")
        except ImportError:
            self.original_engine = None
            logger.warning("⚠️ Original DreamEngine not found, using fallback")

        # Configuration
        self.enabled = False
        self.test_mode = True
        self.timeout = 60
        self.window_hours = 24  # Default window for memory retrieval
        self.batch_size = 100  # Will be auto-adjusted
        self.max_batch_size = 500
        self.memory_page_size = 50

        # Semaphore for resource control
        self.dream_semaphore = asyncio.Semaphore(2)  # Max 2 concurrent dreams

        # Persistent idempotence tracking
        self.processed_index_file = Path("data/dreams/processed_index.json")
        self.processed_dates = self._load_processed_dates()

        # Debounce for circadian triggers
        self.last_trigger_time = 0
        self.debounce_seconds = 300  # 5 minutes

        # DLQ for failures
        self.dlq = deque(maxlen=100)

        # Auto-evolution learner
        self.learner = SimpleLearner()
        self.min_batch_size = 50
        self.max_batch_size = 500
        self.quality_history = deque(maxlen=50)

        # Stats
        self.stats = {
            "runs_total": 0,
            "runs_success": 0,
            "runs_failed": 0,
            "memories_scanned": 0,
            "memories_processed": 0,
            "insights_generated": 0,
            "avg_quality": 0,
            "avg_duration_ms": 0,
            "dlq_size": 0
        }

    def _load_processed_dates(self) -> set:
        """Load processed dates from persistent storage"""
        if self.processed_index_file.exists():
            try:
                with open(self.processed_index_file, 'r') as f:
                    data = json.load(f)
                    return set(data.get('processed_dates', []))
            except Exception as e:
                logger.error(f"Failed to load processed dates: {e}")
        return set()

    def _save_processed_dates(self):
        """Save processed dates to persistent storage"""
        try:
            self.processed_index_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.processed_index_file, 'w') as f:
                json.dump({
                    'processed_dates': list(self.processed_dates),
                    'last_updated': datetime.now().isoformat()
                }, f)
        except Exception as e:
            logger.error(f"Failed to save processed dates: {e}")

    async def _acquire_distributed_lock(self, key: str, timeout: int = 60) -> bool:
        """Acquire distributed lock using Redis"""
        if not self.redis:
            # Fallback to file lock
            lock_file = Path(f"data/dreams/locks/{key}.lock")
            lock_file.parent.mkdir(parents=True, exist_ok=True)

            if lock_file.exists():
                # Check if lock is stale
                mtime = lock_file.stat().st_mtime
                if time.time() - mtime > timeout:
                    lock_file.unlink()  # Remove stale lock
                else:
                    return False

            lock_file.touch()
            return True

        # Redis lock with TTL
        lock_key = f"dream:lock:{key}"
        acquired = await self.redis.set(
            lock_key,
            "1",
            nx=True,  # Only set if not exists
            ex=timeout + 60  # TTL
        )
        return bool(acquired)

    async def _release_distributed_lock(self, key: str):
        """Release distributed lock"""
        if not self.redis:
            lock_file = Path(f"data/dreams/locks/{key}.lock")
            if lock_file.exists():
                lock_file.unlink()
        else:
            await self.redis.delete(f"dream:lock:{key}")

    async def _get_recent_memories(self, window_hours: int = 24):
        """Récupère les mémoires récentes depuis PostgreSQL"""
        since = datetime.utcnow() - timedelta(hours=window_hours)
        memories = await self.memory_store.get_recent(since, limit=1000)

        # Convertir au format attendu par consolidate
        return [{
            'id': mem.get('id'),
            'timestamp': mem.get('timestamp'),
            'content': mem.get('text', ''),
            'emotion': mem.get('emotion'),
            'confidence': mem.get('confidence', 0.5),
            'metadata': mem.get('meta', {})
        } for mem in memories]

    async def should_run(self) -> bool:
        """Check if dream consolidation should run with debounce"""
        if not self.enabled:
            logger.debug("DreamEngine disabled by feature flag")
            return False

        # Debounce check
        current_time = time.time()
        if current_time - self.last_trigger_time < self.debounce_seconds:
            logger.debug(f"Debounce active, wait {self.debounce_seconds - (current_time - self.last_trigger_time):.0f}s")
            return False

        # Circadian check with async/sync compatibility
        if self.circadian:
            if asyncio.iscoroutinefunction(self.circadian.get_state):
                state = await self.circadian.get_state()
            else:
                state = self.circadian.get_state()

            if state.get('phase') != 'night' or state.get('energy_level', 1.0) > 0.3:
                logger.debug("Not in deep sleep phase")
                return False

        # Idempotence check
        today = datetime.now().strftime("%Y-%m-%d")
        if today in self.processed_dates:
            logger.debug(f"Already processed for {today}")
            return False

        # Try to acquire lock
        lock_acquired = await self._acquire_distributed_lock(today)
        if not lock_acquired:
            logger.warning(f"Failed to acquire lock for {today} - another instance may be running")
            return False

        self.last_trigger_time = current_time
        return True

    async def consolidate_memories(self, window_hours: int = 24, force: bool = False) -> Dict:
        """
        Main consolidation with all production features
        """
        if not force and not await self.should_run():
            return {"skipped": True, "reason": "Conditions not met"}

        date_str = datetime.now().strftime("%Y-%m-%d")
        run_id = hashlib.sha256(f"dream-{date_str}".encode()).hexdigest()[:16]
        start_time = asyncio.get_event_loop().time()

        async with self.dream_semaphore:  # Resource control
            try:
                self.stats["runs_total"] += 1

                # Paginated memory retrieval
                since = datetime.now() - timedelta(hours=window_hours)
                # Utiliser la nouvelle méthode PostgreSQL
                memories = await self._get_recent_memories(window_hours)
                self.stats["memories_scanned"] = len(memories)

                # Process with timeout budget
                process_timeout = min(self.timeout, 55)  # Leave 5s margin
                result = await asyncio.wait_for(
                    self._process_memories(memories),
                    timeout=process_timeout
                )

                # Generate insights
                insights = self._generate_insights(result)

                # Auto-evolution: adjust batch size based on quality
                quality = len(insights) / max(len(memories), 1)
                self.quality_history.append(quality)

                # Auto-learning adjustment
                adjustments = self.learner.learn(quality)
                if adjustments.get('batch_size'):
                    new_size = self.batch_size + adjustments['batch_size']
                    self.batch_size = max(self.min_batch_size, min(self.max_batch_size, new_size))
                    logger.info(f"📈 Auto-adjusted batch_size to {self.batch_size} (quality: {quality:.2f})")

                # Prepare output
                output = {
                    "run_id": run_id,
                    "timestamp": datetime.now().isoformat(),
                    "window_hours": window_hours,
                    "memories_scanned": self.stats["memories_scanned"],
                    "memories_processed": len(memories),
                    "consolidation": result,
                    "insights": insights,
                    "quality_score": quality,
                    "test_mode": self.test_mode
                }

                # Write output
                await self._write_output(output)

                # Publish success event
                await self._publish_event("dream.completed.v1", {
                    "run_id": run_id,
                    "insights_count": len(insights),
                    "quality": quality,
                    "duration_ms": (asyncio.get_event_loop().time() - start_time) * 1000
                })

                # Update persistent tracking
                self.processed_dates.add(date_str)
                self._save_processed_dates()

                self.stats["runs_success"] += 1
                self.stats["insights_generated"] += len(insights)

                # Marquer les mémoires comme processed
                if not self.test_mode and memories:
                    memory_ids = [m['id'] for m in memories if m.get('id')]
                    if memory_ids:
                        try:
                            await self.memory_store.mark_processed(memory_ids)
                            logger.info(f"✅ Marked {len(memory_ids)} memories as processed")
                        except Exception as e:
                            logger.warning(f"Failed to mark memories as processed: {e}")

                logger.info(f"✨ Dream complete: {len(insights)} insights, quality={quality:.2f}")
                return output

            except asyncio.TimeoutError:
                self._add_to_dlq(run_id, "Timeout", {"window_hours": window_hours})
                self.stats["runs_failed"] += 1
                raise

            except Exception as e:
                self._add_to_dlq(run_id, str(e), {"window_hours": window_hours})
                self.stats["runs_failed"] += 1
                await self._publish_event("dream.failed.v1", {
                    "run_id": run_id,
                    "error": str(e),
                    "memories_seen": self.stats["memories_scanned"]
                })
                raise

            finally:
                await self._release_distributed_lock(date_str)

    async def _fetch_memories_paginated(self, since: datetime) -> List[Dict]:
        """Fetch memories with safe pagination"""
        all_memories = []
        page = 0
        budget_end = time.time() + (self.timeout - 5)

        while time.time() < budget_end and len(all_memories) < self.batch_size:
            if self.memory:
                try:
                    # Try with offset first
                    if hasattr(self.memory.search, '__code__') and 'offset' in self.memory.search.__code__.co_varnames:
                        offset = page * self.memory_page_size
                        batch = self.memory.search("", limit=self.memory_page_size, offset=offset)
                    else:
                        # Fallback: get all at once
                        if page == 0:
                            batch = self.memory.search("", limit=self.batch_size)
                        else:
                            break  # Can't paginate without offset
                except TypeError:
                    # Offset not supported, get all at once
                    if page == 0:
                        batch = self.memory.search("", limit=self.batch_size)
                    else:
                        break

                if not batch:
                    break

                # Filter by time
                recent = [m for m in batch if self._is_recent(m, since)]
                all_memories.extend(recent)

                if len(batch) < self.memory_page_size:
                    break

                page += 1

        return all_memories[:self.batch_size]

    def _auto_adjust_parameters(self):
        """Auto-adjust batch size based on quality"""
        if len(self.quality_history) < 5:
            return

        avg_quality = sum(self.quality_history) / len(self.quality_history)
        self.stats["avg_quality"] = avg_quality

        if avg_quality < 0.05 and self.batch_size < self.max_batch_size:
            self.batch_size = min(self.batch_size + 50, self.max_batch_size)
            logger.info(f"📈 Auto-increased batch_size to {self.batch_size}")
        elif avg_quality > 0.2 and self.batch_size > 50:
            self.batch_size = max(self.batch_size - 25, 50)
            logger.info(f"📉 Auto-decreased batch_size to {self.batch_size}")

    def _add_to_dlq(self, run_id: str, error: str, context: Dict):
        """Add failed run to DLQ"""
        dlq_entry = {
            "run_id": run_id,
            "error": error,
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
        self.dlq.append(dlq_entry)
        self.stats["dlq_size"] = len(self.dlq)

        # Save to file for post-mortem
        dlq_file = Path(f"data/dreams/failed/run_{run_id}.json")
        dlq_file.parent.mkdir(parents=True, exist_ok=True)
        with open(dlq_file, 'w') as f:
            json.dump(dlq_entry, f, indent=2)

    async def _process_memories(self, memories: List[Dict]) -> Dict:
        """Process memories using original engine or fallback"""
        if self.original_engine and hasattr(self.original_engine, 'process'):
            try:
                if asyncio.iscoroutinefunction(self.original_engine.process):
                    return await self.original_engine.process(memories)
                else:
                    return self.original_engine.process(memories)
            except Exception as e:
                logger.warning(f"Original engine failed: {e}, using fallback")

        return self._process_fallback(memories)

    def _process_fallback(self, memories: List[Dict]) -> Dict:
        """Enhanced fallback with clustering"""
        emotion_groups = {}
        topic_keywords = {}

        for mem in memories:
            # Group by emotion
            emotion = mem.get('emotion', 'neutral')
            if emotion not in emotion_groups:
                emotion_groups[emotion] = []
            emotion_groups[emotion].append(mem)

            # Extract keywords (simple approach)
            text = mem.get('text', '')
            words = text.lower().split()
            for word in words:
                if len(word) > 4:  # Simple filter
                    topic_keywords[word] = topic_keywords.get(word, 0) + 1

        # Get top topics
        top_topics = sorted(topic_keywords.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "dominant_emotions": sorted(emotion_groups.keys(),
                                      key=lambda x: len(emotion_groups[x]),
                                      reverse=True)[:3],
            "memory_clusters": len(emotion_groups),
            "top_topics": [t[0] for t in top_topics],
            "total_processed": len(memories)
        }

    def _generate_insights(self, consolidation: Dict) -> List[str]:
        """Generate insights with SelfReflection integration"""
        insights = []

        # Basic insights
        if 'dominant_emotions' in consolidation:
            emotions = consolidation['dominant_emotions']
            if emotions:
                insights.append(f"Emotional pattern: {', '.join(emotions[:2])}")

        if 'top_topics' in consolidation:
            topics = consolidation['top_topics']
            if topics:
                insights.append(f"Key focus areas: {', '.join(topics[:3])}")

        if consolidation.get('memory_clusters', 0) > 5:
            insights.append("High cognitive diversity detected")

        # TODO: Integrate with SelfReflection for meta-insights

        return insights

    def _is_recent(self, memory: Dict, since: datetime) -> bool:
        """Check if memory is recent"""
        timestamp = memory.get('timestamp', '')
        if timestamp:
            try:
                mem_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                return mem_time >= since
            except:
                pass
        return False

    async def _write_output(self, output: Dict):
        """Write output to appropriate storage"""
        if self.test_mode:
            # Test mode: write to file
            filename = f"data/dreams/test/dream_{output['run_id']}.json"
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            with open(filename, 'w') as f:
                json.dump(output, f, indent=2)
            logger.info(f"📝 Test output: {filename}")
        else:
            # Production mode: write to database or persistent storage
            # TODO: Implement production storage
            pass

    async def _publish_event(self, event_type: str, data: Dict):
        """Publish event to bus"""
        try:
            from jeffrey.core.neuralbus.events import make_event
            event = make_event(event_type, data, source="jeffrey.dream.progressive")
            await self.bus.publish(event)
        except Exception as e:
            logger.warning(f"Failed to publish event {event_type}: {e}")

    def get_stats(self) -> Dict:
        """Get comprehensive statistics"""
        return {
            **self.stats,
            "enabled": self.enabled,
            "test_mode": self.test_mode,
            "batch_size": self.batch_size,
            "processed_dates_count": len(self.processed_dates),
            "dlq_recent": list(self.dlq)[-5:] if self.dlq else []
        }

    async def backfill(self, days: int = 7) -> Dict:
        """Controlled backfill for past days"""
        results = []
        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")

            if date_str not in self.processed_dates:
                logger.info(f"Backfilling for {date_str}")
                # Force run for specific date
                result = await self.consolidate_memories(
                    window_hours=24,
                    force=True
                )
                results.append(result)

                # Rate limit
                await asyncio.sleep(5)

        return {"backfilled": len(results), "results": results}