import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class DreamScheduler:
    """Orchestrates automatic dream consolidation runs"""

    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.enabled = os.getenv("ENABLE_SCHEDULER", "false").lower() == "true"
        self.interval_minutes = int(os.getenv("DREAM_INTERVAL_MINUTES", "15"))

    async def run_dream_job(self):
        """Job that runs dream consolidation"""
        # Import inside method to avoid circular imports
        from jeffrey.core.dreaming.dream_engine_progressive import DreamEngineProgressive

        try:
            logger.info(f"Starting scheduled dream run at {datetime.utcnow()}")
            engine = DreamEngineProgressive()
            result = await engine.run(force=False, test_mode=False)

            if result:
                logger.info(f"Dream run completed: {result.get('memories_processed', 0)} memories processed")
            else:
                logger.info("Dream run skipped - no new memories")

        except Exception as e:
            logger.error(f"Scheduled dream run failed: {e}")

    async def start(self):
        """Start the scheduler"""
        if not self.enabled:
            logger.info("Scheduler disabled via ENABLE_SCHEDULER flag")
            return

        # Schedule dream consolidation
        self.scheduler.add_job(
            self.run_dream_job,
            trigger=IntervalTrigger(minutes=self.interval_minutes),
            id="dream_consolidation",
            name="Dream Consolidation",
            replace_existing=True
        )

        # Add sync fallback buffer job
        self.scheduler.add_job(
            self.sync_fallback_job,
            trigger=IntervalTrigger(minutes=5),
            id="sync_fallback",
            name="Sync Fallback Buffer",
            replace_existing=True
        )

        self.scheduler.start()
        logger.info(f"Scheduler started - Dream runs every {self.interval_minutes} minutes")

    async def sync_fallback_job(self):
        """Sync any pending memories from fallback buffer"""
        from jeffrey.memory.hybrid_store import HybridMemoryStore

        try:
            store = HybridMemoryStore()
            synced = await store.sync_fallback_buffer()
            if synced > 0:
                logger.info(f"Synced {synced} memories from fallback buffer")
        except Exception as e:
            logger.error(f"Fallback sync failed: {e}")

    async def stop(self):
        """Stop the scheduler"""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Scheduler stopped")

# Global scheduler instance
dream_scheduler = DreamScheduler()