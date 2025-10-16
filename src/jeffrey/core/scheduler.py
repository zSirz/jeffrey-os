import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime, timedelta
import os
import asyncio
from sqlalchemy import text

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

        # Add consciousness cycle if enabled (Grok optimization #1)
        if os.getenv("ENABLE_CONSCIOUSNESS", "false").lower() == "true":
            self.scheduler.add_job(
                self._run_cycle_with_timeout,
                trigger=IntervalTrigger(
                    minutes=int(os.getenv("CONSCIOUSNESS_CYCLE_MINUTES", "30"))
                ),
                id="consciousness_cycle",
                name="Consciousness Cycle",
                max_instances=1,  # ✅ Pas de chevauchement
                coalesce=True,    # ✅ Si retard, exécute une seule fois
                replace_existing=True
            )
            logger.info(f"Consciousness cycle scheduled every {os.getenv('CONSCIOUSNESS_CYCLE_MINUTES', '30')} minutes")

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

    async def _run_cycle_with_timeout(self):
        """Run cycle avec timeout soft (Grok optimization #1)"""
        timeout = int(os.getenv("CONSCIOUSNESS_CYCLE_TIMEOUT", "120"))

        try:
            await asyncio.wait_for(
                self.run_consciousness_cycle(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            logger.error(f"Consciousness cycle timeout after {timeout}s")
            # Métrique timeout
            from jeffrey.core.metrics import consciousness_cycle_timeouts_total
            consciousness_cycle_timeouts_total.inc()
        except Exception as e:
            logger.error(f"Consciousness cycle failed: {e}")

    async def run_consciousness_cycle(self):
        """Cycle complet de conscience sécurisé avec flags"""
        if not os.getenv("ENABLE_CONSCIOUSNESS", "false").lower() == "true":
            logger.info("Consciousness cycle skipped - disabled by flag")
            return

        try:
            logger.info("🧠 Starting consciousness cycle...")
            start_time = datetime.utcnow()

            # Import des métriques
            from jeffrey.core.metrics import (
                consciousness_cycles_total,
                consciousness_cycle_errors,
                consciousness_cycle_duration,
                curiosity_questions_generated
            )

            # 1. Analyse ProactiveCuriosity
            from jeffrey.core.consciousness.proactive_curiosity_safe import ProactiveCuriositySafe
            curiosity = ProactiveCuriositySafe()
            analysis = await curiosity.analyze_gaps()
            questions = await curiosity.generate_questions()

            logger.info(f"Generated {len(questions)} curiosity questions")
            curiosity_questions_generated.inc(len(questions))

            # 2. Stocker questions si write enabled
            if os.getenv("ENABLE_CONSCIOUSNESS_WRITE", "false").lower() == "true" and questions:
                from jeffrey.memory.hybrid_store import HybridMemoryStore
                memory_store = HybridMemoryStore()

                max_new = int(os.getenv("CONSCIOUSNESS_MAX_NEW_MEMORIES", "3"))
                for question in questions[:max_new]:
                    await memory_store.store({
                        'text': question,
                        'emotion': 'curiosity',
                        'confidence': 0.7,
                        'meta': {
                            'type': 'proactive_question',
                            'source': 'consciousness_cycle'
                        }
                    })
                logger.info(f"Stored {min(len(questions), max_new)} questions as memories")

            # 3. Update Emotional Bonds pour mémoires récentes
            from jeffrey.core.consciousness.bonds_service import bonds_service
            from jeffrey.ml.embeddings_service import embeddings_service
            from jeffrey.memory.hybrid_store import HybridMemoryStore
            from jeffrey.db.session import AsyncSessionLocal
            memory_store = HybridMemoryStore()

            since = datetime.utcnow() - timedelta(hours=1)
            recent = await memory_store.get_recent(since, limit=10)

            # Cap anti-emballement
            max_bonds_updates = int(os.getenv("CONSCIOUSNESS_MAX_BONDS_UPDATES", "20"))
            bonds_created = 0

            for mem in recent:
                if not mem.get('text') or bonds_created >= max_bonds_updates:
                    continue

                # Chercher mémoires similaires
                embedding = await embeddings_service.generate_embedding(mem['text'])
                similar = await memory_store.semantic_search(
                    embedding, limit=3, threshold=0.6
                )

                for sim in similar:
                    if bonds_created >= max_bonds_updates:
                        break

                    if sim['id'] != mem.get('id'):
                        emotion_match = sim.get('emotion') == mem.get('emotion')
                        delta = 0.1 if emotion_match else -0.05

                        bond = await bonds_service.upsert_bond(
                            mem['id'], sim['id'],
                            delta_strength=delta,
                            emotion_match=emotion_match
                        )
                        if bond:
                            bonds_created += 1

            logger.info(f"Updated {bonds_created} emotional bonds")

            # 4. Prune weak bonds
            pruned = await bonds_service.prune_weak_bonds()
            if pruned > 0:
                logger.info(f"Pruned {pruned} weak bonds")

            # Métriques
            duration = (datetime.utcnow() - start_time).total_seconds()
            consciousness_cycle_duration.observe(duration)
            consciousness_cycles_total.inc()

            # Update bonds gauge
            try:
                from jeffrey.core.metrics import bonds_active_gauge
                async with AsyncSessionLocal() as session:
                    count_query = text("SELECT COUNT(*) FROM emotional_bonds WHERE strength > 0.1")
                    active_count = (await session.execute(count_query)).scalar_one()
                    bonds_active_gauge.set(active_count)
                    logger.info(f"Active bonds gauge updated: {active_count}")
            except Exception as e:
                logger.warning(f"Failed to update bonds gauge: {e}")

            logger.info(f"✅ Consciousness cycle complete in {duration:.2f}s")

        except Exception as e:
            logger.error(f"Consciousness cycle failed: {e}")
            from jeffrey.core.metrics import consciousness_cycle_errors
            consciousness_cycle_errors.inc()

    async def stop(self):
        """Stop the scheduler"""
        if self.scheduler.running:
            self.scheduler.shutdown()
            logger.info("Scheduler stopped")

# Global scheduler instance
dream_scheduler = DreamScheduler()