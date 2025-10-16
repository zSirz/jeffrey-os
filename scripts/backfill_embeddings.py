"""
Backfill embeddings for existing memories

This script generates embeddings for all memories that don't have them yet.
It processes memories in batches for efficiency.
"""

import asyncio
import sys
import os
import logging
from typing import List

# Add src to path for imports
sys.path.insert(0, '/app/src')

from jeffrey.db.session import AsyncSessionLocal
from jeffrey.models.memory import Memory
from jeffrey.ml.embeddings_service import embeddings_service
from sqlalchemy import select, update
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def backfill_embeddings():
    """Generate embeddings for existing memories without them"""

    if not embeddings_service.enabled:
        logger.warning("Embeddings service is disabled. Set ENABLE_EMBEDDINGS=true to enable.")
        return

    async with AsyncSessionLocal() as session:
        # Get memories without embeddings
        query = select(Memory).where(Memory.embedding == None).where(Memory.text != None)
        result = await session.execute(query)
        memories = result.scalars().all()

        logger.info(f"Found {len(memories)} memories without embeddings")

        if not memories:
            logger.info("No memories need embedding generation")
            return

        # Process in batches for efficiency
        batch_size = 10
        total_processed = 0
        successful = 0
        failed = 0

        for i in range(0, len(memories), batch_size):
            batch = memories[i:i+batch_size]
            texts = [m.text for m in batch if m.text and m.text.strip()]

            if not texts:
                logger.warning(f"Batch {i//batch_size + 1}: No valid texts to process")
                continue

            logger.info(f"Processing batch {i//batch_size + 1}/{(len(memories) + batch_size - 1)//batch_size}: {len(texts)} texts")

            try:
                # Generate embeddings for batch
                embeddings = await embeddings_service.generate_embeddings_batch(texts)

                # Update memories with embeddings
                for memory, embedding in zip(batch, embeddings):
                    if embedding is not None and memory.text and memory.text.strip():
                        try:
                            # Update the memory with embedding
                            await session.execute(
                                update(Memory)
                                .where(Memory.id == memory.id)
                                .values(embedding=embedding.tolist())
                            )
                            successful += 1
                            logger.debug(f"Updated memory {memory.id} with embedding")
                        except Exception as e:
                            logger.error(f"Failed to update memory {memory.id}: {e}")
                            failed += 1
                    else:
                        logger.warning(f"No embedding generated for memory {memory.id}")
                        failed += 1

                # Commit batch
                await session.commit()
                total_processed += len(batch)
                logger.info(f"Batch completed. Total processed: {total_processed}/{len(memories)}")

            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                await session.rollback()
                failed += len(batch)

        logger.info(f"Backfill complete! Processed: {total_processed}, Successful: {successful}, Failed: {failed}")

        # Verify results
        verification_query = select(Memory).where(Memory.embedding != None)
        verification_result = await session.execute(verification_query)
        memories_with_embeddings = len(verification_result.scalars().all())

        logger.info(f"Total memories with embeddings: {memories_with_embeddings}")

async def main():
    """Main entry point"""
    try:
        logger.info("Starting embeddings backfill process...")
        await backfill_embeddings()
        logger.info("Embeddings backfill completed successfully")
    except Exception as e:
        logger.error(f"Embeddings backfill failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())