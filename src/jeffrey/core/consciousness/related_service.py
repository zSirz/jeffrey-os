"""
Related Memories Service - VERSION OPTIMISÉE BATCH
Grok optimization #4: O(1) batch queries au lieu de O(n)
"""
from typing import List, Dict
import logging
from jeffrey.memory.hybrid_store import HybridMemoryStore
from jeffrey.ml.embeddings_service import embeddings_service
from sqlalchemy import text
from jeffrey.db.session import AsyncSessionLocal

logger = logging.getLogger(__name__)

async def get_related_memories_batch(
    query: str,
    limit: int = 10
) -> List[Dict]:
    """
    Get related memories avec BATCH JOIN (O(1) au lieu de O(n))

    Algorithm:
    1. Semantic search (top-k * 2)
    2. Single JOIN query pour tous les bonds
    3. Score enrichi avec avg_bond_strength
    4. Reorder et limit
    """
    try:
        store = HybridMemoryStore()

        # Step 1: Query embedding
        query_embedding = await embeddings_service.generate_embedding(query)
        if query_embedding is None:
            return []

        # Step 2: Semantic search
        semantic_results = await store.semantic_search(
            query_embedding=query_embedding,
            limit=limit * 2,
            threshold=0.3
        )

        if not semantic_results:
            return []

        # Step 3: BATCH JOIN pour bonds (O(1) au lieu de O(n))
        memory_ids = [m['id'] for m in semantic_results]

        async with AsyncSessionLocal() as session:
            # Single query avec JOIN
            bonds_query = text("""
                SELECT
                    m.id AS memory_id,
                    COUNT(eb.id) AS bonds_count,
                    COALESCE(AVG(eb.strength), 0) AS avg_strength
                FROM
                    (SELECT unnest(:memory_ids::uuid[]) AS id) m
                LEFT JOIN emotional_bonds eb
                    ON m.id = eb.memory_id_a OR m.id = eb.memory_id_b
                WHERE eb.strength > 0.1
                GROUP BY m.id
            """)

            result = await session.execute(
                bonds_query,
                {"memory_ids": memory_ids}
            )

            bonds_map = {
                str(row.memory_id): {
                    'count': row.bonds_count,
                    'avg_strength': float(row.avg_strength)
                }
                for row in result.fetchall()
            }

        # Step 4: Enrich et score
        enriched_results = []
        for memory in semantic_results:
            memory_id = memory['id']
            bonds_info = bonds_map.get(memory_id, {'count': 0, 'avg_strength': 0.0})

            semantic_score = memory['similarity']
            bond_boost = 1 + (bonds_info['avg_strength'] * 0.5)
            enriched_score = semantic_score * bond_boost

            enriched_results.append({
                **memory,
                'bonds_count': bonds_info['count'],
                'avg_bond_strength': bonds_info['avg_strength'],
                'enriched_score': enriched_score
            })

        # Step 5: Reorder et limit
        enriched_results.sort(key=lambda x: x['enriched_score'], reverse=True)
        return enriched_results[:limit]

    except Exception as e:
        logger.error(f"get_related_memories_batch failed: {e}")
        return []