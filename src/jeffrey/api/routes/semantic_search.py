from fastapi import APIRouter, Query, Depends, HTTPException
from typing import List, Optional
from pydantic import BaseModel, Field
from jeffrey.memory.hybrid_store import HybridMemoryStore
from jeffrey.ml.embeddings_service import embeddings_service
from jeffrey.core.auth import verify_api_key
import numpy as np
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/search", tags=["search"])

class SemanticSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Search query text")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum number of results")
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum similarity score")

    class Config:
        example = {
            "query": "happiness and joy",
            "limit": 5,
            "threshold": 0.3
        }

class SearchResult(BaseModel):
    id: str
    text: str
    emotion: Optional[str]
    confidence: Optional[float]
    similarity: float = Field(..., ge=0.0, le=1.0, description="Semantic similarity score")
    timestamp: str
    meta: dict = {}

@router.post("/semantic", response_model=List[SearchResult])
async def semantic_search(
    request: SemanticSearchRequest,
    _: bool = Depends(verify_api_key)
) -> List[SearchResult]:
    """
    Search memories using semantic similarity

    Uses cosine similarity between query and memory embeddings via pgvector.
    Requires embeddings to be enabled (ENABLE_EMBEDDINGS=true).
    """
    try:
        # Check if embeddings are enabled
        if not embeddings_service.enabled:
            raise HTTPException(
                status_code=503,
                detail="Semantic search unavailable: embeddings service disabled"
            )

        # Generate embedding for query
        query_embedding = await embeddings_service.generate_embedding(request.query)

        if query_embedding is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to generate embedding for query"
            )

        # Search in database using pgvector
        store = HybridMemoryStore()
        results = await store.semantic_search(
            query_embedding=query_embedding,
            limit=request.limit,
            threshold=request.threshold
        )

        # Convert to response format
        search_results = [
            SearchResult(
                id=r['id'],
                text=r['text'],
                emotion=r.get('emotion'),
                confidence=r.get('confidence'),
                similarity=r['similarity'],
                timestamp=r['timestamp'].isoformat() if hasattr(r['timestamp'], 'isoformat') else str(r['timestamp']),
                meta=r.get('meta', {})
            )
            for r in results
        ]

        logger.info(f"Semantic search for '{request.query}' returned {len(search_results)} results")

        return search_results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Semantic search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")

@router.get("/semantic/status")
async def semantic_search_status():
    """
    Get semantic search service status

    Returns information about embeddings service availability and configuration.
    """
    try:
        return {
            "service": "semantic_search",
            "embeddings_enabled": embeddings_service.enabled,
            "model": embeddings_service.model_name,
            "dimensions": 384,
            "status": "available" if embeddings_service.enabled else "disabled",
            "features": [
                "pgvector_cosine_similarity",
                "ivfflat_index",
                "batch_processing"
            ]
        }
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to get status")