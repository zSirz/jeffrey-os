from fastapi import APIRouter, HTTPException, Query, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timedelta
from typing import List, Optional
from jeffrey.memory.hybrid_store import HybridMemoryStore
from jeffrey.schemas.memory import MemoryCreate, MemoryResponse
from jeffrey.db.session import get_db
from jeffrey.core.config import settings
from jeffrey.core.auth import require_write_permission, require_admin_permission

router = APIRouter(prefix="/api/v1/memories", tags=["memories"])

# Utiliser HybridStore si configuré
if settings.MEMORY_BACKEND == "hybrid":
    memory_store = HybridMemoryStore()
else:
    memory_store = HybridMemoryStore()  # Default to hybrid for now

@router.post("/", response_model=MemoryResponse, dependencies=[Depends(require_write_permission)])
async def create_memory(
    payload: MemoryCreate,
    db: AsyncSession = Depends(get_db)
):
    """Crée une nouvelle mémoire"""
    try:
        memory_dict = payload.dict()
        memory_id = await memory_store.store(memory_dict)

        return MemoryResponse(
            id=memory_id,
            timestamp=datetime.utcnow(),
            processed=False,
            **payload.dict()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/recent", response_model=List[MemoryResponse])
async def get_recent_memories(
    hours: int = Query(24, ge=1, le=168, description="Heures à regarder en arrière"),
    limit: int = Query(50, ge=1, le=500, description="Nombre max de résultats"),
    db: AsyncSession = Depends(get_db)
):
    """Récupère les mémoires récentes"""
    since = datetime.utcnow() - timedelta(hours=hours)
    memories = await memory_store.get_recent(since, limit)

    # Conversion vers schema
    return [
        MemoryResponse(
            **{**mem, "metadata": mem.get("meta", mem.get("metadata", {}))}
        ) for mem in memories
    ]

@router.get("/search", response_model=List[MemoryResponse])
async def search_memories(
    query: str = Query(..., min_length=1, max_length=100),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db)
):
    """Recherche dans les mémoires"""
    results = await memory_store.search(query, limit, offset)

    return [
        MemoryResponse(
            **{**mem, "metadata": mem.get("meta", mem.get("metadata", {}))}
        ) for mem in results
    ]

@router.post("/sync-buffer", dependencies=[Depends(require_admin_permission)])
async def sync_fallback_buffer(db: AsyncSession = Depends(get_db)):
    """Force la synchronisation du buffer fallback"""
    synced = await memory_store.sync_fallback_buffer()

    return {
        "status": "completed",
        "synced_count": synced,
        "remaining_buffer": len(memory_store.fallback_buffer)
    }

@router.get("/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    memory_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Récupère une mémoire spécifique"""
    memory = await memory_store.retrieve(memory_id)

    if not memory:
        raise HTTPException(status_code=404, detail="Memory not found")

    return MemoryResponse(
        **{**memory, "metadata": memory.get("meta", memory.get("metadata", {}))}
    )