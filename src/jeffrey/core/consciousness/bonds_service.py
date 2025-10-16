import logging
import uuid
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from sqlalchemy import select, update, delete, and_, or_
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.sql import func
from jeffrey.db.session import AsyncSessionLocal
from jeffrey.models.emotional_bond import EmotionalBond
from jeffrey.core.metrics import bonds_upserted_total, bonds_pruned_total
import asyncio

logger = logging.getLogger(__name__)

class EmotionalBondsService:
    """Service pour gérer les liens émotionnels entre mémoires"""
    
    async def upsert_bond(
        self, 
        memory_id_a: str, 
        memory_id_b: str, 
        delta_strength: float,
        emotion_match: bool = False
    ) -> Optional[Dict]:
        """Upsert un bond avec incrémentation atomique"""
        try:
            # Ordonner les IDs avec de vrais UUID pour respecter la contrainte
            ua = uuid.UUID(str(memory_id_a))
            ub = uuid.UUID(str(memory_id_b))
            id_a, id_b = (ua, ub) if ua < ub else (ub, ua)
            id_a, id_b = str(id_a), str(id_b)
            
            async with AsyncSessionLocal() as session:
                # UPSERT avec ON CONFLICT UPDATE
                stmt = insert(EmotionalBond).values(
                    memory_id_a=id_a,
                    memory_id_b=id_b,
                    strength=max(0, min(1, delta_strength)),  # Clamp initial
                    emotion_match=emotion_match
                )
                
                # Sur conflit, incrémenter strength
                stmt = stmt.on_conflict_do_update(
                    constraint='unique_bond_pair',  # Plus fiable que index_elements
                    set_={
                        'strength': func.least(1.0, func.greatest(0.0, 
                            EmotionalBond.strength + delta_strength)),
                        'emotion_match': emotion_match,
                        'last_updated': func.now()
                    }
                )
                
                result = await session.execute(stmt)
                await session.commit()
                
                # Récupérer le bond mis à jour
                query = select(EmotionalBond).where(
                    and_(
                        EmotionalBond.memory_id_a == id_a,
                        EmotionalBond.memory_id_b == id_b
                    )
                )
                bond = await session.execute(query)
                bond = bond.scalar_one_or_none()
                
                if bond:
                    bonds_upserted_total.inc()
                    return {
                        'id': str(bond.id),
                        'memory_pair': [str(bond.memory_id_a), str(bond.memory_id_b)],
                        'strength': bond.strength,
                        'emotion_match': bond.emotion_match,
                        'last_updated': bond.last_updated.isoformat()
                    }
                    
        except Exception as e:
            logger.error(f"Failed to upsert bond: {e}")
            return None
    
    async def prune_weak_bonds(self, threshold: float = 0.1, days_old: int = 7) -> int:
        """Supprime les bonds faibles et anciens"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            
            async with AsyncSessionLocal() as session:
                stmt = delete(EmotionalBond).where(
                    and_(
                        EmotionalBond.strength < threshold,
                        EmotionalBond.last_updated < cutoff_date
                    )
                )
                result = await session.execute(stmt)
                await session.commit()
                
                pruned_count = result.rowcount
                if pruned_count > 0:
                    logger.info(f"Pruned {pruned_count} weak bonds")
                    bonds_pruned_total.inc(pruned_count)
                    
                return pruned_count
                
        except Exception as e:
            logger.error(f"Failed to prune bonds: {e}")
            return 0
    
    async def get_bonds_for_memory(self, memory_id: str, limit: int = 10) -> List[Dict]:
        """Récupère les bonds d'une mémoire"""
        try:
            async with AsyncSessionLocal() as session:
                query = select(EmotionalBond).where(
                    or_(
                        EmotionalBond.memory_id_a == memory_id,
                        EmotionalBond.memory_id_b == memory_id
                    )
                ).order_by(EmotionalBond.strength.desc()).limit(limit)
                
                result = await session.execute(query)
                bonds = result.scalars().all()
                
                return [
                    {
                        'id': str(bond.id),
                        'other_memory': str(bond.memory_id_b if str(bond.memory_id_a) == memory_id else bond.memory_id_a),
                        'strength': bond.strength,
                        'emotion_match': bond.emotion_match,
                        'last_updated': bond.last_updated.isoformat()
                    }
                    for bond in bonds
                ]
                
        except Exception as e:
            logger.error(f"Failed to get bonds: {e}")
            return []

# Global instance
bonds_service = EmotionalBondsService()