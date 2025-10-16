import asyncio
import logging
from typing import AsyncGenerator, Dict, Any
from collections import defaultdict
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from jeffrey.core.config import settings

logger = logging.getLogger(__name__)

# Engine avec configuration robuste
engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False
)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency injection pour FastAPI"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            logger.error(f"DB session error: {e}")
            raise
        finally:
            await session.close()

class AdaptiveRetryManager:
    """Gestionnaire de retry adaptatif basé sur les patterns d'erreur"""

    def __init__(self):
        self.error_counts = defaultdict(int)
        self.base_attempts = 3
        self.base_backoff = 2

    def get_retry_params(self, error_type: str) -> Dict[str, Any]:
        """Calcule les paramètres de retry basés sur l'historique"""
        error_count = self.error_counts[error_type]

        if 'network' in error_type.lower() or 'connection' in error_type.lower():
            # Plus de retries pour les erreurs réseau
            return {
                'max_attempts': min(self.base_attempts + 2, 10),
                'backoff_factor': self.base_backoff + 0.5
            }
        elif error_count > 5:
            # Réduire les retries si trop d'échecs répétés
            return {
                'max_attempts': max(1, self.base_attempts - 1),
                'backoff_factor': self.base_backoff * 2
            }
        else:
            return {
                'max_attempts': self.base_attempts,
                'backoff_factor': self.base_backoff
            }

    def record_error(self, error_type: str):
        """Enregistre une erreur pour l'apprentissage"""
        self.error_counts[error_type] += 1

retry_manager = AdaptiveRetryManager()

def with_adaptive_retry(func):
    """Décorateur de retry adaptatif"""
    async def wrapper(*args, **kwargs):
        last_error = None

        for attempt in range(10):  # Max absolu de sécurité
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_type = type(e).__name__
                retry_manager.record_error(error_type)

                if settings.ENABLE_ADAPTIVE_RETRY:
                    params = retry_manager.get_retry_params(error_type)
                else:
                    params = {'max_attempts': 3, 'backoff_factor': 2}

                if attempt >= params['max_attempts'] - 1:
                    raise

                wait = params['backoff_factor'] ** attempt
                logger.warning(f"Retry {attempt + 1}/{params['max_attempts']} after {wait}s: {e}")
                await asyncio.sleep(wait)
                last_error = e

        raise last_error
    return wrapper