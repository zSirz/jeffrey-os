import pytest
import asyncio
from typing import AsyncGenerator
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from jeffrey.interfaces.bridge.api import app
from jeffrey.core.config import settings
from jeffrey.models.base import Base
import pytest_asyncio

# Override test database
TEST_DATABASE_URL = settings.DATABASE_URL.replace("jeffrey_brain", "jeffrey_brain_test")

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest_asyncio.fixture(scope="session")
async def test_engine():
    """Create test database engine"""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()

@pytest_asyncio.fixture(scope="function")
async def test_db(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Provide test database session with transaction rollback"""
    SessionLocal = async_sessionmaker(test_engine, expire_on_commit=False)
    async with SessionLocal() as session:
        async with session.begin():
            yield session
            await session.rollback()

@pytest_asyncio.fixture
async def test_client() -> AsyncGenerator[AsyncClient, None]:
    """Provide test client for API testing - using real API"""
    async with AsyncClient(base_url="http://localhost:8000", timeout=30.0) as client:
        yield client