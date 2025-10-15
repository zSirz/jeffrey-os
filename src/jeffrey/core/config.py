"""
NeuralBus Configuration with auto-detection
Auto-detects available dependencies and configures accordingly
"""

import logging
import os

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from jeffrey.utils.logger import get_logger

logger = get_logger("neuralbus.config")

# Auto-detect optional dependencies
HAS_MSGPACK = False
HAS_UVLOOP = False
HAS_REDIS = False

try:
    import msgpack

    HAS_MSGPACK = True
    logger.info("msgpack detected - binary serialization enabled")
except ImportError:
    logger.info("msgpack not found - using JSON serialization")

try:
    import uvloop

    HAS_UVLOOP = True
    # Don't install automatically - let main() do it
except ImportError:
    HAS_UVLOOP = False


def install_async_optimizations():
    """Install async optimizations (call from main only)"""
    if HAS_UVLOOP:
        try:
            import uvloop

            uvloop.install()
            logger.info("uvloop installed - async performance optimized")
            return True
        except Exception as e:
            logger.warning(f"uvloop not available: {e}")
            return False
    else:
        logger.info("uvloop not found - using standard asyncio")
        return False


try:
    import redis

    HAS_REDIS = True
    logger.info("redis detected - distributed deduplication available")
except ImportError:
    logger.info("redis not found - using local LRU deduplication")


class NeuralBusConfig(BaseSettings):
    """Production configuration with smart defaults"""

    # NATS Configuration
    NATS_URL: str = Field(default="nats://127.0.0.1:4223")
    NATS_USER: str | None = Field(default="")
    NATS_PASSWORD: str | None = Field(default="")
    NATS_CONNECT_TIMEOUT: int = 5
    NATS_MAX_RECONNECT: int = -1  # Infinite retries
    NATS_RECONNECT_WAIT: int = 2

    # Stream Configuration
    STREAM_NAME: str = "EVENTS"
    STREAM_REPLICAS: int = 1
    STREAM_MAX_AGE_DAYS: int = 7
    STREAM_DUPLICATE_WINDOW: int = 120  # seconds

    # Consumer Configuration
    CONSUMER_NAME: str = "WORKERS"
    PRIORITY_CONSUMER_NAME: str = "PRIORITY_WORKERS"
    ACK_WAIT: int = 30  # seconds
    MAX_DELIVER: int = 5
    MAX_ACK_PENDING: int = 1000

    # Performance
    BATCH_SIZE: int = 50
    BATCH_SIZE_MIN: int = 10
    BATCH_SIZE_MAX: int = 200
    BATCH_TIMEOUT: float = 1.0  # seconds
    BATCH_DYNAMIC: bool = True

    # Auto-detected capabilities
    USE_MSGPACK: bool = Field(default=HAS_MSGPACK)
    USE_UVLOOP: bool = Field(default=HAS_UVLOOP)

    # Deduplication
    DEDUP_CACHE_MAX: int = 10000
    DEDUP_WINDOW_MINUTES: int = 5
    USE_REDIS_DEDUP: bool = False  # Off by default, enable if needed

    # Redis URL (dynamic based on environment)
    @property
    def REDIS_URL(self) -> str:
        """Dynamic Redis URL based on environment"""
        if os.getenv("DOCKER_MODE"):
            return "redis://:password@redis-p2:6379"
        else:
            return "redis://localhost:6380"

    # Circuit Breaker
    CIRCUIT_BREAKER_ENABLED: bool = True
    CIRCUIT_BREAKER_MAX_FAILURES: int = 5
    CIRCUIT_BREAKER_TIMEOUT: int = 60

    # Security
    REQUIRE_TENANT_ID: bool = True
    ENABLE_ENCRYPTION: bool = False  # Off until implemented

    # Observability
    LOG_LEVEL: str = "INFO"
    ENABLE_OTEL: bool = False  # Off by default

    # Self-Optimization
    ENABLE_SELF_OPTIMIZE: bool = True
    OPTIMIZE_INTERVAL: int = 60  # seconds
    PURGE_INTERVAL: int = 60  # seconds

    @field_validator("NATS_USER", "NATS_PASSWORD", mode="before")
    @classmethod
    def empty_string_to_none(cls, v):
        """Convert empty strings to None for NATS auth"""
        return None if v == "" else v

    model_config = {
        "env_file": ".env.p2",
        "env_prefix": "NEURALBUS_",
        "extra": "ignore",  # Ignore non-NEURALBUS variables
    }


# Create singleton config
config = NeuralBusConfig()

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Apply uvloop if enabled and available
if config.USE_UVLOOP and HAS_UVLOOP:
    import uvloop

    uvloop.install()
    logger.info("uvloop installed - async performance optimized")
