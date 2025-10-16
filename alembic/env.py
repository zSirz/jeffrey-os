import sys, os, asyncio
from logging.config import fileConfig
from alembic import context
from sqlalchemy.ext.asyncio import create_async_engine

# Rendre le code appli visible
sys.path.insert(0, "/app/src")

# Config Alembic
config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Import config appli + modèles
from jeffrey.core.config import settings
from jeffrey.models.base import Base
# Import explicite de tous les modèles pour peupler metadata
from jeffrey.models.memory import Memory, EmotionEvent, DreamRun

target_metadata = Base.metadata

# Override l'URL depuis les settings de l'app
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

def run_migrations_offline():
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()

def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()

async def run_async_migrations():
    """Run migrations in 'online' mode with async."""
    connectable = create_async_engine(
        config.get_main_option("sqlalchemy.url"),
        future=True
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()

def run_migrations_online():
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()