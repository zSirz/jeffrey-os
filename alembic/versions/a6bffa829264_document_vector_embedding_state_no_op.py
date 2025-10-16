"""document vector embedding state (no-op)

Revision ID: a6bffa829264
Revises: c43a1e16eb32
Create Date: 2025-10-16 16:08:53.429661

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a6bffa829264'
down_revision = 'c43a1e16eb32'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Ces changements ont déjà été appliqués manuellement
    # Cette migration les documente pour l'historique
    op.execute("-- Column embedding already converted from JSON to vector(384)")
    op.execute("-- Index idx_memories_embedding_cosine already created")
    op.execute("-- Index idx_memories_composite already exists")
    op.execute("-- Index idx_memories_emotion already exists")
    op.execute("-- Index idx_memories_processed already exists")
    op.execute("-- Index idx_memories_text_gin already exists")
    op.execute("-- Pgvector extension already enabled")


def downgrade() -> None:
    # Ne pas permettre le rollback car cela casserait les embeddings existants
    raise NotImplementedError("Cannot rollback vector embeddings - would break existing embeddings")