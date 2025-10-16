"""add embeddings column to memories

Revision ID: d3eb54e6b28f
Revises:
Create Date: 2025-10-16 15:39:51.527980

"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision = 'd3eb54e6b28f'
down_revision = 'c43a1e16eb32'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Ensure pgvector extension is available
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')

    # Add embedding column with Vector type (384 dimensions for all-MiniLM-L6-v2)
    op.add_column('memories',
        sa.Column('embedding', Vector(384), nullable=True)
    )

    # Create index for fast cosine similarity search
    op.execute('CREATE INDEX idx_memories_embedding_cosine ON memories USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)')

    # Analyze table for optimal query planning
    op.execute('ANALYZE memories')


def downgrade() -> None:
    # Drop index first
    op.execute('DROP INDEX IF EXISTS idx_memories_embedding_cosine')

    # Drop embedding column
    op.drop_column('memories', 'embedding')