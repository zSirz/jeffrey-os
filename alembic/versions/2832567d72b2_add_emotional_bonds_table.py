"""add emotional bonds table

Revision ID: 2832567d72b2
Revises: 71404e6d107d
Create Date: 2025-10-16 16:11:10.593648

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID
import uuid


# revision identifiers, used by Alembic.
revision = '2832567d72b2'
down_revision = '71404e6d107d'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'emotional_bonds',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('memory_id_a', UUID(as_uuid=True), nullable=False),
        sa.Column('memory_id_b', UUID(as_uuid=True), nullable=False),
        sa.Column('strength', sa.Float(), nullable=False, server_default='0'),
        sa.Column('emotion_match', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('last_updated', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.ForeignKeyConstraint(['memory_id_a'], ['memories.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['memory_id_b'], ['memories.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('memory_id_a', 'memory_id_b', name='unique_bond_pair')
    )

    op.create_index('idx_bonds_strength', 'emotional_bonds', ['strength'])
    op.create_index('idx_bonds_memories', 'emotional_bonds', ['memory_id_a', 'memory_id_b'])


def downgrade() -> None:
    op.drop_table('emotional_bonds')