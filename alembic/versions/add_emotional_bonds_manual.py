"""add emotional bonds table

Revision ID: add_emotional_bonds
Revises: brain_enrich_final
Create Date: 2025-10-16 20:20:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'add_emotional_bonds'
down_revision = 'brain_enrich_final'
branch_labels = None
depends_on = None


def upgrade():
    # Create emotional_bonds table
    op.create_table('emotional_bonds',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('memory_id_a', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('memory_id_b', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('strength', sa.Float(), nullable=False),
        sa.Column('emotion_match', sa.Boolean(), nullable=True),
        sa.Column('last_updated', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['memory_id_a'], ['memories.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['memory_id_b'], ['memories.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('memory_id_a', 'memory_id_b', name='unique_bond_pair'),
        sa.CheckConstraint('memory_id_a < memory_id_b', name='check_ordered_ids')
    )
    
    # Create index on strength for efficient sorting
    op.create_index('ix_emotional_bonds_strength', 'emotional_bonds', ['strength'], unique=False)
    
    # Create indexes for FK lookups
    op.create_index('ix_emotional_bonds_memory_id_a', 'emotional_bonds', ['memory_id_a'], unique=False)
    op.create_index('ix_emotional_bonds_memory_id_b', 'emotional_bonds', ['memory_id_b'], unique=False)


def downgrade():
    # Drop indexes
    op.drop_index('ix_emotional_bonds_memory_id_b', table_name='emotional_bonds')
    op.drop_index('ix_emotional_bonds_memory_id_a', table_name='emotional_bonds')
    op.drop_index('ix_emotional_bonds_strength', table_name='emotional_bonds')
    
    # Drop table
    op.drop_table('emotional_bonds')