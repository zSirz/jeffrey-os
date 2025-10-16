"""baseline schema (empty, DB already at state)

Revision ID: c43a1e16eb32
Revises:
Create Date: 2025-10-16 16:08:26.209668

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'c43a1e16eb32'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Cette migration représente l'état initial
    # Elle est vide car les tables existent déjà
    pass


def downgrade() -> None:
    pass