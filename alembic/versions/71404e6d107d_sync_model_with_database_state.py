"""sync model with database state

Revision ID: 71404e6d107d
Revises: a6bffa829264
Create Date: 2025-10-16 16:10:26.158323

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '71404e6d107d'
down_revision = 'a6bffa829264'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # All database indexes already exist
    # This migration documents the synchronized state
    pass


def downgrade() -> None:
    # Cannot downgrade from synchronized state
    pass