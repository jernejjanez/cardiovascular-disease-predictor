"""add good_doctor and number of given feedback columns to user

Revision ID: 8d4b6770fbf3
Revises: 976209a51e44
Create Date: 2020-08-23 23:16:19.252582

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '8d4b6770fbf3'
down_revision = '976209a51e44'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('user', sa.Column('good_doctor', sa.Boolean(), nullable=True))
    op.add_column('user', sa.Column('num_of_feedback_given', sa.Integer(), nullable=True))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('user', 'num_of_feedback_given')
    op.drop_column('user', 'good_doctor')
    # ### end Alembic commands ###