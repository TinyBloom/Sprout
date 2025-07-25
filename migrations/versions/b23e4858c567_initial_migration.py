"""Initial migration

Revision ID: b23e4858c567
Revises: 
Create Date: 2025-07-09 21:55:45.954662

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'b23e4858c567'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('datasets', schema=None) as batch_op:
        batch_op.alter_column('file_path',
               existing_type=sa.TEXT(),
               nullable=False)

    with op.batch_alter_table('model_files', schema=None) as batch_op:
        batch_op.alter_column('file_size',
               existing_type=sa.BIGINT(),
               nullable=False)
        batch_op.alter_column('file_format',
               existing_type=sa.VARCHAR(length=32),
               nullable=False)
        batch_op.alter_column('file_hash',
               existing_type=sa.VARCHAR(length=64),
               nullable=False)

    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.add_column(sa.Column('case_id', sa.String(length=36), nullable=True))
        batch_op.create_foreign_key(None, 'cases', ['case_id'], ['case_id'])

    with op.batch_alter_table('training_info', schema=None) as batch_op:
        batch_op.add_column(sa.Column('case_id', sa.String(length=36), nullable=True))
        batch_op.create_foreign_key(None, 'cases', ['case_id'], ['case_id'])

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('training_info', schema=None) as batch_op:
        batch_op.drop_constraint(None, type_='foreignkey')
        batch_op.drop_column('case_id')

    with op.batch_alter_table('models', schema=None) as batch_op:
        batch_op.drop_constraint(None, type_='foreignkey')
        batch_op.drop_column('case_id')

    with op.batch_alter_table('model_files', schema=None) as batch_op:
        batch_op.alter_column('file_hash',
               existing_type=sa.VARCHAR(length=64),
               nullable=True)
        batch_op.alter_column('file_format',
               existing_type=sa.VARCHAR(length=32),
               nullable=True)
        batch_op.alter_column('file_size',
               existing_type=sa.BIGINT(),
               nullable=True)

    with op.batch_alter_table('datasets', schema=None) as batch_op:
        batch_op.alter_column('file_path',
               existing_type=sa.TEXT(),
               nullable=True)

    # ### end Alembic commands ###
