"""Clean up schema and add extra_data to trials

Revision ID: 74609be229b0
Revises: 55e64ee84a59
Create Date: 2026-05-09 16:39:52.068503

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite

# revision identifiers, used by Alembic.
revision: str = '74609be229b0'
down_revision: Union[str, Sequence[str], None] = '55e64ee84a59'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    ## Drop unrelated tables that had been accidently included in prior schema
    op.drop_index(op.f('ix_benchmarklib_clique_oracle_trials_compiler_name'), table_name='benchmarklib_clique_oracle_trials')
    op.drop_index(op.f('ix_benchmarklib_clique_oracle_trials_is_pending'), table_name='benchmarklib_clique_oracle_trials', sqlite_where=sa.text('counts IS NULL AND is_failed = 0'))
    op.drop_index(op.f('ix_benchmarklib_clique_oracle_trials_job_id'), table_name='benchmarklib_clique_oracle_trials')
    op.drop_index(op.f('ix_benchmarklib_clique_oracle_trials_pipeline_config_id'), table_name='benchmarklib_clique_oracle_trials')
    op.drop_index(op.f('ix_benchmarklib_clique_oracle_trials_problem_id'), table_name='benchmarklib_clique_oracle_trials')
    op.drop_table('benchmarklib_clique_oracle_trials')
    op.drop_table('benchmarklib_clique_problems')
    op.drop_index(op.f('ix_num_vars_complexity'), table_name='random_boolean_functions')
    op.drop_table('random_boolean_functions')
    op.drop_index(op.f('ix_three_sat_trials_compiler_name'), table_name='three_sat_trials')
    op.drop_index(op.f('ix_three_sat_trials_is_pending'), table_name='three_sat_trials', sqlite_where=sa.text('counts IS NULL AND is_failed = 0'))
    op.drop_index(op.f('ix_three_sat_trials_job_id'), table_name='three_sat_trials')
    op.drop_index(op.f('ix_three_sat_trials_pipeline_config_id'), table_name='three_sat_trials')
    op.drop_index(op.f('ix_three_sat_trials_problem_id'), table_name='three_sat_trials')
    op.drop_table('three_sat_trials')
    op.drop_index(op.f('ix_random_boolean_function_trials_compiler_name'), table_name='random_boolean_function_trials')
    op.drop_index(op.f('ix_random_boolean_function_trials_is_pending'), table_name='random_boolean_function_trials', sqlite_where=sa.text('counts IS NULL AND is_failed = 0'))
    op.drop_index(op.f('ix_random_boolean_function_trials_job_id'), table_name='random_boolean_function_trials')
    op.drop_index(op.f('ix_random_boolean_function_trials_problem_id'), table_name='random_boolean_function_trials')
    op.drop_table('random_boolean_function_trials')
    op.drop_index(op.f('ix_quantum_fidelity_trials_backend_name'), table_name='quantum_fidelity_trials')
    op.drop_index(op.f('ix_quantum_fidelity_trials_is_pending'), table_name='quantum_fidelity_trials', sqlite_where=sa.text('counts IS NULL AND is_failed = 0'))
    op.drop_index(op.f('ix_quantum_fidelity_trials_job_id'), table_name='quantum_fidelity_trials')
    op.drop_index(op.f('ix_quantum_fidelity_trials_pipeline_config_id'), table_name='quantum_fidelity_trials')
    op.drop_index(op.f('ix_quantum_fidelity_trials_problem_id'), table_name='quantum_fidelity_trials')
    op.drop_index(op.f('ix_quantum_fidelity_trials_trial_category'), table_name='quantum_fidelity_trials')
    op.drop_table('quantum_fidelity_trials')
    op.drop_table('three_sat_problems')
    op.drop_index(op.f('ix_experiment_trials_compiler_name'), table_name='experiment_trials')
    op.drop_index(op.f('ix_experiment_trials_is_pending'), table_name='experiment_trials', sqlite_where=sa.text('counts IS NULL AND is_failed = 0'))
    op.drop_index(op.f('ix_experiment_trials_job_id'), table_name='experiment_trials')
    op.drop_index(op.f('ix_experiment_trials_pipeline_config_id'), table_name='experiment_trials')
    op.drop_index(op.f('ix_experiment_trials_problem_id'), table_name='experiment_trials')
    op.drop_table('experiment_trials')
    op.drop_index(op.f('ix_benchmarklib_clique_trials_compiler_name'), table_name='benchmarklib_clique_trials')
    op.drop_index(op.f('ix_benchmarklib_clique_trials_is_pending'), table_name='benchmarklib_clique_trials', sqlite_where=sa.text('counts IS NULL AND is_failed = 0'))
    op.drop_index(op.f('ix_benchmarklib_clique_trials_job_id'), table_name='benchmarklib_clique_trials')
    op.drop_index(op.f('ix_benchmarklib_clique_trials_pipeline_config_id'), table_name='benchmarklib_clique_trials')
    op.drop_index(op.f('ix_benchmarklib_clique_trials_problem_id'), table_name='benchmarklib_clique_trials')
    op.drop_table('benchmarklib_clique_trials')
    op.drop_index(op.f('ix_quantum_fidelity_problems_circuit_category'), table_name='quantum_fidelity_problems')
    op.drop_table('quantum_fidelity_problems')
    op.drop_index(op.f('ix_experiment_problems_name'), table_name='experiment_problems')
    op.drop_index(op.f('ix_experiment_problems_tag'), table_name='experiment_problems')
    op.drop_table('experiment_problems')
    op.drop_index(op.f('ix_three_sat_oracle_trials_compiler_name'), table_name='three_sat_oracle_trials')
    op.drop_index(op.f('ix_three_sat_oracle_trials_is_pending'), table_name='three_sat_oracle_trials', sqlite_where=sa.text('counts IS NULL AND is_failed = 0'))
    op.drop_index(op.f('ix_three_sat_oracle_trials_job_id'), table_name='three_sat_oracle_trials')
    op.drop_index(op.f('ix_three_sat_oracle_trials_pipeline_config_id'), table_name='three_sat_oracle_trials')
    op.drop_index(op.f('ix_three_sat_oracle_trials_problem_id'), table_name='three_sat_oracle_trials')
    op.drop_table('three_sat_oracle_trials')
    op.add_column('backend_properties', sa.Column('properties', sa.JSON(), nullable=False))
    op.drop_column('backend_properties', 'gates_data')

    ## Main change
    op.add_column('clique_oracle_trials', sa.Column('extra_data', sa.JSON(), nullable=True))
    op.alter_column('clique_oracle_trials', 'expected_output',
               existing_type=sa.BOOLEAN(),
               nullable=True)
    op.add_column('clique_trials', sa.Column('extra_data', sa.JSON(), nullable=True))
    op.alter_column('clique_trials', 'compiler_name',
               existing_type=sa.VARCHAR(length=255),
               nullable=True)
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column('clique_trials', 'compiler_name',
               existing_type=sa.VARCHAR(length=255),
               nullable=False)
    op.drop_column('clique_trials', 'extra_data')
    op.alter_column('clique_oracle_trials', 'compiler_name',
               existing_type=sa.VARCHAR(length=255),
               nullable=False)
    op.alter_column('clique_oracle_trials', 'expected_output',
               existing_type=sa.BOOLEAN(),
               nullable=False)
    op.drop_column('clique_oracle_trials', 'extra_data')
    op.add_column('backend_properties', sa.Column('gates_data', sqlite.JSON(), nullable=False))
    op.drop_column('backend_properties', 'properties')
    op.create_table('three_sat_oracle_trials',
    sa.Column('input_state', sa.VARCHAR(), nullable=False),
    sa.Column('expected_output', sa.BOOLEAN(), nullable=True),
    sa.Column('circuit_depth', sa.INTEGER(), nullable=True),
    sa.Column('circuit_op_counts', sqlite.JSON(), nullable=True),
    sa.Column('circuit_num_single_qubit_gates', sa.INTEGER(), nullable=True),
    sa.Column('circuit_num_gates', sa.INTEGER(), nullable=True),
    sa.Column('circuit_num_qubits', sa.INTEGER(), nullable=True),
    sa.Column('compiler_name', sa.VARCHAR(length=255), nullable=True),
    sa.Column('job_id', sa.VARCHAR(length=128), nullable=True),
    sa.Column('job_pub_idx', sa.INTEGER(), nullable=True),
    sa.Column('counts', sqlite.JSON(), nullable=True),
    sa.Column('simulation_counts', sqlite.JSON(), nullable=True),
    sa.Column('is_failed', sa.BOOLEAN(), nullable=False),
    sa.Column('_circuit_qpy', sa.BLOB(), nullable=True),
    sa.Column('_circuit_pretranspile_qpy', sa.BLOB(), nullable=True),
    sa.Column('problem_id', sa.INTEGER(), nullable=False),
    sa.Column('pipeline_config_id', sa.INTEGER(), nullable=True),
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('created_at', sa.DATETIME(), nullable=False),
    sa.Column('updated_at', sa.DATETIME(), nullable=False),
    sa.ForeignKeyConstraint(['pipeline_config_id'], ['pipeline_configs.id'], ),
    sa.ForeignKeyConstraint(['problem_id'], ['three_sat_problems.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_three_sat_oracle_trials_problem_id'), 'three_sat_oracle_trials', ['problem_id'], unique=False)
    op.create_index(op.f('ix_three_sat_oracle_trials_pipeline_config_id'), 'three_sat_oracle_trials', ['pipeline_config_id'], unique=False)
    op.create_index(op.f('ix_three_sat_oracle_trials_job_id'), 'three_sat_oracle_trials', ['job_id'], unique=False)
    op.create_index(op.f('ix_three_sat_oracle_trials_is_pending'), 'three_sat_oracle_trials', ['job_id'], unique=False, sqlite_where=sa.text('counts IS NULL AND is_failed = 0'))
    op.create_index(op.f('ix_three_sat_oracle_trials_compiler_name'), 'three_sat_oracle_trials', ['compiler_name'], unique=False)
    op.create_table('experiment_problems',
    sa.Column('n', sa.INTEGER(), nullable=False),
    sa.Column('name', sa.VARCHAR(length=128), nullable=False),
    sa.Column('tag', sa.VARCHAR(length=64), nullable=True),
    sa.Column('notes', sa.TEXT(), nullable=True),
    sa.Column('extra_data', sqlite.JSON(), nullable=False),
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('created_at', sa.DATETIME(), nullable=False),
    sa.Column('updated_at', sa.DATETIME(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_experiment_problems_tag'), 'experiment_problems', ['tag'], unique=False)
    op.create_index(op.f('ix_experiment_problems_name'), 'experiment_problems', ['name'], unique=False)
    op.create_table('quantum_fidelity_problems',
    sa.Column('num_qubits', sa.INTEGER(), nullable=False),
    sa.Column('circuit_category', sa.VARCHAR(length=14), nullable=False),
    sa.Column('circuit_qasm', sa.TEXT(), nullable=True),
    sa.Column('expected_measurement', sa.VARCHAR(length=256), nullable=True),
    sa.Column('extra_data', sqlite.JSON(), nullable=False),
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('created_at', sa.DATETIME(), nullable=False),
    sa.Column('updated_at', sa.DATETIME(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_quantum_fidelity_problems_circuit_category'), 'quantum_fidelity_problems', ['circuit_category'], unique=False)
    op.create_table('benchmarklib_clique_trials',
    sa.Column('grover_iterations', sa.INTEGER(), nullable=True),
    sa.Column('clique_size', sa.INTEGER(), nullable=False),
    sa.Column('circuit_depth', sa.INTEGER(), nullable=True),
    sa.Column('circuit_op_counts', sqlite.JSON(), nullable=True),
    sa.Column('circuit_num_single_qubit_gates', sa.INTEGER(), nullable=True),
    sa.Column('circuit_num_gates', sa.INTEGER(), nullable=True),
    sa.Column('circuit_num_qubits', sa.INTEGER(), nullable=True),
    sa.Column('compiler_name', sa.VARCHAR(length=255), nullable=True),
    sa.Column('job_id', sa.VARCHAR(length=128), nullable=True),
    sa.Column('job_pub_idx', sa.INTEGER(), nullable=True),
    sa.Column('counts', sqlite.JSON(), nullable=True),
    sa.Column('simulation_counts', sqlite.JSON(), nullable=True),
    sa.Column('is_failed', sa.BOOLEAN(), nullable=False),
    sa.Column('_circuit_qpy', sa.BLOB(), nullable=True),
    sa.Column('_circuit_pretranspile_qpy', sa.BLOB(), nullable=True),
    sa.Column('problem_id', sa.INTEGER(), nullable=False),
    sa.Column('pipeline_config_id', sa.INTEGER(), nullable=True),
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('created_at', sa.DATETIME(), nullable=False),
    sa.Column('updated_at', sa.DATETIME(), nullable=False),
    sa.ForeignKeyConstraint(['pipeline_config_id'], ['pipeline_configs.id'], ),
    sa.ForeignKeyConstraint(['problem_id'], ['benchmarklib_clique_problems.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_benchmarklib_clique_trials_problem_id'), 'benchmarklib_clique_trials', ['problem_id'], unique=False)
    op.create_index(op.f('ix_benchmarklib_clique_trials_pipeline_config_id'), 'benchmarklib_clique_trials', ['pipeline_config_id'], unique=False)
    op.create_index(op.f('ix_benchmarklib_clique_trials_job_id'), 'benchmarklib_clique_trials', ['job_id'], unique=False)
    op.create_index(op.f('ix_benchmarklib_clique_trials_is_pending'), 'benchmarklib_clique_trials', ['job_id'], unique=False, sqlite_where=sa.text('counts IS NULL AND is_failed = 0'))
    op.create_index(op.f('ix_benchmarklib_clique_trials_compiler_name'), 'benchmarklib_clique_trials', ['compiler_name'], unique=False)
    op.create_table('experiment_trials',
    sa.Column('compiler_name', sa.VARCHAR(length=255), nullable=True),
    sa.Column('extra_data', sqlite.JSON(), nullable=False),
    sa.Column('circuit_depth', sa.INTEGER(), nullable=True),
    sa.Column('circuit_op_counts', sqlite.JSON(), nullable=True),
    sa.Column('circuit_num_single_qubit_gates', sa.INTEGER(), nullable=True),
    sa.Column('circuit_num_gates', sa.INTEGER(), nullable=True),
    sa.Column('circuit_num_qubits', sa.INTEGER(), nullable=True),
    sa.Column('job_id', sa.VARCHAR(length=128), nullable=True),
    sa.Column('job_pub_idx', sa.INTEGER(), nullable=True),
    sa.Column('counts', sqlite.JSON(), nullable=True),
    sa.Column('simulation_counts', sqlite.JSON(), nullable=True),
    sa.Column('is_failed', sa.BOOLEAN(), nullable=False),
    sa.Column('_circuit_qpy', sa.BLOB(), nullable=True),
    sa.Column('_circuit_pretranspile_qpy', sa.BLOB(), nullable=True),
    sa.Column('problem_id', sa.INTEGER(), nullable=False),
    sa.Column('pipeline_config_id', sa.INTEGER(), nullable=True),
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('created_at', sa.DATETIME(), nullable=False),
    sa.Column('updated_at', sa.DATETIME(), nullable=False),
    sa.ForeignKeyConstraint(['pipeline_config_id'], ['pipeline_configs.id'], ),
    sa.ForeignKeyConstraint(['problem_id'], ['experiment_problems.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_experiment_trials_problem_id'), 'experiment_trials', ['problem_id'], unique=False)
    op.create_index(op.f('ix_experiment_trials_pipeline_config_id'), 'experiment_trials', ['pipeline_config_id'], unique=False)
    op.create_index(op.f('ix_experiment_trials_job_id'), 'experiment_trials', ['job_id'], unique=False)
    op.create_index(op.f('ix_experiment_trials_is_pending'), 'experiment_trials', ['job_id'], unique=False, sqlite_where=sa.text('counts IS NULL AND is_failed = 0'))
    op.create_index(op.f('ix_experiment_trials_compiler_name'), 'experiment_trials', ['compiler_name'], unique=False)
    op.create_table('three_sat_problems',
    sa.Column('num_vars', sa.INTEGER(), nullable=False),
    sa.Column('expr', sqlite.JSON(), nullable=False),
    sa.Column('solutions', sqlite.JSON(), nullable=False),
    sa.Column('seed', sa.INTEGER(), nullable=False),
    sa.Column('extra_data', sqlite.JSON(), nullable=False),
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('created_at', sa.DATETIME(), nullable=False),
    sa.Column('updated_at', sa.DATETIME(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('quantum_fidelity_trials',
    sa.Column('trial_category', sa.VARCHAR(length=23), nullable=False),
    sa.Column('extra_data', sqlite.JSON(), nullable=False),
    sa.Column('backend_name', sa.VARCHAR(length=128), nullable=True),
    sa.Column('circuit_depth', sa.INTEGER(), nullable=True),
    sa.Column('circuit_op_counts', sqlite.JSON(), nullable=True),
    sa.Column('circuit_num_single_qubit_gates', sa.INTEGER(), nullable=True),
    sa.Column('circuit_num_gates', sa.INTEGER(), nullable=True),
    sa.Column('circuit_num_qubits', sa.INTEGER(), nullable=True),
    sa.Column('job_id', sa.VARCHAR(length=128), nullable=True),
    sa.Column('job_pub_idx', sa.INTEGER(), nullable=True),
    sa.Column('counts', sqlite.JSON(), nullable=True),
    sa.Column('simulation_counts', sqlite.JSON(), nullable=True),
    sa.Column('is_failed', sa.BOOLEAN(), nullable=False),
    sa.Column('_circuit_qpy', sa.BLOB(), nullable=True),
    sa.Column('_circuit_pretranspile_qpy', sa.BLOB(), nullable=True),
    sa.Column('problem_id', sa.INTEGER(), nullable=False),
    sa.Column('pipeline_config_id', sa.INTEGER(), nullable=True),
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('created_at', sa.DATETIME(), nullable=False),
    sa.Column('updated_at', sa.DATETIME(), nullable=False),
    sa.ForeignKeyConstraint(['pipeline_config_id'], ['pipeline_configs.id'], ),
    sa.ForeignKeyConstraint(['problem_id'], ['quantum_fidelity_problems.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_quantum_fidelity_trials_trial_category'), 'quantum_fidelity_trials', ['trial_category'], unique=False)
    op.create_index(op.f('ix_quantum_fidelity_trials_problem_id'), 'quantum_fidelity_trials', ['problem_id'], unique=False)
    op.create_index(op.f('ix_quantum_fidelity_trials_pipeline_config_id'), 'quantum_fidelity_trials', ['pipeline_config_id'], unique=False)
    op.create_index(op.f('ix_quantum_fidelity_trials_job_id'), 'quantum_fidelity_trials', ['job_id'], unique=False)
    op.create_index(op.f('ix_quantum_fidelity_trials_is_pending'), 'quantum_fidelity_trials', ['job_id'], unique=False, sqlite_where=sa.text('counts IS NULL AND is_failed = 0'))
    op.create_index(op.f('ix_quantum_fidelity_trials_backend_name'), 'quantum_fidelity_trials', ['backend_name'], unique=False)
    op.create_table('random_boolean_function_trials',
    sa.Column('input_state', sa.VARCHAR(length=255), nullable=False),
    sa.Column('circuit_depth', sa.INTEGER(), nullable=True),
    sa.Column('circuit_op_counts', sqlite.JSON(), nullable=True),
    sa.Column('circuit_num_single_qubit_gates', sa.INTEGER(), nullable=True),
    sa.Column('circuit_num_gates', sa.INTEGER(), nullable=True),
    sa.Column('circuit_num_qubits', sa.INTEGER(), nullable=True),
    sa.Column('compiler_name', sa.VARCHAR(length=255), nullable=True),
    sa.Column('job_id', sa.VARCHAR(length=128), nullable=True),
    sa.Column('job_pub_idx', sa.INTEGER(), nullable=True),
    sa.Column('counts', sqlite.JSON(), nullable=True),
    sa.Column('simulation_counts', sqlite.JSON(), nullable=True),
    sa.Column('is_failed', sa.BOOLEAN(), nullable=False),
    sa.Column('_circuit_qpy', sa.BLOB(), nullable=True),
    sa.Column('_circuit_pretranspile_qpy', sa.BLOB(), nullable=True),
    sa.Column('problem_id', sa.INTEGER(), nullable=False),
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('created_at', sa.DATETIME(), nullable=False),
    sa.Column('updated_at', sa.DATETIME(), nullable=False),
    sa.ForeignKeyConstraint(['problem_id'], ['random_boolean_functions.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_random_boolean_function_trials_problem_id'), 'random_boolean_function_trials', ['problem_id'], unique=False)
    op.create_index(op.f('ix_random_boolean_function_trials_job_id'), 'random_boolean_function_trials', ['job_id'], unique=False)
    op.create_index(op.f('ix_random_boolean_function_trials_is_pending'), 'random_boolean_function_trials', ['job_id'], unique=False, sqlite_where=sa.text('counts IS NULL AND is_failed = 0'))
    op.create_index(op.f('ix_random_boolean_function_trials_compiler_name'), 'random_boolean_function_trials', ['compiler_name'], unique=False)
    op.create_table('three_sat_trials',
    sa.Column('grover_iters', sa.INTEGER(), nullable=True),
    sa.Column('circuit_depth', sa.INTEGER(), nullable=True),
    sa.Column('circuit_op_counts', sqlite.JSON(), nullable=True),
    sa.Column('circuit_num_single_qubit_gates', sa.INTEGER(), nullable=True),
    sa.Column('circuit_num_gates', sa.INTEGER(), nullable=True),
    sa.Column('circuit_num_qubits', sa.INTEGER(), nullable=True),
    sa.Column('compiler_name', sa.VARCHAR(length=255), nullable=True),
    sa.Column('job_id', sa.VARCHAR(length=128), nullable=True),
    sa.Column('job_pub_idx', sa.INTEGER(), nullable=True),
    sa.Column('counts', sqlite.JSON(), nullable=True),
    sa.Column('simulation_counts', sqlite.JSON(), nullable=True),
    sa.Column('is_failed', sa.BOOLEAN(), nullable=False),
    sa.Column('_circuit_qpy', sa.BLOB(), nullable=True),
    sa.Column('_circuit_pretranspile_qpy', sa.BLOB(), nullable=True),
    sa.Column('problem_id', sa.INTEGER(), nullable=False),
    sa.Column('pipeline_config_id', sa.INTEGER(), nullable=True),
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('created_at', sa.DATETIME(), nullable=False),
    sa.Column('updated_at', sa.DATETIME(), nullable=False),
    sa.ForeignKeyConstraint(['pipeline_config_id'], ['pipeline_configs.id'], ),
    sa.ForeignKeyConstraint(['problem_id'], ['three_sat_problems.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_three_sat_trials_problem_id'), 'three_sat_trials', ['problem_id'], unique=False)
    op.create_index(op.f('ix_three_sat_trials_pipeline_config_id'), 'three_sat_trials', ['pipeline_config_id'], unique=False)
    op.create_index(op.f('ix_three_sat_trials_job_id'), 'three_sat_trials', ['job_id'], unique=False)
    op.create_index(op.f('ix_three_sat_trials_is_pending'), 'three_sat_trials', ['job_id'], unique=False, sqlite_where=sa.text('counts IS NULL AND is_failed = 0'))
    op.create_index(op.f('ix_three_sat_trials_compiler_name'), 'three_sat_trials', ['compiler_name'], unique=False)
    op.create_table('random_boolean_functions',
    sa.Column('num_vars', sa.INTEGER(), nullable=False),
    sa.Column('complexity', sa.INTEGER(), nullable=False),
    sa.Column('statement', sa.TEXT(), nullable=False),
    sa.Column('num_satisfying_inputs', sa.INTEGER(), nullable=True),
    sa.Column('extra_data', sqlite.JSON(), nullable=False),
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('created_at', sa.DATETIME(), nullable=False),
    sa.Column('updated_at', sa.DATETIME(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_num_vars_complexity'), 'random_boolean_functions', ['num_vars', 'complexity'], unique=False)
    op.create_table('benchmarklib_clique_problems',
    sa.Column('graph', sa.VARCHAR(), nullable=False),
    sa.Column('nodes', sa.INTEGER(), nullable=False),
    sa.Column('edge_probability', sa.INTEGER(), nullable=True),
    sa.Column('_clique_counts', sqlite.JSON(), nullable=True),
    sa.Column('extra_data', sqlite.JSON(), nullable=False),
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('created_at', sa.DATETIME(), nullable=False),
    sa.Column('updated_at', sa.DATETIME(), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('graph')
    )
    op.create_table('benchmarklib_clique_oracle_trials',
    sa.Column('input_state', sa.VARCHAR(), nullable=False),
    sa.Column('expected_output', sa.BOOLEAN(), nullable=True),
    sa.Column('circuit_depth', sa.INTEGER(), nullable=True),
    sa.Column('circuit_op_counts', sqlite.JSON(), nullable=True),
    sa.Column('circuit_num_single_qubit_gates', sa.INTEGER(), nullable=True),
    sa.Column('circuit_num_gates', sa.INTEGER(), nullable=True),
    sa.Column('circuit_num_qubits', sa.INTEGER(), nullable=True),
    sa.Column('compiler_name', sa.VARCHAR(length=255), nullable=True),
    sa.Column('job_id', sa.VARCHAR(length=128), nullable=True),
    sa.Column('job_pub_idx', sa.INTEGER(), nullable=True),
    sa.Column('counts', sqlite.JSON(), nullable=True),
    sa.Column('simulation_counts', sqlite.JSON(), nullable=True),
    sa.Column('is_failed', sa.BOOLEAN(), nullable=False),
    sa.Column('_circuit_qpy', sa.BLOB(), nullable=True),
    sa.Column('_circuit_pretranspile_qpy', sa.BLOB(), nullable=True),
    sa.Column('problem_id', sa.INTEGER(), nullable=False),
    sa.Column('pipeline_config_id', sa.INTEGER(), nullable=True),
    sa.Column('id', sa.INTEGER(), nullable=False),
    sa.Column('created_at', sa.DATETIME(), nullable=False),
    sa.Column('updated_at', sa.DATETIME(), nullable=False),
    sa.ForeignKeyConstraint(['pipeline_config_id'], ['pipeline_configs.id'], ),
    sa.ForeignKeyConstraint(['problem_id'], ['benchmarklib_clique_problems.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_benchmarklib_clique_oracle_trials_problem_id'), 'benchmarklib_clique_oracle_trials', ['problem_id'], unique=False)
    op.create_index(op.f('ix_benchmarklib_clique_oracle_trials_pipeline_config_id'), 'benchmarklib_clique_oracle_trials', ['pipeline_config_id'], unique=False)
    op.create_index(op.f('ix_benchmarklib_clique_oracle_trials_job_id'), 'benchmarklib_clique_oracle_trials', ['job_id'], unique=False)
    op.create_index(op.f('ix_benchmarklib_clique_oracle_trials_is_pending'), 'benchmarklib_clique_oracle_trials', ['job_id'], unique=False, sqlite_where=sa.text('counts IS NULL AND is_failed = 0'))
    op.create_index(op.f('ix_benchmarklib_clique_oracle_trials_compiler_name'), 'benchmarklib_clique_oracle_trials', ['compiler_name'], unique=False)
    # ### end Alembic commands ###
