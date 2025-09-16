import logging

import tweedledum as td
from qiskit import QuantumCircuit
from tweedledum.bool_function_compiler import QuantumCircuitFunction
from tweedledum.passes import linear_resynth, parity_decomp
from tweedledum.synthesis import pkrm_synth

from .base import SynthesisCompiler, clique_oracle
from ..core import ProblemInstance
from ..problems import CliqueProblem

logger = logging.getLogger("benchmarklib.compiler.truth_table")


class TruthTableCompiler(SynthesisCompiler):
    """
    Compiler using truth table synthesis (PKRM - Positive-polarity Reed-Muller).

    This approach:
    1. Simulates the classical function to get its truth table
    2. Uses PKRM synthesis for the truth table
    3. Applies optimization passes
    4. Converts to phase-flip oracle

    Note: Truth table approach scales exponentially with problem size!
    """

    @property
    def name(self) -> str:
        return "TRUTH_TABLE"

    def compile(self, problem: ProblemInstance, **kwargs) -> QuantumCircuit:
        """
        Compile problem instance to phase-flip oracle using truth table synthesis.

        Args:
            problem: Problem instance to compile
            **kwargs: Problem-specific parameters

        Returns:
            Phase-flip oracle quantum circuit
        """
        if isinstance(problem, CliqueProblem):
            return self._compile_clique(problem, **kwargs)
        else:
            raise NotImplementedError(
                f"TruthTableCompiler doesn't support {problem.problem_type} problems yet"
            )

    def _compile_clique(self, problem: CliqueProblem, **kwargs) -> QuantumCircuit:
        """Compile clique problem using truth table synthesis."""
        clique_size = kwargs.get("clique_size")
        if clique_size is None:
            raise ValueError("clique_size must be specified for clique problems")

        # Choose the parameterized function
        param_func = clique_oracle

        # Get edge list from problem
        edges = problem.as_adjacency_matrix().flatten().tolist()

        # Create QuantumCircuitFunction
        n = problem.nodes
        classical_inputs = {"n": n, "k": clique_size, "edges": edges}
        qc_func = QuantumCircuitFunction(param_func, **classical_inputs)

        # Simulate to get truth table
        logger.debug("Simulating to get truth table...")
        qc_func.simulate_all()

        # Synthesize from truth table
        logger.debug("Synthesizing from truth table...")
        td_circuit = pkrm_synth(qc_func._truth_table[0])

        # Apply optimization passes
        logger.debug("Applying optimization passes...")
        td_circuit = parity_decomp(td_circuit)
        td_circuit = linear_resynth(td_circuit)

        # Convert to Qiskit
        qiskit_circuit = td.converters.to_qiskit(td_circuit, circuit_type="gatelist")

        # Convert to phase-flip oracle
        oracle_qubit = qiskit_circuit.num_qubits - 1

        phase_oracle = QuantumCircuit(qiskit_circuit.num_qubits)
        phase_oracle.x(oracle_qubit)
        phase_oracle.h(oracle_qubit)
        phase_oracle.compose(qiskit_circuit, inplace=True)
        phase_oracle.h(oracle_qubit)
        phase_oracle.x(oracle_qubit)

        return phase_oracle
