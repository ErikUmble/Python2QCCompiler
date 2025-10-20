import logging
from typing import Optional

import tweedledum as td
from qiskit import QuantumCircuit
from tweedledum.bool_function_compiler import QuantumCircuitFunction
from tweedledum.classical import optimize
from tweedledum.passes import linear_resynth, parity_decomp
from tweedledum.synthesis import xag_cleanup, xag_synth

from ... import CliqueProblem, BaseProblem
from .synthesizer import Synthesizer, SynthesizerRegistry, clique_oracle

logger = logging.getLogger("benchmarklib.pipeline.synthesis.xag")


@SynthesizerRegistry.register
class XAGSynthesizer(Synthesizer):
    """
    Synthesis using XAG (XOR-AND Graph) synthesis from Tweedledum.

    This compiler:
    1. Creates a QuantumCircuitFunction from the problem's classical function
    2. Optionally optimizes the XAG representation
    3. Synthesizes using xag_synth
    4. Applies optimization passes
    5. Converts to phase-flip oracle
    """

    def __init__(self):
        """
        Initialize XAG compiler.
        """

    @property
    def name(self) -> str:
        return str(self.__class__.__name__)

    def synthesize(self, problem: BaseProblem, **kwargs) -> QuantumCircuit:
        """
        Compile problem instance to phase-flip oracle using XAG synthesis.

        Args:
            problem: Problem instance to compile
            **kwargs: Problem-specific parameters (e.g., clique_size)

        Returns:
            Phase-flip oracle quantum circuit
        """
        # Determine which classical function to use based on problem type
        if isinstance(problem, CliqueProblem):
            return self._compile_clique(problem, **kwargs)
        else:
            raise NotImplementedError(
                f"XAGCompiler doesn't support {problem.problem_type} problems yet"
            )

    def _compile_clique(self, problem: CliqueProblem, **kwargs) -> QuantumCircuit:
        """Compile clique problem to oracle."""
        clique_size = kwargs.get("clique_size")
        if clique_size is None:
            raise ValueError("clique_size must be specified for clique problems")

        param_func = clique_oracle

        # Get edge list from problem
        edges = problem.as_adjacency_matrix().flatten().tolist()

        # Create QuantumCircuitFunction
        n = problem.nodes
        classical_inputs = {"n": n, "k": clique_size, "edges": edges}
        qc_func = QuantumCircuitFunction(param_func, **classical_inputs)

        # Get XAG and optionally optimize
        xag = qc_func.logic_network()

        logger.debug("Optimizing XAG...")
        xag = xag_cleanup(xag)
        optimize(xag)

        # Synthesize
        logger.debug("Synthesizing from XAG...")
        td_circuit = xag_synth(xag)

        # Apply Tweedledum optimization passes
        logger.debug("Applying optimization passes...")
        td_circuit = parity_decomp(td_circuit)
        td_circuit = linear_resynth(td_circuit)

        # Convert to Qiskit
        qiskit_circuit = td.converters.to_qiskit(td_circuit, circuit_type="gatelist")

        # Convert to phase-flip oracle
        # The oracle qubit is the last qubit (output of the function)
        self.oracle_qubit = n

        phase_oracle = QuantumCircuit(qiskit_circuit.num_qubits)
        phase_oracle.x(self.oracle_qubit)
        phase_oracle.h(self.oracle_qubit)
        phase_oracle.compose(qiskit_circuit, inplace=True)
        phase_oracle.h(self.oracle_qubit)
        phase_oracle.x(self.oracle_qubit)

        return phase_oracle

    def target_qubit(self) -> Optional[int]:
        """
        Return the index of the target qubit of the oracle
        """

        return self.oracle_qubit if self.oracle_qubit else None
