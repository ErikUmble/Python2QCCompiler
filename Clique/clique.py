# Import the research library
import importlib.util
import itertools
import json
import math
import tempfile
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import qiskit
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import grover_operator
from qiskit.providers import Backend
from sqlalchemy import JSON, String
from sqlalchemy.orm import Mapped, mapped_column
from tweedledum import BitVec
from tweedledum.bool_function_compiler import QuantumCircuitFunction
from tweedledum.bool_function_compiler.decorators import circuit_input

from benchmarklib import BaseTrial, BaseProblem, BenchmarkDatabase, CompileType, TrialCircuitMetricsMixin
from benchmarklib.algorithms.grover import build_grover_circuit, calculate_grover_iterations
from benchmarklib.runners.queue import BatchQueue
from benchmarklib.core.types import _BaseTrial, _ProblemInstance, classproperty


@circuit_input(vertices=lambda n: BitVec(n))
def parameterized_clique_counter_cardinality(n: int, k: int, edges) -> BitVec(1):
    """Counts cliques of size 2 in a graph specified by the edge list."""
    s = BitVec(1, 1)  # Start with True (assuming a clique)

    # Check if any non-connected vertices are both selected
    for i in range(n):
        for j in range(i + 1, n):
            # If vertices i and j are both selected (=1) AND there's no edge between them (=0)
            # then it's not a clique
            if edges[i * n + j] == 0:
                s = s & ~(vertices[i] & vertices[j])

    generate_at_least_k_counter(vertices, n, k)

    return s & at_least_k


@circuit_input(vertices=lambda n: BitVec(n))
def parameterized_clique_counter_batcher(n: int, k: int, edges) -> BitVec(1):
    """Counts cliques of size k in a graph specified by the edge list."""
    s = BitVec(1, 1)  # Start with True (assuming a clique)

    # Check if any non-connected vertices are both selected
    for i in range(n):
        for j in range(i + 1, n):
            # If vertices i and j are both selected (=1) AND there's no edge between them (=0)
            # then it's not a clique
            if edges[i * n + j] == 0:
                s = s & ~(vertices[i] & vertices[j])

    # generate_sorting_network(vertices, n, k)
    generate_batcher_sort_network(vertices, n, k)

    return s & sorted_bit_0


class SortPairNode:
    def __init__(self, high, low):
        self.high = high
        self.low = low

def get_sort_statements(variables):
    num_variables = len(variables)
    statements = []

    nodes = [[SortPairNode(None, None) for _ in range(num_variables)] for _ in range(num_variables)]
    for i in range(num_variables):
        nodes[i][0] = SortPairNode(variables[i], None)

    for i in range(1, num_variables):
        for j in range(1, i+1):
            s_high = f"s_{i}_{j}_high"
            s_low = f"s_{i}_{j}_low"
            nodes[i][j] = SortPairNode(s_high, s_low)

            if j == i:
                statements.append(f"{s_high} = {nodes[i-1][j-1].high} | {nodes[i][j-1].high}")
                statements.append(f"{s_low} = {nodes[i-1][j-1].high} & {nodes[i][j-1].high}")
            else:
                statements.append(f"{s_high} = {nodes[i-1][j].low} | {nodes[i][j-1].high}")
                statements.append(f"{s_low} = {nodes[i-1][j].low} & {nodes[i][j-1].high}")

    outputs = [nodes[num_variables-1][num_variables-1].high] + [nodes[num_variables-1][i].low for i in range(num_variables-1, 0, -1)]

    return statements, outputs

def construct_clique_verifier(graph, clique_size=None):
    """ 
    Given a graph in the form of binary string 
    e_11 e_12 e_13 ... e_1n e_23 e_24 ... e_2n ... e_n-1n, returns the string of a python function that takes n boolean variables denoting vertices 
    True if in the clique and False if not,
    and returns whether the input is a clique of size at least n/2 in the graph.

    if clique_size is unspecified, the default is to require at least n/2 vertices
    """
    n = int((1 + (1 + 8*len(graph))**0.5) / 2)
    variables = [f'inpt[{i}]' for i in range(n)]
    statements, sort_outputs = get_sort_statements(variables)
    clique_size = clique_size or n//2

    # count whether there are at least clique_size vertices in the clique
    statements.append("count = " + sort_outputs[clique_size-1])

    # whenever there is not an edge between two vertices, they cannot both be in the clique
    if True:
        statements.append(f"edge_sat = {variables[0]} | ~ {variables[0]}") # this should be initialized to True, but qiskit classical function cannot yet parse True
    else:
        statements.append("edge_sat = True")
    edge_idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge = graph[edge_idx]
            edge_idx += 1
            if edge == '0':
                # TODO: we could reduce depth to log instead of linear by applying AND more efficiently
                # for now, we'll let tweedledum optimize this
                statements.append(f"edge_sat = edge_sat & ~ ({variables[i]} & {variables[j]})")

    statements.append("return count & edge_sat")
    output = f"def verify(inpt: Tuple[bool]) -> bool:\n    "
    output += "\n    ".join(statements)
    return output

class CliqueProblem(BaseProblem):
    """
    Clique Problem Instance

    Args:
        g: Edge representation as binary string (e_12 e_13 ... e_1n e_23 ... e_(n-1)n)
        n: Number of vertices in the graph
        p: Edge probability (integer percentage, optional)
        clique_counts: Precomputed clique counts (optional, will compute if needed)
        instance_id: Database ID (None for unsaved instances)
    """
    __tablename__ = "clique_problems"
    TrialClass = "CliqueTrial"
    graph: Mapped[str] = mapped_column(String, unique=True)
    nodes: Mapped[int]
    edge_probability: Mapped[Optional[int]]
    _clique_counts: Mapped[Optional[List[int]]] = mapped_column(JSON)

    def __init__(self, *args, **kwargs):
        if "clique_counts" in kwargs:
            kwargs["_clique_counts"] = kwargs.pop("clique_counts")
        super().__init__(*args, **kwargs)
        if not self._clique_counts:
            self.compute_clique_counts()

    @property
    def clique_counts(self) -> List[int]:
        """Get clique counts, computing if necessary."""
        if not self._clique_counts:
            self.compute_clique_counts()
        return self._clique_counts

    def compute_clique_counts(self) -> List[int]:
        """Compute the number of vertex subsets that form cliques of at least size k."""
        adjacency_matrix = self.as_adjacency_matrix()
        n = self.nodes
        clique_counts = [0 for _ in range(n + 1)]

        # All subsets are cliques of size 0
        clique_counts[0] = 2**n

        # All single vertices are cliques of size 1
        clique_counts[1] = n

        # Count edges for cliques of size 2
        clique_counts[2] = sum([1 for e in self.graph if e == "1"])

        # Count larger cliques
        for i in range(3, n + 1):
            for clique in itertools.combinations(range(n), i):
                if all(
                    adjacency_matrix[u, v] for u, v in itertools.combinations(clique, 2)
                ):
                    clique_counts[i] += 1

        # Make counts cumulative (at least k vertices in clique)
        for i in range(n - 1, 0, -1):
            clique_counts[i] += clique_counts[i + 1]

        self._clique_counts = clique_counts
        return clique_counts

    def as_adjacency_matrix(self) -> np.ndarray:
        """Convert edge representation to adjacency matrix."""
        adjacency_matrix = np.zeros((self.nodes, self.nodes))
        edge_idx = 0

        for i in range(self.nodes):
            for j in range(i + 1, self.nodes):
                if self.graph[edge_idx] == "1":
                    adjacency_matrix[i, j] = 1
                    adjacency_matrix[j, i] = 1
                edge_idx += 1

        return adjacency_matrix

    def verify_clique(self, vertex_assignment: str, clique_size: int) -> bool:
        """Verify if a vertex assignment represents a valid clique."""
        if len(vertex_assignment) != self.nodes:
            return False

        # Check if enough vertices are selected
        if sum(1 for v in vertex_assignment if v == "1") < clique_size:
            return False

        # Check that selected vertices form a clique
        edge_idx = 0
        for i in range(self.nodes):
            for j in range(i + 1, self.nodes):
                if self.graph[edge_idx] == "0":  # No edge between i and j
                    if vertex_assignment[i] == "1" and vertex_assignment[j] == "1":
                        return False  # Both selected but no edge
                edge_idx += 1

        return True

    def get_verifier_src(self) -> str:
        return construct_clique_verifier(self.graph, clique_size=max(self.nodes//2, 2))

    #### ProblemInstance Methods ####

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "graph": self.graph,
            "nodes": self.nodes,
            "edge_probability": self.edge_probability,
            "clique_counts": self._clique_counts,
        }

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], instance_id: Optional[int] = None
    ) -> "CliqueProblem":
        """Create instance from dictionary."""
        return cls(
            graph=data["graph"],
            nodes=data["nodes"],
            edge_probability=data.get("edge_probability"),
            clique_counts=data.get("clique_counts", []),
            instance_id=instance_id,
        )

    @classproperty
    def problem_type(cls) -> str:
        return "CLIQUE"

    def get_problem_size(self) -> Dict[str, int]:
        """Return key size metrics."""
        num_edges = sum(1 for e in self.graph if e == "1")
        return {
            "num_vertices": self.nodes,
            "num_edges": num_edges,
            "edge_probability": self.edge_probability or 0,
        }

    def number_of_input_bits(self) -> int:
        """Number of input bits for quantum oracle."""
        return self.nodes

    def oracle(self, compile_type: CompileType, **kwargs) -> QuantumCircuit:
        """Generate quantum oracle circuit for clique detection."""
        clique_size = kwargs.get("clique_size")
        if clique_size is None:
            raise ValueError("clique_size must be specified in kwargs")

        edges = self.as_adjacency_matrix().flatten().tolist()
        classical_circuit = QuantumCircuitFunction(
            parameterized_clique_counter_batcher, self.nodes, clique_size, edges
        )

        if compile_type == CompileType.DIRECT:
            raise ValueError("DIRECT not supported")
        elif compile_type == CompileType.CLASSICAL_FUNCTION:
            return classical_circuit.truth_table_synthesis()
        elif compile_type == CompileType.XAG:
            return classical_circuit.synthesize_quantum_circuit()
        else:
            raise NotImplementedError(f"Compile type {compile_type} not implemented")

    def get_number_of_solutions(self, trial: "CliqueTrial") -> int:
        clique_size = trial.clique_size
        if clique_size is None:
            raise ValueError(
                "No clique size for this trial, cannot compute number of solutions"
            )

        return self.clique_counts[clique_size]

class CliqueTrial(TrialCircuitMetricsMixin, BaseTrial):
    """Trial for clique detection using Grover's algorithm."""

    __tablename__ = "clique_trials"
    ProblemClass = CliqueProblem

    grover_iterations: Mapped[Optional[int]]
    clique_size: Mapped[int]

    def calculate_expected_success_rate(
        self,
        db_manager: Optional[BenchmarkDatabase] = None,
    ) -> float:
        """Calculate theoretical expected success rate."""

        grover_iterations = self.grover_iterations or 1
        clique_size = self.clique_size

        if clique_size is None:
            raise ValueError("clique_size not found in trial_params")

        # Get number of solutions (cliques of at least the specified size)
        m = self.problem.clique_counts[clique_size]
        N = 2**self.problem.nodes

        if m == 0:
            return 0.0

        # Grover success probability calculation
        q = (2 * m) / N
        theta = math.atan(math.sqrt(q * (2 - q)) / (1 - q))
        phi = math.atan(math.sqrt(q / (2 - q)))

        return math.sin(grover_iterations * theta + phi) ** 2

    def calculate_success_rate(
        self,
        db_manager: Optional[BenchmarkDatabase] = None,
    ) -> float:
        """Calculate actual success rate from measurement results."""
        if self.counts is None:
            raise ValueError("counts is empty -- cannot compute success rate")

        if self.is_failed:
            return 0.0

        clique_size = self.clique_size
        if clique_size is None:
            raise ValueError("clique_size not found in trial_params")

        # Count successful measurements
        num_valid_cliques = 0
        total_shots = 0

        for measurement, count in self.counts.items():
            if measurement == "-1":  # Failed measurement
                total_shots += count
                continue

            # Reverse bit order to match graph representation
            reversed_measurement = measurement[::-1]

            if self.problem.verify_clique(reversed_measurement, clique_size):
                num_valid_cliques += count

            total_shots += count

        return num_valid_cliques / total_shots if total_shots > 0 else 0.0

class _CliqueProblem(_ProblemInstance):
    def __init__(
        self,
        graph: str,
        nodes: int,
        edge_probability: Optional[int] = None,
        clique_counts: Optional[List[int]] = None,
        instance_id: Optional[int] = None,
    ):
        """
        Clique Problem Instance

        Args:
            g: Edge representation as binary string (e_12 e_13 ... e_1n e_23 ... e_(n-1)n)
            n: Number of vertices in the graph
            p: Edge probability (integer percentage, optional)
            clique_counts: Precomputed clique counts (optional, will compute if needed)
            instance_id: Database ID (None for unsaved instances)
        """
        super().__init__(instance_id)

        self.graph = graph
        self.nodes = nodes
        self.edge_probability = edge_probability
        self._clique_counts = clique_counts or []

        # Validate edge representation
        expected_edges = nodes * (nodes - 1) // 2
        if len(graph) != expected_edges:
            raise ValueError(
                f"Invalid edge representation: expected {expected_edges} edges, got {len(graph)}"
            )

    @property
    def clique_counts(self) -> List[int]:
        """Get clique counts, computing if necessary."""
        if not self._clique_counts:
            self.compute_clique_counts()
        return self._clique_counts

    def compute_clique_counts(self) -> List[int]:
        """Compute the number of vertex subsets that form cliques of at least size k."""
        adjacency_matrix = self.as_adjacency_matrix()
        n = self.nodes
        clique_counts = [0 for _ in range(n + 1)]

        # All subsets are cliques of size 0
        clique_counts[0] = 2**n

        # All single vertices are cliques of size 1
        clique_counts[1] = n

        # Count edges for cliques of size 2
        clique_counts[2] = sum([1 for e in self.graph if e == "1"])

        # Count larger cliques
        for i in range(3, n + 1):
            for clique in itertools.combinations(range(n), i):
                if all(
                    adjacency_matrix[u, v] for u, v in itertools.combinations(clique, 2)
                ):
                    clique_counts[i] += 1

        # Make counts cumulative (at least k vertices in clique)
        for i in range(n - 1, 0, -1):
            clique_counts[i] += clique_counts[i + 1]

        self._clique_counts = clique_counts
        return clique_counts

    def as_adjacency_matrix(self) -> np.ndarray:
        """Convert edge representation to adjacency matrix."""
        adjacency_matrix = np.zeros((self.nodes, self.nodes))
        edge_idx = 0

        for i in range(self.nodes):
            for j in range(i + 1, self.nodes):
                if self.graph[edge_idx] == "1":
                    adjacency_matrix[i, j] = 1
                    adjacency_matrix[j, i] = 1
                edge_idx += 1

        return adjacency_matrix

    def verify_clique(self, vertex_assignment: str, clique_size: int) -> bool:
        """Verify if a vertex assignment represents a valid clique."""
        if len(vertex_assignment) != self.nodes:
            return False

        # Check if enough vertices are selected
        if sum(1 for v in vertex_assignment if v == "1") < clique_size:
            return False

        # Check that selected vertices form a clique
        edge_idx = 0
        for i in range(self.nodes):
            for j in range(i + 1, self.nodes):
                if self.graph[edge_idx] == "0":  # No edge between i and j
                    if vertex_assignment[i] == "1" and vertex_assignment[j] == "1":
                        return False  # Both selected but no edge
                edge_idx += 1

        return True

    #### ProblemInstance Methods ####

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "graph": self.graph,
            "nodes": self.nodes,
            "edge_probability": self.edge_probability,
            "clique_counts": self._clique_counts,
        }

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], instance_id: Optional[int] = None
    ) -> "CliqueProblem":
        """Create instance from dictionary."""
        return cls(
            graph=data["graph"],
            nodes=data["nodes"],
            edge_probability=data.get("edge_probability"),
            clique_counts=data.get("clique_counts", []),
            instance_id=instance_id,
        )

    @property
    def problem_type(self) -> str:
        return "CLIQUE"

    def get_problem_size(self) -> Dict[str, int]:
        """Return key size metrics."""
        num_edges = sum(1 for e in self.graph if e == "1")
        return {
            "num_vertices": self.nodes,
            "num_edges": num_edges,
            "edge_probability": self.edge_probability or 0,
        }

    def number_of_input_bits(self) -> int:
        """Number of input bits for quantum oracle."""
        return self.nodes

    def oracle(self, compile_type: CompileType, **kwargs) -> QuantumCircuit:
        """Generate quantum oracle circuit for clique detection."""
        clique_size = kwargs.get("clique_size")
        if clique_size is None:
            raise ValueError("clique_size must be specified in kwargs")

        edges = self.as_adjacency_matrix().flatten().tolist()
        classical_circuit = QuantumCircuitFunction(
            parameterized_clique_counter_batcher, self.nodes, clique_size, edges
        )

        if compile_type == CompileType.DIRECT:
            raise ValueError("DIRECT not supported")
        elif compile_type == CompileType.CLASSICAL_FUNCTION:
            return classical_circuit.truth_table_synthesis()
        elif compile_type == CompileType.XAG:
            return classical_circuit.synthesize_quantum_circuit()
        else:
            raise NotImplementedError(f"Compile type {compile_type} not implemented")

    def get_number_of_solutions(self, **trial_params) -> int:
        clique_size = trial_params.get("clique_size", None)
        if clique_size is None:
            raise ValueError(
                "No clique size for this trial, cannot compute number of solutions"
            )

        return self.clique_counts[clique_size]


class _CliqueTrial(_BaseTrial):
    """Trial for clique detection using Grover's algorithm."""

    def calculate_expected_success_rate(
        self,
        db_manager: Optional[BenchmarkDatabase] = None,
    ) -> float:
        """Calculate theoretical expected success rate."""
        if self._problem_instance is None:
            if db_manager is None:
                raise ValueError(
                    "Either problem_instance or db_manager must be provided"
                )
            self._problem_instance = self.get_problem_instance(db_manager)

        grover_iterations = self.trial_params.get("grover_iterations", 1)
        clique_size = self.trial_params.get("clique_size")

        if clique_size is None:
            raise ValueError("clique_size not found in trial_params")

        # Get number of solutions (cliques of at least the specified size)
        m = self._problem_instance.clique_counts[clique_size]
        N = 2**self._problem_instance.nodes

        if m == 0:
            return 0.0

        # Grover success probability calculation
        q = (2 * m) / N
        theta = math.atan(math.sqrt(q * (2 - q)) / (1 - q))
        phi = math.atan(math.sqrt(q / (2 - q)))

        return math.sin(grover_iterations * theta + phi) ** 2

    def calculate_success_rate(
        self,
        db_manager: Optional[BenchmarkDatabase] = None,
    ) -> float:
        """Calculate actual success rate from measurement results."""
        if self.counts is None:
            raise ValueError("counts is empty -- cannot compute success rate")

        if self.is_failed:
            return 0.0

        # Load problem instance if needed
        if self._problem_instance is None:
            if db_manager is None:
                raise ValueError(
                    "Either problem_instance or db_manager must be provided"
                )
            self._problem_instance = self.get_problem_instance(db_manager)

        clique_size = self.trial_params.get("clique_size")
        if clique_size is None:
            raise ValueError("clique_size not found in trial_params")

        # Count successful measurements
        num_valid_cliques = 0
        total_shots = 0

        for measurement, count in self.counts.items():
            if measurement == "-1":  # Failed measurement
                total_shots += count
                continue

            # Reverse bit order to match graph representation
            reversed_measurement = measurement[::-1]

            if self._problem_instance.verify_clique(reversed_measurement, clique_size):
                num_valid_cliques += count

            total_shots += count

        return num_valid_cliques / total_shots if total_shots > 0 else 0.0


# Utility functions for creating problem instances


def create_random_graph_instance(
    n: int, p: int, compute_clique_counts: bool = True
) -> CliqueProblem:
    """Create a random graph instance."""
    import random

    num_edges = n * (n - 1) // 2
    g = "".join(["1" if random.random() * 100 < p else "0" for _ in range(num_edges)])

    instance = CliqueProblem(graph=g, nodes=n, edge_probability=p)

    if compute_clique_counts:
        instance.compute_clique_counts()

    return instance


def populate_clique_database(
    db: BenchmarkDatabase,
    n_range: range,
    p_range: List[int],
    graphs_per_config: int = 10,
) -> None:
    """Populate database with random clique problem instances."""
    for n in n_range:
        for p in p_range:
            for _ in range(graphs_per_config):
                instance = create_random_graph_instance(
                    n, p, compute_clique_counts=True
                )
                db.save_problem_instance(instance)

def run_clique_benchmark(db: BenchmarkDatabase, compiler: "SynthesisCompiler", backend: Backend, nodes_iter: Iterable[int], edge_probability_iter: Iterable[int], num_problems: int = 20, shots: int = 10**3,):
    with BatchQueue(db, backend=backend, shots=shots) as q:
        for nodes in nodes_iter:
            for prob in edge_probability_iter:
                for problem in db.find_problem_instances(
                    nodes=nodes,
                    edge_probability=prob,
                    limit=num_problems, 
                    compiler_name=compiler.name,
                    choose_untested=True,
                    random_sample=True
                ):
                    target_clique_size = max(nodes//2, 2)
                    cliques_of_target_size = problem.clique_counts[target_clique_size]
                    if cliques_of_target_size == 0:
                        # clique of size target_clique_size DNE for this graph
                        continue

                    oracle = compiler.compile(problem, clique_size=target_clique_size)

                    optimal_grover_iters = calculate_grover_iterations(cliques_of_target_size, 2**nodes)
                    for grover_iters in range(1, optimal_grover_iters):

                        circuit = build_grover_circuit(oracle, problem.number_of_input_bits(), grover_iters)
                        circuit_transpiled = transpile(circuit, backend=backend)

                        trial = CliqueTrial(
                            problem=problem,
                            compiler_name=compiler.name,
                            grover_iterations=grover_iters,
                            clique_size=target_clique_size,
                            circuit_pretranspile=circuit,
                            circuit=circuit_transpiled,
                        )
                        q.enqueue(trial, circuit_transpiled, run_simulation=(circuit_transpiled.num_qubits <= 12))
    

