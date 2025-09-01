from tweedledum.bool_function_compiler.decorators import circuit_input
from tweedledum.bool_function_compiler.function_parser import FunctionParser
from tweedledum.synthesis import xag_synth, xag_cleanup
from tweedledum.classical import optimize
from tweedledum.passes import parity_decomp, linear_resynth
from tweedledum import BitVec
import tweedledum as td
from qiskit import QuantumCircuit
import networkx as nx


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
    """Counts cliques of size 2 in a graph specified by the edge list."""
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


@circuit_input(vertices=lambda n: BitVec(n))
def parameterized_clique_counter(n: int, k: int, edges) -> BitVec(1):
    """Counts cliques of size 2 in a graph specified by the edge list."""
    s = BitVec(1, 1)  # Start with True (assuming a clique)

    # Check if any non-connected vertices are both selected
    for i in range(n):
        for j in range(i + 1, n):
            # If vertices i and j are both selected (=1) AND there's no edge between them (=0)
            # then it's not a clique
            if edges[i * n + j] == 0:
                s = s & ~(vertices[i] & vertices[j])

    generate_sorting_network(vertices, n, k)

    return s & sorted_bit_0


def direct_clique_to_cnf(n: int, k: int, edges: list) -> tuple:
    """
    Direct encoding of k-clique to CNF without AST transformation.
    
    Args:
        n: Number of vertices
        k: Clique size
        edges: List of (i,j) tuples representing edges
        
    Returns:
        clauses: List[List[int]] - CNF clauses
        var_info: Dict with variable mapping and metadata
    """
    clauses = []
    var_info = {
        'input_vars': list(range(1, n+1)),  # Variables 1 to n are vertex selections
        'aux_vars': [],
        'total_vars': n,
        'mapping': {f'v_{i}': i+1 for i in range(n)}
    }
    
    # 1. Non-edge constraints: For each missing edge, at most one vertex can be selected
    for i in range(n):
        for j in range(i+1, n):
            if edges[i * n + j] == 0:
                # NOT (v_i AND v_j) = NOT v_i OR NOT v_j
                clauses.append([-(i+1), -(j+1)])

    dimacs_str = f'p cnf+ {n} {len(clauses)+1}\n'
    for clause in clauses:
        clause_str = ' '.join(str(var) for var in clause).strip()
        dimacs_str += clause_str + ' 0\n'

    cardinality_condition = ' '.join(str(var+1) for var in range(n)).strip()
    cardinality_condition += f' >= {k}'
    dimacs_str += cardinality_condition
    
    return dimacs_str
