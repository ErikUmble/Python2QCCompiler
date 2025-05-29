from typing import List, Tuple

import tweedledum as td
from qiskit import QuantumCircuit
from tweedledum import BitVec
from tweedledum.bool_function_compiler.decorators import circuit_input
from tweedledum.bool_function_compiler.function_parser import FunctionParser
from tweedledum.classical import optimize
from tweedledum.passes import linear_resynth, parity_decomp
from tweedledum.synthesis import xag_cleanup, xag_synth

from sat_database import ThreeSat

td.bool_function_compiler.setup_logging()

# 3SAT Circuit Representation -- List of Clauses List[Tuple[int, int, int]]
# the absolute value is the index, the negation means apply not
# indices start at 1
# [(1, 2, 3), (-4, 1, 2)] = (x1 V x2 V x3) ^ (~x4 V x1 V x2)


@circuit_input(vars=lambda n: BitVec(n))
def parameterized_3sat(n: int, circuit) -> BitVec(1):
    """Counts cliques of size 2 in a graph specified by the edge list."""
    s = BitVec(1, 1)  # Start with True

    # loop over each clause
    for clause in circuit:
        # construct clause result
        expr = BitVec(1, 0)
        for var in clause:
            if var < 0:  # negative -> apply not
                expr = expr | ~vars[abs(var) - 1]
            else:
                expr = expr | vars[abs(var) - 1]

        # apply clause result to final result
        s = s & expr
    return s


def oracle_from_3sat(
    sat: List[Tuple[int, int, int]], num_vars: int, output_dot=False, optimize_xag=True
) -> QuantumCircuit:
    # generate classical function source
    src, _ = parameterized_3sat(num_nodes, sat)
    print(f"SOURCE: \n{src}")
    parsed_function = FunctionParser(src.strip())
    xag = parsed_function._logic_network

    # XAG operations
    xag = xag_cleanup(xag)
    if output_dot:
        td.utils.xag_export_dot(xag, "initial_xag.dot")

    if optimize_xag:
        optimize(xag)

    # write optimized xag to DOT format
    if output_dot:
        td.utils.xag_export_dot(xag, "optimized_xag.dot")

    circ = xag_synth(xag)

    # Circuit Optimization Passes
    circ = parity_decomp(circ)
    circ = linear_resynth(circ)
    return td.converters.to_qiskit(circ, "gatelist")


import random
from typing import List, Optional, Tuple


def generate_random_3sat(
    num_vars: int, num_clauses: int, seed: Optional[int] = None
) -> List[Tuple[int, int, int]]:
    """
    Generates a random 3-SAT formula (list of clauses) with reproducibility.

    Args:
        num_vars: The number of variables (n). Variables will be numbered 1 to n.
        num_clauses: The number of clauses (m).
        seed: An optional integer seed for the random number generator.
              Providing the same seed guarantees the same formula output
              for the same num_vars and num_clauses. If None, the generator
              is initialized randomly (likely based on system time).

    Returns:
        A list of clauses. Each clause is a tuple of 3 integers.
        Positive integer 'i' represents variable x_i.
        Negative integer '-i' represents the negation of variable x_i (¬x_i).
        Returns an empty list if num_vars < 3.

    Raises:
        ValueError: If num_vars < 3, as we cannot pick 3 distinct variables.
    """
    if num_vars < 3:
        raise ValueError(
            "Number of variables (num_vars) must be at least 3 "
            "to form 3-literal clauses."
        )
    if num_clauses <= 0:
        return []

    # --- Create a dedicated RNG instance ---
    # If seed is None, Random() initializes itself based on system sources.
    rng = random.Random(seed)
    # --- End RNG instance creation ---

    variables = list(range(1, num_vars + 1))
    clauses: List[Tuple[int, int, int]] = []

    for _ in range(num_clauses):
        while True:  # Loop until a valid clause is generated (optional for uniqueness)
            # 1. Choose 3 distinct variables using the dedicated RNG
            # Use rng.sample instead of random.sample
            chosen_vars = rng.sample(variables, 3)

            # 2. Assign random polarity (negated or not) to each using the dedicated RNG
            # Use rng.choice instead of random.choice
            clause_list = []
            for var in chosen_vars:
                if rng.choice([True, False]):
                    clause_list.append(var)  # Positive literal
                else:
                    clause_list.append(-var)  # Negative literal

            # Ensure the clause is sorted internally for consistent checking
            clause = tuple(
                sorted(clause_list, key=abs)
            )  # Sort by absolute value for consistency

            # --- Optional: Check for uniqueness if required ---
            # if clause not in unique_clauses_set:
            #     unique_clauses_set.add(clause)
            #     clauses.append(clause)
            #     break # Exit while loop, clause accepted
            # else:
            #     # Clause already exists, loop again to generate a new one
            #     pass
            # --- End Optional ---

            # --- If not checking uniqueness, just add and break ---
            clauses.append(clause)
            break

    return clauses


def general_3sat_circuit(sat: ThreeSat):
    """
    uses construction from class
    z = (x1 V ~x2 V ~ x3)  is equivalent to ~(~x1 ^ x2 ^ x3 )

    We negate each clause and use a multiply-controlled-not on an ancillae to track the
    result of this clause Then we use another multiply controlled not on the target qubit
    """

    num_vars = sat.num_vars
    num_clauses = sat.num_clauses
    num_qubits = num_vars + num_clauses + 1
    # create Quantum Circuit
    oracle = QuantumCircuit(num_qubits)

    # For each clause
    #     set the state of each qubit for an mcx by creating a circuit
    #     apply mcx
    #     use circuit.inverse() to uncompute qubits leaving ancillae in computed state

    for ancillae_offset, clause in enumerate(sat.expr):
        qubit_indices = [abs(var) - 1 for var in clause]
        clause_circuit = QuantumCircuit(num_qubits)

        negated_clause = [-var for var in clause]

        for qubit_idx, var in zip(qubit_indices, negated_clause):
            if var < 0:
                clause_circuit.x(qubit_idx)

        oracle.compose(clause_circuit, list(range(num_qubits)), inplace=True)
        oracle.mcx(qubit_indices, num_vars + ancillae_offset)
        oracle.x(num_vars + ancillae_offset)
        oracle.compose(clause_circuit.inverse(), list(range(num_qubits)), inplace=True)
        oracle.barrier(list(range(num_qubits)), label=f"Clause {ancillae_offset + 1}")

    # final mcx from all ancialle to target
    ancillae_idxs = list(range(num_vars, num_vars + num_clauses))
    print(ancillae_idxs)
    oracle.mcx(ancillae_idxs, num_qubits - 1)

    return oracle
