# Import te research library
from math import asin, sin, sqrt
from typing import Any, Dict, List, Optional

from qiskit import QuantumCircuit
from tweedledum import BitVec
from tweedledum.bool_function_compiler import QuantumCircuitFunction, circuit_input

from benchmarklib import BaseTrial, BenchmarkDatabase, CompileType, ProblemInstance


# verifier for propagated CNF
@circuit_input(vars=lambda n: BitVec(n))
def unit_propped_sat(n: int, circuits) -> BitVec(1):
    """Takes a list of CNFs and joins them with an OR and checks for satisfiability"""
    s = BitVec(1, 0)  # Start with True

    # loop over each clause
    for circuit in circuits:
        circuit_output = BitVec(1, 1)
        for clause in circuit:
            # construct clause result
            expr = BitVec(1, 0)
            for var in clause:
                if var < 0:  # negative -> apply not
                    expr = expr | ~vars[abs(var) - 1]
                else:
                    expr = expr | vars[abs(var) - 1]

            # apply clause result to final result
            circuit_output = circuit_output & expr
        s = s | circuit_output

    return s


@circuit_input(vars=lambda n: BitVec(n))
def parameterized_3sat(n: int, circuit) -> BitVec(1):
    """Determines if the given input satisfies the input circuit"""
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


class ThreeSat(ProblemInstance):
    def __init__(
        self,
        expr: List[List[int]],
        num_vars: int,
        solutions: [List[List[int]]],
        seed: int,
        instance_id: Optional[int] = None,
    ):
        """
        3SAT Circuit Representation -- List of Clauses List[List[int, int, int]]
        the absolute value is the index, the negation means apply not
        indices start at 1
        [(1, 2, 3), (-4, 1, 2)] = (x1 V x2 V x3) ^ (~x4 V x1 V x2)

        Args:
            expr: The 3SAT expression as a list of 3-literal clauses.
            num_vars: The highest index of a varaible in the expression.
                NOTE: this may be larger than the number of unique varaibles in the expression
            solutions: Pre-computed list of satisfying assignments (as lists of integers).
            seed: seed used for random generation function
        """
        super().__init__()

        if not all(len(clause) == 3 for clause in expr):
            raise ValueError(
                "All clauses in the expression must have exactly 3 literals."
            )

        # flatten expression and collect the unique variables into a sorted list
        # Sorting ensures a consistent order for solutions
        self.num_vars = num_vars
        self.vars: List[int] = sorted(
            list(set(abs(var) for clause in expr for var in clause))
        )
        self.expr: List[List[int]] = expr
        self.solutions: List[List[int]] = solutions
        self.seed = seed
        self.instance_id = instance_id

    # ProblemInstance Methods ####

    def to_dict(self) -> Dict[str, Any]:
        """Converts the ThreeSat instance to a JSON-serializable dictionary."""
        return {
            "expr": self.expr,
            "num_vars": self.num_vars,
            "solutions": self.solutions,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], instance_id: Optional[int]) -> "ThreeSat":
        """Creates a ThreeSat instance from a dictionary."""
        return cls(
            expr=data.get("expr"),
            num_vars=data.get("num_vars", 0),  # Default num_vars if not present
            solutions=data.get("solutions"),  # This will be List[List[int]]
            seed=data.get("seed"),
            instance_id=instance_id,
        )

    @property
    def problem_type(self) -> str:
        return "3SAT"

    def get_problem_size(self) -> Dict[str, int]:
        # for 3SAT this is the number of variables and clauses
        return {"num_vars": self.num_vars, "num_clauses": len(self.expr)}

    def number_of_input_bits(self) -> int:
        return self.num_vars

    def oracle(self, compile_type: CompileType, **kwargs) -> QuantumCircuit:
        # builds a classical representation of THIS particular 3SAT problem
        classical_circuit = QuantumCircuitFunction(
            parameterized_3sat, self.num_vars, self.expr
        )
        if compile_type is CompileType.XAG:
            return classical_circuit.synthesize_quantum_circuit()
        else:
            return classical_circuit.truth_table_synthesis()

    def get_number_of_solutions(self, **trial_params) -> int:
        return len(self.solutions)


class ThreeSatTrial(BaseTrial):
    def calculate_expected_success_rate(
        self,
        db_manager: Optional[BenchmarkDatabase] = None,
    ) -> float:
        # default to one grover iteration if there is no key in the metadata
        grover_iters = self.trial_params.get("grover_iters", 1)
        m = len(self._problem_instance.solutions)
        N = 2**self._problem_instance.num_vars
        theta = 2 * asin(sqrt(m / N))

        # probability calculation using geometric visualization
        return sin((grover_iters + 1 / 2) * theta) ** 2

    def calculate_success_rate(
        self,
        db_manager: Optional[BenchmarkDatabase] = None,
    ) -> float:
        # count the number of successful measurements in the counts dict
        if self.counts is None:
            return ValueError(
                "counts is empty -- cannot compute the success rate of this trial"
            )

        if self.is_failed:
            return 0.0

        # Load problem instance if not provided
        if self._problem_instance is None:
            if db_manager is None:
                raise ValueError(
                    "Either problem_instance or db_manager must be provided"
                )
            self.get_problem_instance(db_manager)

        # convert the solutions into classical state bitstrings
        solution_states_classical = [
            [1 if val > 0 else 0 for val in solution]
            for solution in self._problem_instance.solutions
        ]
        solution_states_classical = [
            "".join(str(bit) for bit in solution)
            for solution in solution_states_classical
        ]
        # reverse the bit order of counts to match what we are used to
        reversed_counts = dict()
        for k, v in self.counts.items():
            reversed_counts[k[::-1]] = v

        number_of_measured_solutions = 0
        for solution_state in solution_states_classical:
            number_of_measured_solutions += reversed_counts.get(solution_state, 0)

        total_measured_states_count = sum(self.counts.values())
        return number_of_measured_solutions / total_measured_states_count
