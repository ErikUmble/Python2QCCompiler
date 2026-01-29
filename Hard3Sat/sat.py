from benchmarklib import BaseProblem, BaseTrial, BenchmarkDatabase
from typing import Iterator, List, Tuple
from benchmarklib import BaseProblem, CompileType, BaseTrial, BenchmarkDatabase
from benchmarklib.core.types import TrialCircuitMetricsMixin
from benchmarklib.runners.queue import BatchQueue
from benchmarklib.runners.resource_management import run_with_resource_limits
from benchmarklib.pipeline import PipelineCompiler
from benchmarklib.pipeline.config import PipelineConfig
from benchmarklib.algorithms.grover import build_grover_circuit, calculate_grover_iterations,  verify_oracle

from qiskit.providers.backend import Backend
import qiskit
import logging
from typing import Iterator, Iterable, Optional

from sqlalchemy import Column, Integer, String, JSON, ForeignKey, Text, Float, DateTime, LargeBinary, Index, select, func
from sqlalchemy.orm import declarative_base, Mapped, relationship, mapped_column, reconstructor, declared_attr

from tweedledum.bool_function_compiler import circuit_input, QuantumCircuitFunction
from tweedledum import BitVec
from typing import List, Tuple, Dict, Any, Union, Optional, ClassVar, Type
from qiskit import QuantumCircuit
from math import asin, sqrt, sin, atan

logger = logging.getLogger(__name__)
                                                                     
@circuit_input(vars=lambda n: BitVec(n))
def parameterized_3sat(n: int, circuit) -> BitVec(1):
    """ Determines if the given input satisfies the input circuit"""
    s = BitVec(1, 1)  # Start with True 

    # loop over each clause
    for clause in circuit:
        # construct clause result
        expr = BitVec(1, 0)
        for var in clause:
            if var < 0: # negative -> apply not
                expr = expr | ~vars[abs(var)-1]
            else:
                expr = expr | vars[abs(var)-1]

        # apply clause result to final result
        s = s & expr
    return s


class ThreeSat(BaseProblem):
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

    __tablename__ = 'three_sat_problems'
    TrialClass: ClassVar[Type[BaseTrial]] = "ThreeSatTrial"

    num_vars: Mapped[int]
    expr: Mapped[List[List[int]]] = mapped_column(JSON)
    solutions: Mapped[List[List[int]]] = mapped_column(JSON)
    seed: Mapped[int]


    def __init__(self, *args, **kwargs):
        # __init__ is called when creating a new instance
        # but not when loading from the database
        # so use __init__ for input validation
        # but create a reconstructor method for initializing extra class variables
        # that depend on database-loaded values
        super().__init__(*args, **kwargs)

        if not all(len(clause) == 3 for clause in self.expr):
            raise ValueError(
                "All clauses in the expression must have exactly 3 literals."
            )
        
        self.vars: List[int] = sorted(
            list(set(abs(var) for clause in self.expr for var in clause))
        )

    @reconstructor
    def init_on_load(self):
        # __init__ is bypassed when loading from the database, so we initialize extra class variables here
        self.vars: List[int] = sorted(
            list(set(abs(var) for clause in self.expr for var in clause))
        )

    #### ProblemInstance Methods ####

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
            num_vars=data.get("num_vars", 0), # Default num_vars if not present
            solutions=data.get("solutions"),   # This will be List[List[int]]
            seed=data.get("seed"),
        )


    @property
    def problem_type(self) -> str:
        return '3SAT'

    def get_problem_size(self) -> Dict[str, int]:
        # for 3SAT this is the number of variables and clauses
        return {'num_vars': self.num_vars, 'num_clauses': len(self.expr)}
    
    def number_of_input_bits(self) -> int:
        return self.num_vars
    
    def get_verifier_src(self) -> str:
        src = "def verify(inpt: Tuple[bool]) -> bool:\n"
        src += "    return " + " and ".join(
            "(" + " or ".join(
                (f"inpt[{abs(lit)-1}]" if lit > 0 else f"not inpt[{abs(lit)-1}]") 
                for lit in clause
            ) + ")"
            for clause in self.expr
        ) + "\n"
        return src 


class ThreeSatTrial(TrialCircuitMetricsMixin, BaseTrial):
    __tablename__ = 'three_sat_trials'
    ProblemClass = ThreeSat
    grover_iters: Mapped[Optional[int]]

    def calculate_expected_success_rate(
        self,
       db_manager: Optional[BenchmarkDatabase] = None,
    ) -> float:
        grover_iters = self.grover_iters
        if grover_iters is None:
            grover_iters = 1
            print("Warning: grover_iters not found in trial -- defaulting to 1")

        m = len(self.problem.solutions)
        N = 2 ** self.problem.num_vars
        theta = 2*asin(sqrt(m/N))

        # probability calculation using geometric visualization
        return sin((grover_iters + 1/2) * theta) ** 2

    def calculate_success_rate(
        self,
        db_manager: Optional[BenchmarkDatabase] = None,
    ) -> float:
        # count the number of successful measurements in the counts dict
        if self.counts is None:
            return ValueError("counts is empty -- cannot compute the success rate of this trial")

        if self.is_failed:
            return 0.0
    
        # convert the solutions into classical state bitstrings
        solution_states_classical = [['1' if val > 0 else '0' for val in solution] for solution in self.problem.solutions]
        # reverse the bit order of counts to match what we are used to
        reversed_counts = dict()
        for k, v in self.counts.items():
            reversed_counts[k[::-1]] = v

        number_of_measured_solutions = 0
        for solution_state in solution_states_classical:
            number_of_measured_solutions += reversed_counts[solution_state]

        total_measured_states_count = sum(self.counts.values())
        return number_of_measured_solutions / total_measured_states_count 
    
class ThreeSatOracleTrial(TrialCircuitMetricsMixin, BaseTrial):
    """Trial for measuring the output to a specific input to a 3-sat oracle."""

    __tablename__ = "three_sat_oracle_trials"
    ProblemClass = ThreeSat

    input_state: Mapped[str]
    expected_output: Mapped[Optional[bool]]

    def __init__(self, *args, **kwargs):
        if kwargs.get("is_failed", False):
            kwargs["input_state"] = "-1"
        super().__init__(*args, **kwargs)
        if self.expected_output is None:
            self.expected_output = self.problem.verify_solution(self.input_state)
        else:
            self.expected_output = self.expected_output

    def calculate_success_rate(self, *args, **kwargs) -> float:
        """Calculate success rate based on measurement results."""
        if self.is_failed:
            return 0.0

        if self.counts is None:
            raise ValueError("counts is empty -- cannot compute success rate")
        
        total_shots = sum(self.counts.values())
        total_expected_output = (self.input_state + ("1" if self.expected_output else "0"))[::-1]  # reverse bit order for qiskit measurement
        successful_shots = self.counts.get(total_expected_output, 0)

        return successful_shots / total_shots if total_shots > 0 else 0.0
        
            
import random
from pysat.solvers import Glucose3 as Glucose
from pysat.formula import CNF

# generate a random 3SAT and return the expression and list of solutions
def generate_planted_3sat(num_vars, num_clauses, seed: int) -> ThreeSat:
    """
    Generates a planted 3SAT instance.

    Args:
        num_vars (int): The number of Boolean variables.
        num_clauses (int): The target number of 3-literal clauses.
        seed: For reproducability

    Returns:
        ThreeSat: The ProblemInstance for our BenchmarkDatabase
    """
    rng_instance = random.Random(seed)
    
    # 1. Choose a unique solution (planted_solution)
    random_assignment = [rng_instance.randint(0, 1) for _ in range(num_vars)]
    plant = [(i + 1) if bit == 1 else -(i + 1) for (i, bit) in enumerate(random_assignment)]
    planted_solution_set = set(plant)

    formula = CNF()
    current_clauses = 0

    # 2. Generate clauses that are satisfied by the planted_solution
    while current_clauses < num_clauses:
        # Generate a random 3-literal clause
        # Ensure distinct variables in the clause
        variables_in_clause = rng_instance.sample(range(1, num_vars + 1), 3)
        clause = []
        for var in variables_in_clause:
            # Randomly decide polarity (positive or negative literal)
            literal = var if rng_instance.random() < 0.5 else -var
            clause.append(literal)

        # Check if the generated clause is satisfied by the planted_solution
        is_satisfied_by_planted_solution = False
        for literal in clause:
            if literal in planted_solution_set:
                is_satisfied_by_planted_solution = True
                break
            elif -literal not in planted_solution_set:
                is_satisfied_by_planted_solution = True
                break

        if is_satisfied_by_planted_solution:
            formula.append(clause)
            current_clauses += 1

    with Glucose(bootstrap_with=formula) as solver: 
        solutions = [model for model in solver.enum_models()]

    # Build ThreeSat
    return ThreeSat(expr=formula.clauses, num_vars=num_vars, solutions=solutions, seed=seed)

def populate_problems(db: BenchmarkDatabase, var_range: Iterator[int] = range(3, 13), num_problems: int = 100):
    seed = 0
    rng = random.Random(seed)
    for num_vars in var_range:
        # critical_clauses = int(4.26 * num_vars) for random 3 sat (but we are doing planted)
        low = int(3 * num_vars)
        high = int(6 * num_vars)
        count = db.query(select(func.count(ThreeSat.id)).where(ThreeSat.num_vars == num_vars,))[0]
        if count >= num_problems:
            continue
        for i in range(num_problems - count):
            num_clauses = rng.randint(low, high)
            problem_instance = generate_planted_3sat(num_vars, num_clauses, seed)
            db.save_problem_instance(problem_instance)
            seed += 1


def _get_three_sat_trials(problem: ThreeSat, compiler: "PipelineCompiler", config: PipelineConfig) -> Optional[List[ThreeSatTrial]]:
    """
    returns a list of ThreeSatTrial instances for the given problem and compiler
    """
    trials = []
    num_vars = problem.num_vars

    try:
        compile_result = compiler.compile(problem)
        if not compile_result.synthesis_circuit:
            raise Exception("No synthesis circuit returned")
    except Exception as e:
        logger.error(f"Compilation failed for problem ID {problem.id} with error: {e}")
        return None
    oracle = compile_result.synthesis_circuit

    # verify small oracles as a sanity check (but skip large ones which take too long in simulation)
    #if problem.num_vars <= 5 and not verify_oracle(oracle, problem):
    #    logger.warning(f"Oracle verification failed for problem ID {problem.id}, skipping trial.")
    #    return None

    optimal_grover_iters = calculate_grover_iterations(len(problem.solutions), 2**num_vars)
    for grover_iters in range(1, optimal_grover_iters):

        circuit = build_grover_circuit(oracle, problem.number_of_input_bits(), grover_iters)
        circuit_transpiled = compiler.transpile(circuit)

        trial = ThreeSatTrial(
            problem=problem,
            compiler_name=compiler.name,
            grover_iterations=grover_iters,
            pipeline_config=config,
            circuit=circuit_transpiled,
            circuit_pretranspile=circuit,
        )
        trials.append(trial)

    return trials

def _get_three_sat_oracle_trials(problem: ThreeSat, compiler: "PipelineCompiler", config: PipelineConfig, num_trials: int = 5) -> Optional[List[ThreeSatOracleTrial]]:
    """
    returns a list of ThreeSatOracleTrial instances for the given problem and compiler
    """
    trials = []
    n = problem.num_vars

    try:
        compile_result = compiler.compile(problem)
        if not compile_result.synthesis_circuit:
            raise Exception("No synthesis circuit returned")
    except Exception as e:
        logger.error(f"Compilation failed for problem ID {problem.id} with error: {e}")
        return None
    oracle = compile_result.synthesis_circuit

    # verify small oracles as a sanity check (but skip large ones which take too long in simulation)
    #if problem.num_vars <= 5 and not verify_oracle(oracle, problem):
    #    logger.warning(f"Oracle verification failed for problem ID {problem.id}, skipping trial.")
    #    return None

    import random
    for _ in range(num_trials):
        input_state = ''.join(random.choice(['0', '1']) for _ in range(n))
        qc = qiskit.QuantumCircuit(max(oracle.num_qubits, n + 1), n + 1)
        for i, bit in enumerate(input_state):
            if bit == '1':
                qc.x(qc.qubits[i])

        qc.compose(oracle, inplace=True)
        qc.measure(range(n + 1), range(n + 1))

        transpiled_qc = compiler.transpile(qc)
        trials.append(ThreeSatOracleTrial(
                problem=problem,
                compiler_name=compiler.name,
                input_state=input_state,
                pipeline_config=config,
                circuit = transpiled_qc,
                circuit_pretranspile = qc,
            )
        )

    return trials

def run_three_sat_benchmark(db: BenchmarkDatabase, compiler: "PipelineCompiler", backend: Backend, num_vars_iter: Iterable[int], num_problems: int = 20, shots: int = 10**3, max_problems_per_job: Optional[int] = None, save_circuits: bool = False):
    # get compiler pipeline config to save with each trial
    config = db.get_saved_config(compiler.config)

    assert backend.name == config.backend.name

    with BatchQueue(db, backend=backend, shots=shots) as q:
        for n in num_vars_iter:
                
            count = db.query(
                select(func.count(func.distinct(db.problem_class.id)))
                .select_from(db.problem_class)
                        .join(db.trial_class, db.trial_class.problem_id == db.problem_class.id)
                        .where(
                            db.trial_class.pipeline_config == config,
                            db.problem_class.num_vars == n,
                    )
                )[0]
            if count >= num_problems:
                logger.info(f"Skipping (num_vars={n}) -- already have {count} instances")
                continue

            pending_trial_problem_count = 0

            for problem in db.find_problem_instances(
                num_vars=n,
                limit=num_problems - count, 
                compiler_name=compiler.name,
                choose_untested=True,
                random_sample=True
            ):
                logger.info(f"Compiling problem ID {problem.id} with {n} num_vars.")
                compiler_run = run_with_resource_limits(
                    _get_three_sat_trials if db.trial_class == ThreeSatTrial else _get_three_sat_oracle_trials,
                    kwargs={
                        "problem": problem,
                        "compiler": compiler,
                        "config": config,
                    },
                    memory_limit_mb=2024,
                    timeout_seconds=240
                )
                trials = compiler_run.result if compiler_run.success else None
                if trials is None:
                    logger.warning(f"Compilation failed for problem ID {problem.id}: {compiler_run.error_message}.")
                    db.create_compilation_failure(problem, compiler.name)
                    continue
                
                for trial in trials:
                        qc = trial.circuit_pretranspile
                        transpiled_qc = trial.circuit
                        if not save_circuits:
                            trial.circuit = None
                            trial.circuit_pretranspile = None
                        q.enqueue(trial, transpiled_qc, run_simulation=(qc.num_qubits <= 10))

                pending_trial_problem_count += 1
                if max_problems_per_job and pending_trial_problem_count >= max_problems_per_job:
                    q.submit_tasks()
                    pending_trial_problem_count = 0