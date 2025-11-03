import logging
from typing import Iterable, List, Tuple, Dict, Any, Union, Optional, ClassVar, Type
import qiskit
from qiskit.providers import Backend
from qiskit import QuantumCircuit, transpile
import random

from sqlalchemy import Column, Integer, String, JSON, ForeignKey, Text, Float, DateTime, LargeBinary, Index, select, func
from sqlalchemy.orm import declarative_base, Mapped, relationship, mapped_column, reconstructor, declared_attr

from tweedledum.bool_function_compiler import circuit_input, QuantumCircuitFunction
from tweedledum import BitVec

from benchmarklib import BaseProblem, CompileType, BaseTrial, BenchmarkDatabase, TrialCircuitMetricsMixin
from benchmarklib import BatchQueue
from benchmarklib.compilers import SynthesisCompiler

# temporary during db migration
from benchmarklib.core.types import _ProblemInstance, _BaseTrial


logger = logging.getLogger("RandomBooleanFunctions.rbf")

operators = ("and", "or", "not", "^")

def get_variables(num_vars):
    return ["x" + str(i) for i in range(num_vars)]

def get_random_statement(num_vars, complexity, seed=None):
    if seed:
        random.seed(seed)

    if num_vars < 2:
        raise ValueError("num_vars must be at least 2")

    if complexity < 1:
        raise ValueError("complexity must be at least 1")

    variables = get_variables(num_vars)
    statements = list(variables)

    for i in range(complexity):
        next_complexity_statements = []

        for _ in range(num_vars):

            operator = random.choice(operators)
            s1 = random.choice(statements[-num_vars:])  # ensure s1 has complexity i
            s2 = random.choice(statements)              # s2 can have any complexity <= i
        
            if operator == "not":
                next_complexity_statements.append(f"not ({s1})")
            else:
                next_complexity_statements.append(f"({s1}) {operator} ({s2})")

        statements += next_complexity_statements

    return statements[-1], variables
                                                                     

def hamming_distance(a, b):
    """
    returns the hamming distance between two binary strings
    the strings must be the same length
    """
    assert len(a) == len(b)
    return sum([1 for i in range(len(a)) if a[i] != b[i]])

def get_circuit_input_function(statement, variables, name="f"):
    """
    Returns a Python function definition as a string, which implements the given boolean statement of the provided variables
    """
    # convert variables to BitVec indices
    var_map = {var: f"vars[{i}]" for i, var in enumerate(variables)}

    # replace variables with their BitVec index notation
    # start with longest variable names to avoid partial replacements
    for var in sorted(variables, key=len, reverse=True):
        statement = statement.replace(var, var_map[var])

    return f"""
@circuit_input(vars=BitVec({len(variables)}))
def {name}() -> BitVec(1):
    return {statement}
    """

class RandomBooleanFunction(BaseProblem):
    __tablename__ = "random_boolean_functions"
    TrialClass: ClassVar[Type[BaseTrial]] = "RandomBooleanFunctionTrial"

    num_vars: Mapped[int]
    complexity: Mapped[int]
    statement: Mapped[str] = mapped_column(Text)

    @declared_attr
    def __table_args__(cls):
        base_args = super().__table_args__ if hasattr(super(), '__table_args__') else ()

        return base_args + (
            Index('ix_num_vars_complexity', 'num_vars', 'complexity'),
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vars = get_variables(self.num_vars)

    @reconstructor
    def init_on_load(self):
        # __init__ is bypassed when loading from the database, so we initialize extra class variables here
        self.vars = get_variables(self.num_vars)

    def to_dict(self) -> Dict[str, Any]:
        """Converts the RandomBooleanFunction instance to a JSON-serializable dictionary."""
        return {
            "statement": self.statement,
            "num_vars": self.num_vars,
            "complexity": self.complexity,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], instance_id: Optional[int]) -> "RandomBooleanFunction":
        """Creates a RandomBooleanFunction instance from a dictionary."""
        return cls(
            statement=data.get("statement"),
            num_vars=data.get("num_vars"),
            complexity=data.get("complexity"),
            instance_id=instance_id,
        )

    def get_problem_size(self) -> Dict[str, int]:
        return {'num_vars': self.num_vars, 'complexity': self.complexity}
    
    def number_of_input_bits(self) -> int:
        return self.num_vars

    def verify_solution(self, inpt: BitVec) -> bool:
        """
        Checks whether or not inpt satisfies this problem
        Returns:
            True if inpt satisfies this problem, False otherwise
        """
        if len(inpt) != self.num_vars:
            raise ValueError("inpt size should match num_vars")

        input_variables = {}
        for i, var in enumerate(self.vars):
            input_variables[var] = int(inpt[i])
        return eval(self.statement, {}, input_variables)
    
    def get_verifier_src(self) -> str:
        import re
    
        # Find all variable references (x followed by digits)
        def replace_var(match):
            var_num = int(match.group(1))
            return f"inpt[{var_num - 1}]"
    
        # Replace x1, x2, etc. with inpt[0], inpt[1], etc.
        transformed = re.sub(r'x(\d+)', replace_var, self.statement)
        
        # Create the function body
        func_code = f"def verify(inpt: Tuple[bool]) -> bool:\n    return {transformed}"
        return func_code

    def oracle(self, compile_type: CompileType) -> QuantumCircuit:
        import tempfile
        import importlib.util
        import os

        temp_function = get_circuit_input_function(self.statement, self.vars, name="temporary_function")
        required_imports = """
from tweedledum import BitVec
from tweedledum.bool_function_compiler import circuit_input
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            module_name = "temp_boolean_func"
            file_path = os.path.join(temp_dir, f"{module_name}.py")

            with open(file_path, "w") as f:
                f.write(required_imports)
                f.write(temp_function)

            spec = importlib.util.spec_from_file_location(module_name, file_path)
            temp_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(temp_module)

            classical_function = QuantumCircuitFunction(temp_module.temporary_function)
        
        if compile_type == CompileType.CLASSICAL_FUNCTION:
            return classical_function.truth_table_synthesis()
        
        elif compile_type == CompileType.XAG:
            return classical_function.synthesize_quantum_circuit(remove_unused_inputs=False)
        
        raise ValueError(f"{compile_type} not yet supported for RandomBooleanFunction")
    

class RandomBooleanFunctionTrial(TrialCircuitMetricsMixin, BaseTrial):
    __tablename__ = "random_boolean_function_trials"
    ProblemClass = RandomBooleanFunction

    input_state: Mapped[str] = mapped_column(String(255))  # can make larger as quantum computers scale up

    @property
    def expected_result(self):

        if self.input_state == "-1":
            return None
        
        input_variables = {}
        problem = self.problem
        for i, var in enumerate(problem.vars):
            input_variables[var] = int(self.input_state[i])
        return eval(problem.statement, {}, input_variables)

    @property
    def exact_match_rate(self):
        if self.is_failed:
            return 0.0
        
        if self.counts is None:
            raise ValueError("Counts must be set before calculating exact match rate. Use get_counts() or get_counts_async() to update the counts.")
        
        successes = self.counts.get(self.total_expected_results(), 0)
        return successes / sum(self.counts.values())
    
    @property
    def mean_hamming_distance(self):
        """
        the mean hamming distance between the expected result and the measured results per shot, per qubit
        """

        if self.is_failed:
            return float(self._problem_instance.num_vars + 1)  # max hamming distance
    
        if self.counts is None:
            raise ValueError("Counts must be set before calculating mean hamming distance. Use get_counts() or get_counts_async() to update the counts.")
        
        total_distance = 0
        expected = self.total_expected_results()
        for result, count in self.counts.items():
            total_distance += hamming_distance(expected, result) * count

        return total_distance / sum(self.counts.values())

    @property
    def result_qubit_success_rate(self):
        """
        the rate of success for the result qubit of Uf
        """
        if self.is_failed:
            return 0.0
        
        if self.counts is None:
            raise ValueError("Counts must be set before calculating correct result qubit rate. Use get_counts() or get_counts_async() to update the counts.")
        
        correct_result_count = 0
        expected = "1" if self.expected_result else "0"
        result_idx = self.problem.num_vars # result qubit is the qubit which immediately follows the input qubits
        for result, count in self.counts.items():
            if result[result_idx] == expected:
                correct_result_count += count

        return correct_result_count / sum(self.counts.values())

    def calculate_expected_success_rate(
            self,
            db_manager: Optional[BenchmarkDatabase] = None,
    ):
        # expected success rate is 1, as this kind of trial tests
        # the accuracy of pure state (expected) output given pure state input
        return 1
    
    def total_expected_results(self):
        """ 
        Returns the expected binary string to be measured after the circuit is run
        Note that due to little-endian encoding of the counts, the first bit is the result bit
        and the input bits are in reverse order
        """
        result_bit = "1" if self.expected_result else "0"
        return result_bit + self.input_state[::-1]
    
    def calculate_success_rate(
        self,
        db_manager: Optional[BenchmarkDatabase] = None,
    ) -> float:
        # using exact match rate across all qubits as the success metric
        # note that although the compiler may introduce additional ancilla qubits,
        # we only measure the input and output qubits, so the ancilla qubits do not affect 
        # the expected counts or success rate
        return self.exact_match_rate
    

class _RandomBooleanFunction(_ProblemInstance):
    def __init__(
        self,
        statement: str,
        num_vars: int,
        complexity: int,
        instance_id: Optional[int] = None,
    ):
        """
        RandomBooleanFunction Representation -- string such as "x1 & (x2 | ~x3)"
        
        Args:
            statement: The boolean statement.
            num_vars: The highest index of a varaible in the expression. 
                NOTE: this may be larger than the number of unique varaibles in the expression
        """
        super().__init__(instance_id=instance_id)

        # flatten expression and collect the unique variables into a sorted list
        # Sorting ensures a consistent order for solutions
        self.num_vars = num_vars
        self.complexity = complexity
        self.vars = get_variables(num_vars)
        self.statement: str = statement

    #### ProblemInstance Methods ####

    def to_dict(self) -> Dict[str, Any]:
        """Converts the RandomBooleanFunction instance to a JSON-serializable dictionary."""
        return {
            "statement": self.statement,
            "num_vars": self.num_vars,
            "complexity": self.complexity,
        }


    @classmethod
    def from_dict(cls, data: Dict[str, Any], instance_id: Optional[int]) -> "RandomBooleanFunction":
        """Creates a RandomBooleanFunction instance from a dictionary."""
        return cls(
            statement=data.get("statement"),
            num_vars=data.get("num_vars"),
            complexity=data.get("complexity"),
            instance_id=instance_id,
        )


    @property
    def problem_type(self) -> str:
        return 'RandomBooleanFunction'

    def get_problem_size(self) -> Dict[str, int]:
        return {'num_vars': self.num_vars, 'complexity': self.complexity}
    
    def number_of_input_bits(self) -> int:
        return self.num_vars

    def verify(self, inpt: BitVec) -> bool:
        """
        Checks whether or not inpt satisfies this problem
        Returns:
            True if inpt satisfies this problem, False otherwise
        """
        if len(inpt) != self.num_vars:
            raise ValueError("inpt size should match num_vars")

        input_variables = {}
        for i, var in enumerate(self.vars):
            input_variables[var] = int(inpt[i])
        return eval(self.statement, {}, input_variables)

    def oracle(self, compile_type: CompileType) -> QuantumCircuit:
        import tempfile
        import importlib.util
        import os

        temp_function = get_circuit_input_function(self.statement, self.vars, name="temporary_function")
        required_imports = """
from tweedledum import BitVec
from tweedledum.bool_function_compiler import circuit_input
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            module_name = "temp_boolean_func"
            file_path = os.path.join(temp_dir, f"{module_name}.py")

            with open(file_path, "w") as f:
                f.write(required_imports)
                f.write(temp_function)

            spec = importlib.util.spec_from_file_location(module_name, file_path)
            temp_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(temp_module)

            classical_function = QuantumCircuitFunction(temp_module.temporary_function)
        
        if compile_type == CompileType.CLASSICAL_FUNCTION:
            return classical_function.truth_table_synthesis()
        
        elif compile_type == CompileType.XAG:
            return classical_function.synthesize_quantum_circuit(remove_unused_inputs=False)
        
        raise ValueError(f"{compile_type} not yet supported for RandomBooleanFunction")
    

def create_compilation_failure(db_manager: BenchmarkDatabase, problem: RandomBooleanFunction, compiler_name: str):
    trial = RandomBooleanFunctionTrial(
        problem=problem,
        compiler_name=compiler_name,
        is_failed=True,
        input_state=""
    )
    db_manager.save_trial(trial)


def run_rbf_benchmark(db_manager: BenchmarkDatabase, compiler: "SynthesisCompiler", backend: Backend, num_vars_iter: Iterable[int], complexity_iter: Iterable[int], num_functions: int = 10, trials_per_instance: int = 5, shots: int = 10**3, max_problems_per_job: Optional[int] = None, save_circuits: Optional[bool] = True):
    """
    Run benchmarks for random boolean functions with varying number of variables and complexity.

    Args:
        db_manager: BenchmarkDatabase instance for storing results
        num_vars_iter: num_vars range
        complexity_iter: complexity range
        num_functions: Number of random functions to create trials for from each (num_vars, complexity) pair
        trials_per_instance: Number of trials (random input states) to run per problem instance (rbf function)
        shots: Number of shots to run per trial
        max_problems_per_job: Maximum number of problems to include in each job submission to the backend (set this to 1 to force trial transpilation and submission to the backend before compiling the next problem)
    """
    with BatchQueue(db_manager, backend=backend, shots=shots) as q:
        pending_trial_problem_count = 0
        for num_vars in num_vars_iter:
            for complexity in complexity_iter:
                logger.info(f"Benchmarking functions with num_vars={num_vars}, complexity={complexity}")
                # check if we already have enough data for this size
                count = db_manager.query(
                    select(func.count(func.distinct(db_manager.problem_class.id)))
                    .select_from(db_manager.problem_class)
                            .join(db_manager.trial_class, db_manager.trial_class.problem_id == db_manager.problem_class.id)
                            .where(
                                db_manager.trial_class.compiler_name == compiler.name,
                                db_manager.problem_class.num_vars == num_vars,
                                db_manager.problem_class.complexity == complexity
                        )
                    )[0]
                if count >= num_functions:
                    logger.info(f"Skipping (num_vars={num_vars}, complexity={complexity}) -- already have {count} instances")
                    continue

                problem_instances = db_manager.find_problem_instances(
                    num_vars=num_vars,
                    complexity=complexity,
                    choose_untested=True,
                    compiler_name=compiler.name,
                    random_sample=True,
                    limit=num_functions-count
                )
                for problem_instance in problem_instances:
                    logger.info(f"Benchmarking with problem {problem_instance.id}: {problem_instance.statement}")
                    oracle = problem_instance.oracle(compiler.name)  # TODO: should update this to use new compiler API
                    if oracle is None:
                        logger.warning(f"Compilation failed for problem {problem_instance.id}: {problem_instance.statement} with compiler {compiler.name}")
                        create_compilation_failure(db_manager, problem_instance, compiler.name)
                        continue
                    for _ in range(trials_per_instance):
                        input_state = ''.join(random.choice('01') for _ in range(num_vars))
                        qc = qiskit.QuantumCircuit(max(oracle.num_qubits, num_vars + 1), num_vars + 1)
                        for i, bit in enumerate(input_state):
                            if bit == '1':
                                qc.x(qc.qubits[i])

                        qc.compose(oracle, inplace=True)
                        qc.measure(range(num_vars + 1), range(num_vars + 1))
                        transpiled_qc = transpile(qc, backend=backend)

                        trial = RandomBooleanFunctionTrial(
                            problem=problem_instance,
                            compiler_name=compiler.name,
                            input_state=input_state,
                            circuit = transpiled_qc if save_circuits else None,
                            circuit_pretranspile = qc if save_circuits else None,
                        )
                        
                        q.enqueue(trial, transpiled_qc, run_simulation=(qc.num_qubits <= 10))

                    pending_trial_problem_count += 1
                    if max_problems_per_job and pending_trial_problem_count >= max_problems_per_job:
                        q.submit_tasks()
                        pending_trial_problem_count = 0


def generate_rbf_problems(db_manager: BenchmarkDatabase, num_vars_iter: Iterable[int], complexity_iter: Iterable[int], num_functions: int = 10):
    """
    Generate and store random boolean function problems in the database.

    Args:
        db_manager: BenchmarkDatabase instance for storing results
        num_vars_iter: num_vars range
        complexity_iter: complexity range
        num_functions: Number of random functions to create from each (num_vars, complexity) pair
    """
    for num_vars in num_vars_iter:
        for complexity in complexity_iter:
            logger.info(f"Generating functions with num_vars={num_vars}, complexity={complexity}")
            existing_count = db_manager.query(
                select(func.count(db_manager.problem_class.id))
                .where(
                    db_manager.problem_class.num_vars == num_vars,
                    db_manager.problem_class.complexity == complexity
                )
            )[0]
            if existing_count >= num_functions:
                logger.info(f"Skipping (num_vars={num_vars}, complexity={complexity}) -- already have {existing_count} instances")
                continue

            statements = set()
            while len(statements) < (num_functions - existing_count):
                statement, _ = get_random_statement(num_vars, complexity)
                statements.add(statement)

            for statement in statements:
                problem_instance = RandomBooleanFunction(statement=statement, num_vars=num_vars, complexity=complexity)
                db_manager.save_problem_instance(problem_instance)


class _RandomBooleanFunctionTrial(_BaseTrial):

    @property
    def input_state(self) -> str:
        if not self.trial_params.get('input_state'):
            raise ValueError("input_state must be provided in trial_params")
        return self.trial_params.get('input_state')

    @property
    def expected_result(self):

        if self.input_state == "-1":
            return None
        
        input_variables = {}
        problem = self.get_problem_instance()
        for i, var in enumerate(problem.vars):
            input_variables[var] = int(self.input_state[i])
        return eval(problem.statement, {}, input_variables)

    @property
    def exact_match_rate(self):
        if self.counts is None:
            raise ValueError("Counts must be set before calculating exact match rate. Use get_counts() or get_counts_async() to update the counts.")
        if self.is_failed:
            return 0.0
        successes = self.counts.get(self.total_expected_results(), 0)
        return successes / sum(self.counts.values())
    
    @property
    def mean_hamming_distance(self):
        """
        the mean hamming distance between the expected result and the measured results per shot, per qubit
        """
        if self.counts is None:
            raise ValueError("Counts must be set before calculating mean hamming distance. Use get_counts() or get_counts_async() to update the counts.")
        
        if self.is_failed:
            return float(self._problem_instance.num_vars + 1)  # max hamming distance
        
        total_distance = 0
        expected = self.total_expected_results()
        for result, count in self.counts.items():
            total_distance += hamming_distance(expected, result) * count

        return total_distance / sum(self.counts.values())

    @property
    def result_qubit_success_rate(self):
        """
        the rate of success for the result qubit of Uf
        """
        if self.counts is None:
            raise ValueError("Counts must be set before calculating correct result qubit rate. Use get_counts() or get_counts_async() to update the counts.")
        
        if self.is_failed:
            return 0.0
        
        correct_result_count = 0
        expected = "1" if self.expected_result else "0"
        result_idx = self._problem_instance.num_vars # result qubit is the qubit which immediately follows the input qubits
        for result, count in self.counts.items():
            if result[result_idx] == expected:
                correct_result_count += count

        return correct_result_count / sum(self.counts.values())

    def calculate_expected_success_rate(
            self,
            db_manager: Optional[BenchmarkDatabase] = None,
    ):
        # expected success rate is 1, as this kind of trial tests
        # the accuracy of pure state (expected) output given pure state input
        return 1
    
    def total_expected_results(self):
        """ 
        Returns the expected binary string to be measured after the circuit is run
        Note that due to little-endian encoding of the counts, the first bit is the result bit
        and the input bits are in reverse order
        """
        result_bit = "1" if self.expected_result else "0"
        return result_bit + self.input_state[::-1]
    
    def calculate_success_rate(
        self,
        db_manager: Optional[BenchmarkDatabase] = None,
    ) -> float:
        # TODO: implement
        return 0
    
    def get_problem_instance(self, db_manager: Optional["BenchmarkDatabase"] = None) -> _ProblemInstance:
        if self._problem_instance is not None:
            return self._problem_instance
        
        if db_manager is None:
            raise ValueError("Database manager is not set.")
        return db_manager.find_problem_instance(self.problem_instance_id)