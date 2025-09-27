from typing import List, Tuple
from benchmarklib import ProblemInstance, CompileType, BaseTrial, BenchmarkDatabase
from tweedledum.bool_function_compiler import circuit_input, QuantumCircuitFunction
from tweedledum import BitVec
from typing import List, Tuple, Dict, Any, Union, Optional
from qiskit import QuantumCircuit
import random

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
    for var, replacement in var_map.items():
        statement = statement.replace(var, replacement)
        
    return f"""
@circuit_input(vars=BitVec({len(variables)}))
def {name}() -> BitVec(1):
    return {statement}
    """


class RandomBooleanFunction(ProblemInstance):
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
            return classical_function.synthesize_quantum_circuit()
        
        raise ValueError(f"{compile_type} not yet supported for RandomBooleanFunction")


class RandomBooleanFunctionTrial(BaseTrial):

    @property
    def input_state(self) -> str:
        if not self.trial_params.get('input_state'):
            raise ValueError("input_state must be provided in trial_params")
        return self.trial_params.get('input_state')

    @property
    def expected_result(self):
        input_variables = {}
        problem = self.get_problem_instance()
        for i, var in enumerate(problem.vars):
            input_variables[var] = int(self.input_state[i])
        return eval(problem.statement, {}, input_variables)

    @property
    def exact_match_rate(self):
        if self.counts is None:
            raise ValueError("Counts must be set before calculating exact match rate. Use get_counts() or get_counts_async() to update the counts.")
        if self.counts.get('-1') is not None:
            return 0
        successes = self.counts.get(self.total_expected_results(), 0)
        return successes / sum(self.counts.values())
    
    @property
    def mean_hamming_distance(self):
        """
        the mean hamming distance between the expected result and the measured results per shot, per qubit
        """
        if self.counts is None:
            raise ValueError("Counts must be set before calculating mean hamming distance. Use get_counts() or get_counts_async() to update the counts.")
        
        total_distance = 0
        expected = self.total_expected_results()
        for result, count in self.counts.items():
            total_distance += hamming_distance(expected, result) * count

        return total_distance / sum(self.counts.values())

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
    
    def get_problem_instance(self, db_manager: Optional["BenchmarkDatabase"] = None) -> ProblemInstance:
        if self._problem_instance is not None:
            return self._problem_instance
        
        if db_manager is None:
            raise ValueError("Database manager is not set.")
        return db_manager.find_problem_instance(self.problem_instance_id)