"""
Quantum Benchmarking Database Library (Simplified Single-Problem Design)

A unified database interface for quantum circuit benchmarking, designed for
single problem types per database. Each problem type (3SAT, Clique, etc.)
should have its own directory with a dedicated database.

Database Schema (per problem type):
- problem_instances: Stores unique problem instances of one type
- trials: Stores trial results referencing problem instances by ID

Key Features:
- Single problem type per database (simplified design)
- Normalized database (no duplicate problem storage across trials)
- Abstract oracle method for centralized circuit generation
- Async job result fetching from IBM Quantum
- Comprehensive documentation and maintenance tools

"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional

# Configure logging
logger = logging.getLogger("benchmarklib.core.types")


class ProblemInstance(ABC):
    """
    Abstract base class for problem instances.

    Each quantum optimization problem (3SAT, Clique, Boolean Function, etc.)
    should implement this interface. The problem instance represents the
    classical problem being solved, independent of any specific quantum
    circuit compilation or execution.

    Key responsibilities:
    - Serialize/deserialize problem data
    - Generate quantum oracle circuits for the problem
    - Provide problem size metrics
    - Verify solutions
    """

    def __init__(self, instance_id: Optional[int] = None):
        """
        Initialize problem instance.

        Args:
            instance_id: Database ID (None for unsaved instances)
        """
        self.instance_id = instance_id

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the problem instance to a JSON-serializable dictionary.

        This should include all data needed to reconstruct the problem,
        but NOT the instance_id (that's handled by the database layer).

        Returns:
            Dictionary containing all problem data
        """
        pass

    @classmethod
    @abstractmethod
    def from_dict(
        cls, data: Dict[str, Any], instance_id: Optional[int] = None
    ) -> "ProblemInstance":
        """
        Create a problem instance from a dictionary.

        Args:
            data: Dictionary from to_dict()
            instance_id: Database ID to assign

        Returns:
            Reconstructed problem instance
        """
        pass

    @property
    @abstractmethod
    def problem_type(self) -> str:
        """String identifier for the problem type (e.g., '3SAT', 'Clique')."""
        pass

    @abstractmethod
    def get_problem_size(self) -> Dict[str, int]:
        """
        Return key size metrics for this problem.

        Examples:
        - 3SAT: {"num_vars": 10, "num_clauses": 30}
        - Clique: {"num_vertices": 8, "num_edges": 20}
        - Boolean: {"num_vars": 5, "complexity": 15}

        Returns:
            Dictionary of size metrics
        """
        pass

    @abstractmethod
    def number_of_input_bits(self) -> int:
        """
        Returns the number of intput bits for the generatred quantum oracle
        """
        pass

    def get_number_of_solutions(self, **trial_params) -> int:
        """
        Return the number of valid solutions for this problem instance.

        This method calculates the number of bit strings that satisfy the problem
        constraints, which is essential for theoretical analysis of quantum algorithms
        like Grover's search. The number of solutions (M) appears in the theoretical
        success probability formula and expected number of trials calculations.

        Args:
            **trial_params: Trial-specific parameters that may affect the solution count.
                           These should match the parameters used in oracle() and stored
                           in trial.trial_params. Examples:
                           - For clique problems: clique_size=4
                           - For subset problems: subset_size=3
                           - For 3SAT: typically no additional parameters needed

        Returns:
            Integer number of solutions (M) for this problem instance with given parameters.
            Must be between 1 and 2^n where n is the number of input bits.

        Raises:
            NotImplementedError: If the problem class hasn't implemented solution counting
            ValueError: If trial_params are invalid or inconsistent with the problem

        Examples:
            # 3SAT problem with 3 variables, 2 satisfying assignments
            sat_problem.get_number_of_solutions()  # Returns: 2

            # Clique problem looking for 4-cliques in a graph
            clique_problem.get_number_of_solutions(clique_size=4)  # Returns: 5

            # Boolean function with specific output requirements
            bool_problem.get_number_of_solutions(target_output=1)  # Returns: 8

        Implementation Notes:
            - For problems where solution count is expensive to compute, consider caching
            - The result should be deterministic for the same instance and parameters
            - If no solutions exist, return 0 (though this makes Grover's algorithm meaningless)
            - For exhaustive search problems, this might require classical computation
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_number_of_solutions(). "
            "This method is required for theoretical quantum algorithm analysis."
        )

    # @abstractmethod
    # def verify_solution(self, solution: Union[str, List[bool]], **kwargs) -> bool:
    #     """
    #     Verify if a proposed solution satisfies the problem constraints.

    #     Args:
    #         solution: Proposed solution (bit string or boolean list)
    #         **kwargs: Additional verification parameters

    #     Returns:
    #         True if solution is valid, False otherwise
    #     """
    #     pass

    def __str__(self) -> str:
        """Human-readable string representation."""
        size_info = ", ".join(f"{k}={v}" for k, v in self.get_problem_size().items())
        return f"{self.problem_type}({size_info})"


class BaseTrial(ABC):
    """
    Abstract base class for benchmark trials.

    A trial represents a single execution of a quantum circuit on a specific
    problem instance with given parameters. The trial references a problem
    instance by ID rather than storing the full problem data.

    Attributes:
        trial_id: Database primary key (None for unsaved trials)
        problem_instance: ProblemInstance related to this trial by foreign key
        compiler_name: How the quantum circuit was compiled
        job_id: IBM Quantum job identifier
        job_pub_idx: Index within the job for this circuit
        counts: Measurement results from quantum hardware
        simulation_counts: Classical simulation results for comparison
        trial_params: Trial-specific parameters (problem-dependent)
        created_at: Timestamp when trial was created
    """

    def __init__(
        self,
        problem_instance: ProblemInstance,
        compiler_name: str,
        job_id: Optional[str] = None,
        job_pub_idx: int = 0,
        counts: Optional[Dict[str, int]] = None,
        simulation_counts: Optional[Dict[str, int]] = None,
        trial_id: Optional[int] = None,
        created_at: Optional[str] = None,
        **trial_params,
    ):
        """
        Initialize a benchmark trial.

        Args:
            instance_id: ID of the problem instance being solved
            compiler_name: Compilation method used
            job_id: IBM Quantum job ID (None if not submitted)
            job_pub_idx: Index within job for this circuit
            counts: Quantum hardware measurement results
            simulation_counts: Classical simulation results
            trial_id: Database ID (None for unsaved trials)
            created_at: ISO timestamp string
            **trial_params: Trial-specific parameters (vary by problem type)
        """
        self.trial_id = trial_id
        self.compiler_name = compiler_name
        self.job_id = job_id
        self.job_pub_idx = job_pub_idx
        self.counts = counts
        self.simulation_counts = simulation_counts
        self.trial_params = trial_params
        self.created_at = created_at or datetime.now().isoformat()

        self._problem_instance = problem_instance

    @property
    def instance_id(self) -> int:
        return self._problem_instance.instance_id

    @property
    def is_pending(self) -> bool:
        """Check if trial is waiting for results."""
        return self.counts is None and self.job_id is not None

    @property
    def is_failed(self) -> bool:
        """Check if trial failed (marked with '-1' counts)."""
        return self.counts is not None and self.counts.get("-1") is not None

    def mark_failure(self) -> None:
        """Mark this trial as failed."""
        if self.counts is not None:
            raise ValueError("Cannot mark failure - counts already set")
        self.counts = {"-1": 1}

    def load_problem_instance(self, db_manager: "BenchmarkDatabase") -> ProblemInstance:
        """
        Load and cache the problem instance for this trial.

        Args:
            db_manager: Database manager to fetch problem instance

        Returns:
            The problem instance for this trial
        """
        if self._problem_instance is None:
            self._problem_instance = db_manager.get_problem_instance(self.instance_id)
        return self._problem_instance

    @abstractmethod
    def calculate_success_rate(
        self,
        db_manager: Optional["BenchmarkDatabase"] = None,
    ) -> float:
        """
        Calculate the success rate for this trial.

        Args:
            db_manager: Database manager for loading problem instance

        Returns:
            Float between 0 and 1 representing success rate
        """
        pass

    @abstractmethod
    def calculate_expected_success_rate(
        self,
        db_manager: Optional["BenchmarkDatabase"] = None,
    ) -> float:
        """
        Calculate the theoretical expected success rate.

        Args:
            db_manager: Database manager for loading problem instance

        Returns:
            Float between 0 and 1 representing expected success rate
        """
        pass

    def get_problem_instance(self, db_manager: "BenchmarkDatabase") -> ProblemInstance:
        """
        Get the problem instance for this trial, loading and caching if needed.

        Args:
            db_manager: Database manager to load from

        Returns:
            The problem instance for this trial
        """
        if self._problem_instance is None:
            self._problem_instance = db_manager.get_problem_instance(self.instance_id)
        return self._problem_instance

    def to_dict(self) -> Dict[str, Any]:
        """Convert trial to dictionary for serialization."""
        return {
            "trial_id": self.trial_id,
            "instance_id": self.instance_id,
            "compiler_name": self.compiler_name,
            "job_id": self.job_id,
            "job_pub_idx": self.job_pub_idx,
            "counts": self.counts,
            "simulation_counts": self.simulation_counts,
            "trial_params": self.trial_params,
            "created_at": self.created_at,
        }
