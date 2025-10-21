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

import asyncio
from collections import defaultdict
import logging
import io
from abc import ABC, abstractmethod, ABCMeta
from datetime import datetime
from qiskit import QuantumCircuit, qpy
from qiskit_ibm_runtime import QiskitRuntimeService
from sqlalchemy import Boolean, Column, ForeignKey, Index, Integer, String, JSON, Float, DateTime, LargeBinary, select, create_engine, select, func, or_
from sqlalchemy.orm import  declared_attr, Mapped, relationship, mapped_column, sessionmaker, Session, DeclarativeBase, DeclarativeMeta, Mapped, relationship, selectinload, joinedload
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.ext.hybrid import hybrid_property
from typing import Any, Dict, Optional, TypeVar, Type, List, ClassVar, Union

# Configure logging
logger = logging.getLogger("benchmarklib.core.types")

# It would be nice if we could use ABC and abstractmethod for these base classes
# but the metaclass of ABC conflicts with the SQLAlchemy DeclarativeMeta metaclass.
# I tried using a merged metaclass (described here: https://stackoverflow.com/questions/49581907/when-inheriting-sqlalchemy-class-from-abstract-class-exception-thrown-metaclass)
# but it did not work (error sqlalchemy.exc.ArgumentError: Class '<class 'MyClass'>' already has a primary mapper defined.)
# so instead I opt to use NotImplementedError in the abstract methods.

class Base(DeclarativeBase):
    """
    Base class for all ORM models, providing common attributes.
    """
    __abstract__ = True

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, onupdate=datetime.utcnow)

ProblemT = TypeVar("ProblemT", bound="BaseProblem")
TrialT = TypeVar("TrialT", bound="BaseTrial")

class BaseProblem(Base):
    """
    Abstract base class for problem instances.

    Each quantum optimization problem (3SAT, Clique, Boolean Function, etc.)
    should implement this interface. The problem instance represents the
    classical problem being solved, independent of any specific quantum
    circuit compilation or execution.

    Must specify:
        __tablename__: str # Database table name for this problem type
        TrialClass: Type[BaseTrial] # Corresponding trial class for this problem type 

    Key responsibilities:
    - Generate quantum oracle circuits for the problem
    - Verify solutions

    Attributes:
        id: Database primary key (None for unsaved instances)
        extra_data: JSON field for irregular problem-specific data (it is generally better to create separate columns for additional attributes)
        trials: List of related trials for this problem instance
    """
    __abstract__ = True

    TrialClass: ClassVar[Type["BaseTrial"]]
    extra_data: Mapped[Dict[str, Any]] = mapped_column(MutableDict.as_mutable(JSON), default=dict)

    @declared_attr
    def trials(cls) -> Mapped[List[TrialT]]:
        # reverse foreign key relationship to trials depends on the trials subclass for this problem type
        return relationship(
            cls.TrialClass, 
            back_populates="problem", 
            cascade="all, delete-orphan"
        )

    @property
    def problem_type(self) -> str:
        """String identifier for the problem type (e.g., '3SAT', 'Clique')."""
        # default to class name
        return self.__class__.__name__

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
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement get_problem_size(self) -> Dict[str, int]."
        )

    def number_of_input_bits(self) -> int:
        """
        Returns the number of input bits for the generated quantum oracle
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement number_of_input_bits(self) -> int."
        )

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

    def verify_solution(self, solution: Union[str, List[bool]], **kwargs) -> bool:
        """
        Verify if a proposed solution satisfies the problem constraints
        Args:
            solution: Proposed solution (bit string or boolean list)
            **kwargs: Additional verification parameter
        Returns:
            True if solution is valid, False otherwise
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement verify_solution(self, solution: Union[str, List[bool]], **kwargs) -> bool."
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        size_info = ", ".join(f"{k}={v}" for k, v in self.get_problem_size().items())
        return f"{self.problem_type}({size_info})"

class BaseTrial(Base):
    """
    Abstract base class for benchmark trials.

    A trial represents a single execution of a quantum circuit on a specific
    problem instance with given parameters. The trial references a problem
    instance by ID rather than storing the full problem data.

    Attributes:
        id: Database primary key (None for unsaved trials)
        problem_id: foreign key to the problem instance
        problem: the related problem itself, namely the Python object of the problem with id = problem_id
        compiler_name: How the quantum circuit was compiled
        job_id: IBM Quantum job identifier
        job_pub_idx: Index within the job for this circuit
        counts: Measurement results from quantum hardware
        simulation_counts: Classical simulation results for comparison
        is_failed: Was this trial unable to be executed (this includes if it failed to be compiled)
        created_at: Timestamp when trial was created
        updated_at: Timestamp when trial was modified
        circuit: the QiskitCircuit associated with the trial (after transpilation)


    """
    __abstract__ = True

    ProblemClass: ClassVar[Type["BaseProblem"]]

    compiler_name: Mapped[str] = mapped_column(String(255), index=True)
    job_id: Mapped[Optional[str]] = mapped_column(String(128), index=True)
    job_pub_idx: Mapped[Optional[int]]
    counts: Mapped[Optional[Dict[str, int]]] = mapped_column(MutableDict.as_mutable(JSON))
    simulation_counts: Mapped[Optional[Dict[str, int]]] = mapped_column(MutableDict.as_mutable(JSON))
    is_failed: Mapped[bool] = mapped_column(Boolean, default=False)
    _circuit_qpy: Mapped[Optional[bytes]] = mapped_column(LargeBinary)
    
    _circuit: Optional[QuantumCircuit] = None

    ProblemClass: Type[ProblemT]


    @declared_attr
    def __table_args__(cls):
        return (
            Index('ix_is_pending', 'job_id', 'counts', 'is_failed'),
        )

    @declared_attr
    def problem_id(cls) -> Mapped[int]:
        # foreign key to related problem depends on the problem subclass
        problem_table_name = cls.ProblemClass.__tablename__
        return mapped_column(ForeignKey(f"{problem_table_name}.id"), index=True)

    @declared_attr
    def problem(cls) -> Mapped[ProblemT]:
        return relationship(
            cls.ProblemClass, 
            back_populates="trials"
        )

    @hybrid_property
    def circuit(self):
        if self._circuit is not None:
            return self._circuit
        if self._circuit_qpy is not None:
            buffer = io.BytesIO(self._circuit_qpy)
            self._circuit = qpy.load(buffer)[0]
            # cache the loaded circuit
            return self._circuit
        return None
    
    @circuit.setter
    def circuit(self, qc: Optional[QuantumCircuit]):
        self._circuit = qc
        if qc is not None:
            buffer = io.BytesIO()
            qpy.dump(qc, buffer)
            self._circuit_qpy = buffer.getvalue()
        else:
            self._circuit_qpy = None

    @property
    def instance_id(self) -> int:
        # alias for backwards compatibility
        return self.problem_id

    @property
    def is_pending(self) -> bool:
        """Check if trial is waiting for results."""
        return self.job_id is not None and self.counts is None and not self.is_failed

    async def get_ibm_circuit(self, service: QiskitRuntimeService) -> QuantumCircuit:
        """
        Get this trial's Quantum Circuit from IBM Quantum Runtime Service.
        """
        if self.job_id is None or self.job_pub_idx is None:
            raise ValueError("Cannot get IBM circuit - job_id or job_pub_idx is None")
        job = await asyncio.to_thread(service.job, self.job_id)
        circuit = job.inputs['pubs'][self.job_pub_idx][0]
        return circuit

    def mark_failure(self) -> None:
        """Mark this trial as failed."""
        if self.counts is not None:
            raise ValueError("Cannot mark failure - counts already set")
        self.is_failed = True

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
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement calculate_success_rate(self, db_manager: Optional[BenchmarkDatabase] = None) -> float."
        )

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
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement calculate_expected_success_rate(self, db_manager: Optional[BenchmarkDatabase] = None) -> float."
        )

    def get_problem_instance(self, db_manager: "BenchmarkDatabase") -> BaseProblem:
        """
        Get the problem instance for this trial, loading and caching if needed.

        Args:
            db_manager: Database manager to load from

        Returns:
            The problem instance for this trial
        """
        logger.debug(f"get_problem_instance is deprecated, use self.problem instead")
        return self.problem

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


class TrialCircuitMetricsMixin:
    """
    Mixin to store circuit metrics directly (cache these values for fast querying and loading)
    Attributes:
        circuit_depth: maximum operation depth for any qubit in the circuit
        circuit_op_counts: dictionary of operation counts by gate type
        circuit_num_single_qubit_gates: total number of single qubit gates
        circuit_num_gates: total number of gates (does not include measurements)
        circuit_num_qubits: total number of qubits used (qubits with at least one gate applied)

    All of these attributes can be derived from the circuit (whether saved in the local database or loaded from ibm cloud).
    As such, each attribute is optional (nullable). If a circuit is provided to the trial in __init__, 
    this will automatically compute and store the circuit metrics.
    Use .load_circuit_metrics() method to compute these metrics them to the database at any other time.

    Usage:
        class MyTrial(TrialCircuitMetricsMixin, BaseTrial):
    """
    circuit_depth: Mapped[Optional[int]]
    circuit_op_counts: Mapped[Optional[Dict[str, int]]] = mapped_column(MutableDict.as_mutable(JSON))
    circuit_num_single_qubit_gates: Mapped[Optional[int]]
    circuit_num_gates: Mapped[Optional[int]]
    circuit_num_qubits: Mapped[Optional[int]]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # if a circuit is provided, compute and store the metrics
        if self.circuit is None:
            self.circuit = kwargs.get('circuit', None)

        if self.circuit is not None:
            self._compute_circuit_metrics()

    def _compute_circuit_metrics(self, circuit: QuantumCircuit) -> None:
        """Compute and store circuit metrics from the provided circuit"""

        # count number of gates by the number of qubits they act on
        counts = defaultdict(int)
        for inst in circuit.data:
            counts[len(inst.qubits)] += 1

        self.circuit_depth = circuit.depth()
        self.circuit_op_counts = circuit.count_ops()
        self.circuit_num_single_qubit_gates = counts[1]
        self.circuit_num_gates = circuit.size() - self.circuit_op_counts.get("measure", 0)
        self.circuit_num_qubits = len({q for instr, qargs, _ in circuit.data for q in qargs})  # using this instead of circuit.num_qubits to count only qubits actually used

    def load_circuit_metrics(self, circuit: Optional[QuantumCircuit] = None) -> None:
        """
        Compute and circuit metrics (storing as attributes of this trial).
        If circuit is provided, it will be used to derive the metrics; otherwise,
        the existing self.circuit will be used. ValueError is raised if circuit is None and self.circuit is also None.
        Note: this does NOT save to the database, you must call db_manager.save_trial(trial) separately.
        """
        circuit = circuit or self.circuit
        if circuit is None:
            raise ValueError("Cannot compute circuit metrics - circuit is None. Consider using load_circuit_metrics(circuit=self.get_ibm_circuit(service)).")
        self._compute_circuit_metrics(circuit)

class _ProblemInstance(ABC):
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


class _BaseTrial(ABC):
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
        problem_instance: _ProblemInstance,
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

    def load_problem_instance(self, db_manager: "_BenchmarkDatabase") -> _ProblemInstance:
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

    def get_problem_instance(self, db_manager: "_BenchmarkDatabase") -> _ProblemInstance:
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
