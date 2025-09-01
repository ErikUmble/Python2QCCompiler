from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import Union, List, Int

from benchmarklib import CompileType

"""
The goal of these classis is to specify a complete and type-safe configuration.
These can be easily serialized and saved to disk, and then loaded into the python
type system for easy resuse. We want to avoid using dictionaries since they are
prone to mistakes and hard to check for validity.

If a config file cannot be deserialized in a format
that the experiment is expecting that should be an error!
"""


@dataclass_json
@dataclass
class GroverCircuitOptions:
    """Parameters for construction of Grover Circuit"""

    dynamical_decoupling: bool = False
    shots: int = 10**4
    optimization_level: int = 3
    # TOOD: Change to Enum? Does python have Sum Types?
    # a hard cap on the number of iterations, or "full" to indicate
    # usage of the optimal number of Grover iterations
    grover_iters: Union[int, str] = "full"


@dataclass_json
@dataclass
class ProblemSelection:
    choose_untested: bool
    random_sample: bool
    limit: int  # a limit of the number of problems selected for a given query


@dataclass_json
@dataclass
class CliqueOptions:
    """Options for Clique Search Problems"""

    # use clique problems with node counts in this list
    nodes_range: List[Int]

    # use clique problems with edge probabilities in this list
    edge_probability_range: List[Int]

    # size of the clique as a proportion of the total number of nodes
    # 0.5 would mean we are searching for a clique of size floor(N/2)
    # NOTE: if the size of the clique is < 2 then the clique size defaults to 2
    clique_size: float = 0.5


@dataclass_json
@dataclass
class SatOptions:
    """Options for 3SAT Search Problems"""

    # use 3SAT problems with variables counts in this list
    num_vars_range: List[Int]
    unit_propagation_levels = 0


@dataclass_json
@dataclass
class MasterConfig:
    """Plain-old-data config struct"""

    db_filename: str
    circuit_opts: GroverCircuitOptions
    problem_selection: ProblemSelection
    problem_opts: Union[CliqueOptions, SatOptions]
    compile_type: List[CompileType]  # list of compile types to try for each problem
