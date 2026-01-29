from sqlalchemy import Column, Integer, String, JSON, ForeignKey, Text, Float, DateTime, LargeBinary, Index, select, func
from sqlalchemy.orm import declarative_base, Mapped, relationship, mapped_column, reconstructor, declared_attr
from sqlalchemy.ext.mutable import MutableDict
from typing import Any, Callable, Dict, Optional, Tuple, Type
import logging

from benchmarklib import BaseProblem, CompileType, BaseTrial, BenchmarkDatabase, TrialCircuitMetricsMixin

logger = logging.getLogger("experiments.experiments")

class ExperimentProblem(BaseProblem):
    __tablename__ = "experiment_problems"
    TrialClass = "ExperimentTrial"

    n: Mapped[int]
    name: Mapped[str] = Column(String(128), nullable=False, index=True)
    tag: Mapped[Optional[str]] = Column(String(64), nullable=True, index=True)
    notes: Mapped[Optional[str]] = Column(Text, nullable=True)
    
    # extra_data already included by BaseProblem

    def __init__(self, verifier: Optional[Callable[Tuple[bool], bool]] = None, **kwargs):
        super().__init__(**kwargs)
        self.verifier = verifier

    def get_verifier(self) -> Callable[[Tuple[bool]], bool]:
        if not self.verifier:
            raise ValueError("No verifier function provided for this problem.")
        return self.verifier
    
    def get_verifier_src(self) -> str:
        """Get the source code of the verifier function as a string."""
        import inspect

        if not self.verifier:
            raise ValueError("No verifier function provided for this problem.")
        
        source = inspect.getsource(self.verifier)

        # ensure prototype is 
        # def verify(inpt: Tuple[bool]) -> bool:
        if not source.strip().startswith("def verify(inpt: Tuple[bool]) -> bool:"):
            # TODO: should we throw an error instead of warning here?
            logger.warning("Verifier function prototype does not match expected 'def verify(inpt: Tuple[bool]) -> bool:'. Casting to this expected format could introduce errors.")
            source = source.replace(source.split("\n")[0], "def verify(inpt: Tuple[bool]) -> bool:")

        return source
    
    def number_of_input_bits(self) -> int:
        return self.n

class ExperimentTrial(TrialCircuitMetricsMixin, BaseTrial):
    __tablename__ = "experiment_trials"
    ProblemClass = ExperimentProblem

    compiler_name: Mapped[Optional[str]] = mapped_column(String(255), index=True)  # make optional
    extra_data: Mapped[Dict[str, Any]] = mapped_column(MutableDict.as_mutable(JSON), default=dict)