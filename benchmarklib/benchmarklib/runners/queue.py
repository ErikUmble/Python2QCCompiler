import logging
import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import qiskit
from qiskit import transpile
from qiskit.circuit.library import grover_operator, QFT
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.providers import Backend
from qiskit_aer import AerSimulator

# IBM Quantum imports
from qiskit_ibm_runtime import Batch, QiskitRuntimeService, SamplerOptions
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit_ibm_runtime.options import dynamical_decoupling_options
from qiskit_ibm_runtime.exceptions import IBMRuntimeError

from ..core import BaseTrial, BenchmarkDatabase, ProblemInstance
from ..compilers import SynthesisCompiler

logger = logging.getLogger("benchmarklib.runners.queue")


class Task:
     def __init__(self, trial: BaseTrial, circuit: qiskit.QuantumCircuit):
        self.trial = trial
        self.circuit = circuit

class BatchQueue:
    """
    This class provides batch processing management for submitting circuits to a Quantum Computer or Simulator backend.
    It provides an adaptive job submission strategy which seeks to optimize throughput by submitting many circuits per job,
    while decreasing job size exponentially until this succeeds.

    Simply create a BatchQueue, add tasks using add(), and call finish_batch() when done. Jobs will automatically be submitted
    when the number of pending tasks exceeds the adaptively managed job size.
    """
    def __init__(
            self, 
            db_manager: BenchmarkDatabase, 
            backend: Backend,
            max_job_size: int = 100,
            shots: int = 1024,
            transpile_kwargs: Optional[Dict] = {},
            sampler_options: Optional[SamplerOptions] = None,
            run_simulation: bool = False
        ):
            self.db_manager = db_manager
            self.backend = backend
            self.max_job_size = max_job_size
            self.job_size = max_job_size
            self.transpile_kwargs = transpile_kwargs
            self.sampler_options = sampler_options if sampler_options else SamplerOptions()
            self.shots = shots
            self.run_simulation = run_simulation

            # state management
            self._batch_context = None
            self._batch_job_count = 0
            self._tasks = []

    def enqueue(self, trial: BaseTrial, circuit: qiskit.QuantumCircuit):
        """
        Add a trial with associated circuit to the batch queue. If simulation is enabled, this simulates the circuit immediately.
        """
        task = Task(trial, circuit)
        self._tasks.append(task)

        if self.run_simulation and task.trial.simulation_counts is None:
            task.trial.simulation_counts = run_simulation(
                circuit,
                shots=self.shots,
                transpile_kwargs=self.transpile_kwargs
            )
            self.db_manager.save_trial(task.trial)


        if len(self._tasks) >= self.job_size:
            self.submit_tasks()

    def start_batch(self):
        if self._batch_context is not None:
            self.finish_batch()
        self._batch_context = Batch(backend=self.backend)
        self._batch_context.__enter__()

    def finish_batch(self):
        """
        Finish the current batch and clean up resources.
        """
        if len(self._tasks) > 0:
            logger.warning(
                f"Auto-submitting {len(self._tasks)} remaining circuits"
            )
            self.submit_tasks()

        if self._batch_context is None:
            return
        
        # Clean up batch context
        try:
            self._batch_context.__exit__(None, None, None)
        except Exception as e:
            logger.error(f"Error closing batch context: {e}")

        logger.info(
            f"finished batch ({self._batch_job_count} jobs submitted)"
        )
        
        # reset state
        self._batch_context = None
        self._batch_job_count = 0

    def get_batch_stats(self) -> Dict:
        """Get statistics about current batch."""
        return {
            "active_batch": self._batch_context is not None,
            "pending_tasks": len(self._tasks),
            "jobs_submitted": self._batch_job_count,
        }
    
    def _submit_job(self, tasks: list[Task]) -> Optional[str]:
        """
        Submit a job with the given tasks to the backend within the current batch context.
        Returns the job ID if successful, None otherwise.
        """
        if self._batch_context is None:
            raise ValueError("No active batch - call start_batch() first")

        if not tasks or len(tasks) == 0:
            logger.warning("No circuits to submit")
            return None

        logger.info(f"Attempting to submit job with {len(tasks)} circuits")

        
        circuits = [task.circuit for task in tasks]

        # Transpile circuits
        logger.debug(f"Transpiling {len(circuits)} circuits...")
        transpiled_circuits = transpile(
            circuits,
            backend=self.backend,
            **self.transpile_kwargs,
        )

        # Submit job within batch context
        sampler = Sampler(mode=self._batch_context, options=self.sampler_options)

        try:
            # Submit job
            job = sampler.run(transpiled_circuits, shots=self.shots)
            job_id = job.job_id()
            return job_id
        
        except IBMRuntimeError as e:
            logger.error(f"Error submitting job: {e}")
            return None
    

    def submit_tasks(self):
        """
        Fulfill all pending tasks using the current batch context.
        This creates a new batch context if one does not exist.
        """
        if len(self._tasks) == 0:
            return
        
        if self._batch_context is None:
            self.start_batch()

        # attempt to run a job with all the tasks
        # if it fails, try smaller jobs
        job_tasks = [t for t in self._tasks]
        while len(job_tasks) > 0:

            job_id = self._submit_job(job_tasks)

            if job_id is not None:
                # assign job_id to tasks
                for job_pub_idx, task in enumerate(job_tasks):
                    task.trial.job_id = job_id
                    task.trial.job_pub_idx = job_pub_idx
                    self.db_manager.save_trial(task.trial)

                if self.job_size < self.max_job_size:
                    # slowly increase job size if successful
                    self.job_size += 1

                self._batch_job_count += 1

            elif len(job_tasks) == 1:
                logger.error("Failed to submit task")
                job_tasks[0].trial.mark_failure()
                self.db_manager.save_trial(job_tasks[0].trial)

            else:
                # adjust job_tasks size and try again
                self.job_size = math.ceil(len(job_tasks) / 2)
                job_tasks = job_tasks[: self.job_size]

                logger.warning(
                    f"Job submission failed, trying smaller batch of size {len(job_tasks)}"
                )
                continue

            
            # remove completed/failed tasks from queue
            self._tasks = self._tasks[len(job_tasks):]

            # create another job_tasks of at most the same size as the previous
            job_tasks = self._tasks[:len(job_tasks)]


def run_simulation(
        circuit: qiskit.QuantumCircuit, simulator: Optional[AerSimulator] = None, max_circuit_depth: Optional[int] = None, shots: int = 10**3, transpile_kwargs: Optional[Dict] = {}
    ) -> Optional[Dict[str, int]]:
    """
    Run classical simulation.

    Args:
        circuit: Quantum circuit to simulate

    Returns:
        Measurement counts or None if failed
    """
    
    simulator = simulator if simulator else AerSimulator()

    try:
        # Transpile for simulator
        qc = transpile(
            circuit,
            simulator,
            **transpile_kwargs,
        )

        # Check complexity limits
        if (
            max_circuit_depth
            and qc.depth() > max_circuit_depth
        ):
            logger.warning(f"Circuit too deep: {qc.depth()}")
            return None

        # Run simulation
        result = simulator.run(qc, shots=shots).result()
        counts = result.get_counts()

        logger.debug(f"Simulation completed: {len(counts)} unique outcomes")
        return counts

    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        return None


def run_tasks(
        tasks: Iterable[Task], 
            db_manager: BenchmarkDatabase, 
            service: QiskitRuntimeService, 
            backend: Backend,
            max_job_size: int = 100,
            transpile_kwargs: Optional[Dict] = {},
            sampler_options: Optional[SamplerOptions] = None
    ):

    batch_queue = BatchQueue(
        db_manager=db_manager,
        service=service,
        backend=backend,
        max_job_size=max_job_size,
        transpile_kwargs=transpile_kwargs,
        sampler_options=sampler_options
    )

    for task in tasks:
        batch_queue.add(task)

    batch_queue.finish_batch()
