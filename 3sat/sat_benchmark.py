from enum import Enum
import math
import asyncio
import random
from dotenv import load_dotenv
import json
import os
import sqlite3
import qiskit
from qiskit.circuit.library import grover_operator
from qiskit_ibm_runtime import QiskitRuntimeService, Batch, SamplerV2 as Sampler
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from sat_database import ThreeSat, ThreeSatDB

load_dotenv()
API_TOKEN = os.getenv("API_TOKEN")
API_INSTANCE = os.getenv("API_INSTANCE", None)
service = QiskitRuntimeService(
    channel="ibm_quantum", token=API_TOKEN, instance=API_INSTANCE
)
backend = service.backend(name="ibm_rensselaer")

#DEBUG = os.getenv("DEBUG", False)
DEBUG = True


class CompileType:
    DIRECT = "DIRECT"
    CLASSICAL_FUNCTION = "CLASSICAL_FUNCTION"
    XAG = "XAG"


def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


class Trial:
    def __init__(
        self,
        sat_instance: ThreeSat,
        sat_id,
        compile_type,
        num_vars,
        num_clauses,
        grover_iterations,
        job_id,
        job_pub_idx,
        counts=None,
        simulation_counts=None,
        trial_id=None,
    ):
        if sat_id is None:
            raise ValueError("sat_id must be provided")

        self.trial_id = trial_id
        self.sat_id = sat_id
        self.compile_type = compile_type
        self.num_vars = num_vars
        self.num_clauses = num_clauses
        self.grover_iterations = grover_iterations
        self.job_id = job_id
        self.job_pub_idx = job_pub_idx
        self.counts = counts
        self.simulation_counts = simulation_counts

        # cache the SAT instance to avoid multiple database calls
        self.sat_instance = sat_instance

    @property
    def expected_success_rate(self):
        sat = self.sat_instance
        num_vars = sat.num_vars
        sat_solutions = sat.solutions

        m = len(sat_solutions)
        if m == 0:
            return 0.0

        q = (2 * m) / (2**num_vars)
        if(q == 1.0):
            return 0.5
        theta = math.atan(math.sqrt(q * (2 - q)) / (1 - q))
        phi = math.atan(math.sqrt(q / (2 - q)))
        return math.sin(self.grover_iterations * theta + phi) ** 2

    @property
    def success_rate(self):
        if self.counts is None:
            raise ValueError(
                "Counts must be set before calculating success rate. Use get_counts() or get_counts_async() to update the counts."
            )

        sat_instance = self.sat_instance
        num_solutions_found = 0
        num_shots = 0

        for k, v in self.counts.items():
            if k == "-1":
                num_shots += v
                continue

            # build assignment map variable -> True/False
            # NOTE: variable numbers start at 1 rather than 0
            # reverse k for proper bit order
            
            assignment_map = [i + 1 if bit == '1' else -(i+1) for i, bit in enumerate(k[::-1])]

            # Effecient SAT solver
            from pysat.solvers import Glucose3 as Glucose

            clauses = [list(clause) for clause in sat_instance.expr]
            solver = Glucose(bootstrap_with=clauses)

            if solver.solve(assumptions=assignment_map):
                num_solutions_found += v
            num_shots += v

        return num_solutions_found / num_shots if num_shots > 0 else 0

    @property
    def simulation_success_rate(self):
        if self.simulation_counts is None:
            raise ValueError(
                "Simulation counts must be set before calculating simulation success rate."
            )

        sat_instance = self.sat_instance
        num_solutions_found = 0
        num_shots = 0

        for k, v in self.simulation_counts.items():
            if k == "-1":
                num_shots += v
                continue

            # build assignment map variable -> True/False
            # NOTE: variable numbers start at 1 rather than 0
            assignment_map = dict()
            for i, bit in enumerate(k[::-1]):  # reverse to match bit order
                assignment_map[i + 1] = bit == "1"

            if verify_3sat(sat_instance.expr, assignment_map):
                num_solutions_found += v
            num_shots += v

        return num_solutions_found / num_shots if num_shots > 0 else 0

    def mark_failure(self):
        if self.counts is not None:
            raise Exception("Cannot mark failure if counts are already set")
        self.counts = {"-1": 1}

    def as_dict(self):
        return {
            "job_id": self.job_id,
            "job_pub_idx": self.job_pub_idx,
            "sat_id": self.sat_id,
            "compile_type": self.compile_type,
            "num_vars": self.num_vars,
            "num_clauses": self.num_clauses,
            "grover_iterations": self.grover_iterations,
            "trial_id": self.trial_id,
            "counts": self.counts,
            "simulation_counts": self.simulation_counts,
        }

    def set_counts(self, counts, trials=None):
        """
        call this once the job has finished running
        this updates the trials entry if trials is provided and the entry exists
        """
        self.counts = counts
        if trials is not None:
            trials.save(self)

    async def get_counts_async(self, trials=None):
        """
        same as get_counts, but async; useful for batch loading
        """
        if self.counts is None:
            retrieved_job = await asyncio.to_thread(service.job, self.job_id)
            result = await asyncio.to_thread(retrieved_job.result)
            counts = result[self.job_pub_idx].data.c.get_counts()
            self.set_counts(counts, trials)
        return self.counts

    def get_counts(self, trials=None):
        """
        returns the counts for this job (waiting for them if they are not complete)
        this sets the counts if they are not already set
        if trials is provided, the entry is updated in the database
        """
        if self.counts is None:
            retrieved_job = service.job(self.job_id)
            result = retrieved_job.result()
            counts = result[self.job_pub_idx].data.c.get_counts()
            self.set_counts(counts, trials)
        return self.counts


class Trials:
    def __init__(self, db_name="3sat_trials.db"):
        self.db_name = db_name
        self._initialize_database()

    def _connect(self):
        return sqlite3.connect(self.db_name)

    def _as_trials(self, rows: list):
        return [
            Trial(
                sat_id=row[1],
                compile_type=row[2],
                sat_instance=ThreeSat.from_dict(json.loads(row[3])),
                num_vars=row[4],
                num_clauses=row[5],
                grover_iterations=row[6],
                job_id=row[7],
                job_pub_idx=row[8],
                counts=json.loads(row[9]) if row[9] else None,
                simulation_counts=json.loads(row[10]) if row[10] else None,
                trial_id=row[0],
            )
            for row in rows
        ]

    def _initialize_database(self):
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sat_trials (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sat_id INTEGER,
                    compile_type TEXT,
                    sat_instance TEXT,
                    num_vars INTEGER,
                    num_clauses INTEGER,
                    grover_iterations INTEGER,
                    job_id TEXT,
                    job_pub_idx INTEGER,
                    counts TEXT,
                    simulation_counts TEXT,
                    FOREIGN KEY(sat_id) REFERENCES three_sat(sat_id)
                )
                """
            )
            conn.commit()

    def save(self, trial: Trial):
        sat_id = trial.sat_id
        compile_type = trial.compile_type
        sat_instance = json.dumps(trial.sat_instance.to_dict())
        num_vars = trial.num_vars
        num_clauses = trial.num_clauses
        grover_iterations = trial.grover_iterations
        job_id = trial.job_id
        job_pub_idx = trial.job_pub_idx
        counts = json.dumps(trial.counts) if trial.counts is not None else ""
        simulation_counts = (
            json.dumps(trial.simulation_counts)
            if trial.simulation_counts is not None
            else ""
        )
        trial_id = trial.trial_id

        with self._connect() as conn:
            cursor = conn.cursor()
            if trial_id is None:
                cursor.execute(
                    "INSERT INTO sat_trials (sat_id, compile_type, sat_instance, num_vars, num_clauses, grover_iterations, job_id, job_pub_idx, counts, simulation_counts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        sat_id,
                        compile_type,
                        sat_instance,
                        num_vars,
                        num_clauses,
                        grover_iterations,
                        job_id,
                        job_pub_idx,
                        counts,
                        simulation_counts,
                    ),
                )
                trial.trial_id = cursor.lastrowid
            else:
                cursor.execute(
                    "UPDATE sat_trials SET sat_id = ?, compile_type = ?, sat_instance = ?, num_vars = ?, num_clauses = ?, grover_iterations = ?, job_id = ?, job_pub_idx = ?, counts = ?, simulation_counts = ? WHERE id = ?",
                    (
                        sat_id,
                        compile_type,
                        sat_instance,
                        num_vars,
                        num_clauses,
                        grover_iterations,
                        job_id,
                        job_pub_idx,
                        counts,
                        simulation_counts,
                        trial_id,
                    ),
                )
            conn.commit()

    def delete(self, trial: Trial = None, trial_id=None):
        if trial is None and trial_id is None:
            raise ValueError("Either trial or trial_id must be provided")

        if trial is not None:
            assert trial_id is None or trial_id == trial.trial_id
            trial_id = trial.trial_id

        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM sat_trials WHERE id = ?", (trial_id,))
            conn.commit()

    def all(self):
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sat_trials")
            return self._as_trials(cursor.fetchall())

    def get(
        self,
        trial_id=None,
        sat_id=None,
        num_vars=None,
        num_clauses=None,
        compile_type=None,
        grover_iterations=None,
        job_id=None,
        include_pending=False,
    ):
        """
        Retrieves SAT trials from the database based on specified criteria.
        """
        params = []
        query_parts = []

        if trial_id is not None:
            query_parts.append("id = ?")
            params.append(trial_id)

        if sat_id is not None:
            query_parts.append("sat_id = ?")
            params.append(sat_id)

        if num_vars is not None:
            query_parts.append("num_vars = ?")
            params.append(num_vars)

        if num_clauses is not None:
            query_parts.append("num_clauses = ?")
            params.append(num_clauses)

        if compile_type is not None:
            query_parts.append("compile_type = ?")
            params.append(compile_type)

        if grover_iterations is not None:
            query_parts.append("grover_iterations = ?")
            params.append(grover_iterations)

        if job_id is not None:
            query_parts.append("job_id = ?")
            params.append(job_id)

        # Filter out trials where counts is empty (pending) unless include_pending is True
        if not include_pending:
            query_parts.append("counts != ?")
            params.append("")

        query = "SELECT * FROM sat_trials"

        if query_parts:
            query += " WHERE " + " AND ".join(query_parts)

        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return self._as_trials(cursor.fetchall())

    def _use_job_results(self, job_id, results):
        """
        Given a job_id and the results of the job, 
        this updates the counts for all trials with that job_id
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM sat_trials WHERE job_id = ? AND counts = ''",
                (job_id,),
            )
            trials = self._as_trials(cursor.fetchall())
            for trial in trials:
                trial.counts = results[trial.job_pub_idx].data.c.get_counts()
                self.save(trial)

    async def use_job_results(self, job_id):
        """
        Given a job_id, this fetches the results of the job 
        and updates the counts for all trials with that job_id
        """
        retrieved_job = await asyncio.to_thread(service.job, job_id)
        try:
            results = await asyncio.to_thread(retrieved_job.result)
        except Exception as e:
            print(f"Error fetching results for job {job_id}: {e}")
            return
        self._use_job_results(job_id, results)

    async def load_results(self):
        """
        Loads counts results for all trials that do not yet have them, 
        saving the updated trials to the database
        """
        query = "SELECT DISTINCT job_id FROM sat_trials WHERE counts = ''"
        jobs = []
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            jobs = cursor.fetchall()

        tasks = []
        for job_id in [job[0] for job in jobs]:
            tasks.append(self.use_job_results(job_id))

        batch_size = 3
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i : i + batch_size]
            debug_print(
                f"Processing batch {i // batch_size + 1} of {(len(tasks) + batch_size - 1) // batch_size}"
            )
            await asyncio.gather(*batch)


def run_grover(oracle, n, grover_iterations, shots=10**4):
    """
    Given oracle U_f that has m solutions, this runs a Grover's search circuit using U_f
    and returns the job_id, simulation_counts of the job that was submitted to the backend.
    """
    # NOTE: This may be screwing up expected results 
    uf_mode = oracle.num_qubits == n + 1
    grover_op = grover_operator(oracle, reflection_qubits=range(n))

    search_circuit = qiskit.QuantumCircuit(oracle.num_qubits, n)

    # initialize the result qubit to H |1> if uf_mode
    if uf_mode:
        search_circuit.x(n)
        search_circuit.h(n)

    search_circuit.h(range(n))
    search_circuit.compose(grover_op.power(grover_iterations), inplace=True)
    search_circuit.measure(range(n), range(n))

    qc = qiskit.transpile(search_circuit, backend)
    sampler = Sampler(backend)
    job = sampler.run([qc], shots=shots)

    simulation_counts = None
    try:
        simulator = AerSimulator()
        pass_manager = generate_preset_pass_manager(
            optimization_level=1, backend=simulator
        )
        qc = pass_manager.run(search_circuit)
        result = simulator.run(qc, shots=shots).result()
        simulation_counts = result.get_counts()
    except Exception as e:
        debug_print(f"Error {e} creating simulation")

    return job.job_id(), simulation_counts


def run_benchmark_sample(
    trials: Trials,
    sat_instance: ThreeSat,
    compile_type,
    sat_oracle,
    grover_iterations,
    shots=10**4,
    include_existing_trials=False,
):
    """
    Given a 3SAT instance, this compiles it into a quantum circuit, and runs Grover's algorithm
    on it with the specified number of iterations.

    If include_existing_trials is True, this will skip running if there's already a trial
    for this configuration.

    Returns the Trial object created for this sample.
    """
    debug_print(f"Running benchmark sample for 3SAT instance {sat_instance.sat_id}")

    if include_existing_trials:
        existing_trials = trials.get(
            sat_id=sat_instance.sat_id,
            compile_type=compile_type,
            grover_iterations=grover_iterations,
            include_pending=True,
        )
        if len(existing_trials) > 0:
            debug_print(f"Trial already exists, skipping")
            return existing_trials[0]

    debug_print(
        f"Looking for a satisfying assignment using {grover_iterations} iterations"
    )

    job_id, simulation_counts = run_grover(
        sat_oracle, sat_instance.num_vars, grover_iterations, shots=shots
    )

    trial = Trial(
        sat_instance=sat_instance,
        sat_id=sat_instance.sat_id,
        compile_type=compile_type,
        num_vars=sat_instance.num_vars,
        num_clauses=sat_instance.num_clauses,
        grover_iterations=grover_iterations,
        job_id=job_id,
        job_pub_idx=0,
        simulation_counts=simulation_counts,
    )

    # XAG compilation can result in quantum circuits too large to simulate classically
    # on standard hardware, so I count this as a failure. run_grover will return
    # simulation_counts = None if it failed to simulate the circuit
    if simulation_counts is None:
        trial.mark_failure()

    trials.save(trial)
    return trial


def mark_job_failure(job_id):
    trials = Trials()
    for trial in trials.get(job_id=job_id):
        trial.mark_failure()
        trials.save(trial)
