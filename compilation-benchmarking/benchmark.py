import asyncio
import random
from dotenv import load_dotenv
import importlib.util
import json
import os
import sqlite3
import tempfile
import qiskit
from qiskit_ibm_runtime import QiskitRuntimeService, Batch, SamplerV2 as Sampler

from function_generator import get_variables, get_python_function, get_classical_function, get_random_statement

load_dotenv()
API_TOKEN = os.getenv("API_TOKEN")
API_INSTANCE = os.getenv("API_INSTANCE", None)
service = QiskitRuntimeService(channel="ibm_quantum", token=API_TOKEN, instance=API_INSTANCE)
backend = service.backend(name="ibm_rensselaer")

DEBUG = os.getenv("DEBUG", False)

def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

def hamming_distance(a, b):
    """
    returns the hamming distance between two binary strings
    the strings must be the same length
    """
    assert len(a) == len(b)
    return sum([1 for i in range(len(a)) if a[i] != b[i]])

class Trial:
    def __init__(self, num_vars, complexity, job_id, job_pub_idx, input_state, statement, counts=None, trial_id=None):
        self.trial_id = trial_id
        self.num_vars = num_vars
        self.compexity = complexity
        self.job_id = job_id
        self.job_pub_idx = job_pub_idx
        self.input_state = input_state
        self.statement = statement
        self.counts = counts

    @property
    def variables(self):
        return get_variables(self.num_vars)

    @property
    def expected_result(self):
        input_variables = {}
        for i, var in enumerate(self.variables):
            input_variables[var] = int(self.input_state[i])
        return eval(self.statement, {}, input_variables)
    
    @property
    def exact_match_rate(self):
        if self.counts is None:
            raise ValueError("Counts must be set before calculating exact match rate. Use get_counts() or get_counts_async() to update the counts.")
        
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

    def as_dict(self):
        return {
            "job_id": self.job_id,
            "input_state": self.input_state,
            "statement": self.statement,
            "variables": self.variables,
            "expected_result": self.expected_result,
            "counts": self.counts
        }
    
    def set_counts(self, counts, trials = None):
        """ 
        call this once the job has finished running

        this updates the trials entry trials is provided and the entry exists
        """
        self.counts = counts
        if trials is not None:
            trials.save(self)

    async def get_counts_async(self, trials = None):
        """
        same as get_counts, but async; useful for batch loading
        """
        if self.counts is None:
            retrieved_job = await asyncio.to_thread(service.job, self.job_id)
            result = await asyncio.to_thread(retrieved_job.result)
            counts = result[self.job_pub_idx].data.meas.get_counts()
            self.set_counts(counts, trials)
        return self.counts
    
    def get_counts(self, trials = None):
        """
        returns the counts for this job (waiting for them if they are not complete)
        this sets the counts if they are not already set
        if trials is provided, the entry is updated in the database
        """
        if self.counts is None:
            retrieved_job = service.job(self.job_id)
            result = retrieved_job.result()
            counts = result[self.job_pub_idx].data.meas.get_counts()
            self.set_counts(counts, trials)
        return self.counts

    def total_expected_results(self):
        """ 
        Returns the expected binary string to be measured after the circuit is run
        Note that due to little-endian encoding of the counts, the first bit is the result bit
        and the input bits are in reverse order
        """
        result_bit = "1" if self.expected_result else "0"
        return result_bit + self.input_state[::-1]

class Trials:
    def __init__(self, db_name="benchmark.db"):
        self.db_name = db_name
        self._initialize_database()

    def _connect(self):
        return sqlite3.connect(self.db_name)
    
    def _as_trials(self, rows: list):
        return [Trial(
            num_vars = row[1],
            complexity = row[2],
            statement = row[3],
            input_state = row[4],
            job_id = row[5],
            job_pub_idx = row[6],
            counts = json.loads(row[7]) if row[7] else None,
            trial_id = row[0]
        ) for row in rows]

    def _initialize_database(self):
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS trials (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    num_vars INTEGER,
                    complexity INTEGER,
                    statement TEXT,
                    input_state TEXT,
                    job_id TEXT,
                    job_pub_idx INTEGER,
                    counts TEXT
                )
                """
            )
            conn.commit()

    def save(self, trial: Trial):
        num_vars = trial.num_vars
        complexity = trial.compexity
        statement = trial.statement
        input_state = trial.input_state
        job_id = trial.job_id
        job_pub_idx = trial.job_pub_idx
        counts = json.dumps(trial.counts) if trial.counts is not None else ""
        trial_id = trial.trial_id

        with self._connect() as conn:
            cursor = conn.cursor()
            if trial_id is None:
                cursor.execute(
                    "INSERT INTO trials (num_vars, complexity, statement, input_state, job_id, job_pub_idx, counts) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (num_vars, complexity, statement, input_state, job_id, job_pub_idx, counts)
                )
                trial.trial_id = cursor.lastrowid
            else:
                cursor.execute(
                    "UPDATE trials SET num_vars = ?, complexity = ?, statement = ?, input_state = ?, job_id = ?, job_pub_idx = ?, counts = ? WHERE id = ?",
                    (num_vars, complexity, statement, input_state, job_id, job_pub_idx, counts, trial_id)
                )
            conn.commit()

    def delete(self, trial : Trial =None, trial_id=None):
        if trial is None and trial_id is None:
            raise ValueError("Either trial or trial_id must be provided")
        
        if trial is not None:
            assert trial_id is None or trial_id == trial.trial_id
            trial_id = trial.trial_id

        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM trials WHERE id = ?", (trial_id,))
            conn.commit()

    def all(self):
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trials")
            return self._as_trials(cursor.fetchall())
        
    def get(self, num_vars, complexity, statement=None, input_state=None, include_pending=False):
        query = "SELECT * FROM trials WHERE num_vars = ? AND complexity = ?"
        params = [num_vars, complexity]
        
        if statement is not None:
            query += " AND statement = ?"
            params.append(statement)
        
        if input_state is not None:
            query += " AND input_state = ?"
            params.append(input_state)

        if not include_pending:
            query += " AND NOT counts = ''"
        
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return self._as_trials(cursor.fetchall())
        
    
    def get_per_statement(self, num_vars, complexity):
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT statement FROM trials WHERE num_vars = ? AND complexity = ? AND NOT counts = ''", (num_vars, complexity))
            statements = cursor.fetchall()
            return {statement[0]: self.get(num_vars, complexity, statement=statement[0]) for statement in statements}
    
    def _use_job_results(self, job_id, results):
        """
        Given a job_id and the results of the job, this updates the counts for all trials with that job_id
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM trials WHERE job_id = ? AND counts = ''", (job_id,))
            trials = self._as_trials(cursor.fetchall())
            for trial in trials:
                trial.counts = results[trial.job_pub_idx].data.meas.get_counts()
                self.save(trial)

    async def use_job_results(self, job_id):
        """
        Given a job_id, this fetches the results of the job and updates the counts for all trials with that job_id
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
        Loads counts results for all trials that do not yet have them, saving the the updated trials to the database
        """
        query = "SELECT DISTINCT job_id FROM trials WHERE counts = ''"
        jobs = []
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            jobs = cursor.fetchall()
        
        # we could try to parallelize this, but this at least conserves memory
        for job in jobs:
            await self.use_job_results(job[0])


def get_oracle(statement, variables):
    # For now, we write the function to a file and import it then delete the file, since the classical function synthesis wants source code to work with
    temp_function = get_classical_function(statement, variables, name="temporary_function")
    required_imports = """
from qiskit.circuit.classicalfunction import classical_function
from qiskit.circuit.classicalfunction.types import Int1
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

        classical_function = temp_module.temporary_function
        oracle = classical_function.synth(registerless=False)

        return oracle

def run_benchmark_sample(num_vars, complexity, num_functions=100, num_inputs=100, shots=10**4, include_existing_trials=False):
    """
    Given a number of variables and complexity, this generates num_function random functions of that number of variables and complexity,
    compiles each into a quantum circuit, and runs the circuit on num_inputs random inputs.
    If include_existing_trials is True, this will only generate enough trials to bring the current count up to the specified amounts.
    For instance, if there are already trials for 20 functions with the given num_vars and complexity, this will only generate trials for 80 more.
    Returns a list of the new Trial objects created for this sample.

    Note: this does not wait for the quantum jobs to complete, but does the trials' metadata to the database. Call Trials().load_results() to update
    the database with the results of the jobs.
    """
    debug_print(f"Running benchmark sample for {num_vars} variables and complexity {complexity}")

    trials = Trials()
    if include_existing_trials:
        # count the number of distinct statements we already have trials for
        # this could be optimized with a direct SQL query, but it's probably fine like this too
        existing_trials = set([t.statement for t in trials.get(num_vars, complexity)])
        num_functions -= len(existing_trials)

    if num_functions <= 0:
        return []

    new_trials = []
    with Batch(backend,) as batch:
        sampler = Sampler() # backend and mode=batch are implicit inside context manager

        for batch_job_idx in range(num_functions):
            statement, variables = get_random_statement(num_vars, complexity)
            if statement is None:
                continue
            debug_print(f"Compiling statement: {statement}")
            oracle = get_oracle(statement, variables)
            circuits = []
            trials_minibatch = []
            for _ in range(num_inputs):
                qc = qiskit.QuantumCircuit(num_vars + 1) # +1 for result qubit
                inpt = [random.randint(0, 1) for _ in range(num_vars)]
                for i, bit in enumerate(inpt):
                    if bit:
                        qc.x(qc.qubits[i])

                qc.compose(oracle, inplace=True)
                qc.measure_all()
                qc_transpiled = qiskit.transpile(qc, backend=backend)
                job_pub_idx = len(circuits)
                circuits.append(qc_transpiled)

                # use job_id as None for now, we will update this once the job is run
                trials_minibatch.append(Trial(num_vars, complexity, None, job_pub_idx, ''.join(map(str, inpt)), statement, None))

            debug_print(f"Submitting minibatch of {num_inputs} circuits to backend. Job {batch_job_idx} of batch.")
            
            # run batch job of the oracle on each input
            job = sampler.run(circuits, shots=shots)
            job_id = job.job_id()

            # save the trials
            for trial in trials_minibatch:
                trial.job_id = job_id
                trials.save(trial)
                new_trials.append(trial)

    return new_trials
    