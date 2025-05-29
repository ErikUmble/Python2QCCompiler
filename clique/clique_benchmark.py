from enum import Enum
import math
import asyncio
import random
from dotenv import load_dotenv
import importlib.util
import json
import os
import sqlite3
import tempfile
import qiskit
from qiskit.circuit.library import grover_operator, QFT
from qiskit_ibm_runtime import QiskitRuntimeService, Batch, SamplerV2 as Sampler
from qiskit_aer import AerSimulator
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


from graph_database import Graphs, Graph, verify_clique

load_dotenv()
API_TOKEN = os.getenv("API_TOKEN")
API_INSTANCE = os.getenv("API_INSTANCE", None)
service = QiskitRuntimeService(
    channel="ibm_quantum", token=API_TOKEN, instance=API_INSTANCE
)
backend = service.backend(name="ibm_rensselaer")
graph_db = Graphs()

# DEBUG = os.getenv("DEBUG", False)
DEBUG = True


class CompileType:
    DIRECT = "DIRECT"
    CLASSICAL_FUNCTION = "CLASSICAL_FUNCTION"
    XAG = "XAG"


def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)


def num_grover_iterations(n, m):
    return math.floor(math.pi / (4 * math.asin(math.sqrt(m / 2**n))))


class Trial:
    def __init__(
        self,
        graph_id,
        compile_type,
        clique_size,
        grover_iterations,
        job_id,
        job_pub_idx,
        counts=None,
        simulation_counts=None,
        trial_id=None,
    ):
        if graph_id is None:
            raise ValueError("graph_id must be provided")

        self.trial_id = trial_id
        self.graph_id = graph_id
        self.compile_type = compile_type
        self.clique_size = clique_size
        self.grover_iterations = grover_iterations
        self.job_id = job_id
        self.job_pub_idx = job_pub_idx
        self.counts = counts
        self.simulation_counts = simulation_counts

        # cache the graph to avoid multiple database calls
        self.graph = graph_db.get(self.graph_id)[0]

    @property
    def expected_success_rate(self):
        graph = self.graph
        n = graph.n
        m = graph.clique_counts[self.clique_size]

        if m == 0:
            return 0

        q = (2 * m) / (2**n)
        theta = math.atan(math.sqrt(q * (2 - q)) / (1 - q))
        phi = math.atan(math.sqrt(q / (2 - q)))
        return math.sin(self.grover_iterations * theta + phi) ** 2

    @property
    def success_rate(self):
        if self.counts is None:
            raise ValueError(
                "Counts must be set before calculating success rate. Use get_counts() or get_counts_async() to update the counts."
            )

        graph = self.graph
        num_cliques_found = 0
        num_shots = 0
        for k, v in self.counts.items():
            if k == "-1":
                num_shots += v
                continue
            # reverse the bits to match the order in the graph
            if verify_clique(graph.g, k[::-1], self.clique_size):
                num_cliques_found += v
            num_shots += v

        return num_cliques_found / num_shots if num_shots > 0 else 0

    @property
    def simulation_success_rate(self):
        graph = self.graph
        num_cliques_found = 0
        num_shots = 0
        for k, v in self.simulation_counts.items():
            if k == "-1":
                num_shots += v
                continue

            if verify_clique(graph.g, k[::-1], self.clique_size):
                num_cliques_found += v
            num_shots += v

        return num_cliques_found / num_shots if num_shots > 0 else 0

    def mark_failure(self):
        if self.counts is not None:
            raise Exception("Cannot mark failure if counts are already set")
        self.counts = {"-1": 1}

    def as_dict(self):
        return {
            "job_id": self.job_id,
            "job_pub_idx": self.job_pub_idx,
            "graph_id": self.graph_id,
            "compile_type": self.compile_type,
            "clique_size": self.clique_size,
            "grover_iterations": self.grover_iterations,
            "trial_id": self.trial_id,
            "counts": self.counts,
            "simulation_counts": self.simulation_counts,
        }

    def set_counts(self, counts, trials=None):
        """
        call this once the job has finished running

        this updates the trials entry trials is provided and the entry exists
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
    def __init__(self, db_name="graphs.db"):
        self.db_name = db_name
        self._initialize_database()

    def _connect(self):
        return sqlite3.connect(self.db_name)

    def _as_trials(self, rows: list):
        return [
            Trial(
                graph_id=row[1],
                compile_type=row[2],
                clique_size=row[3],
                grover_iterations=row[4],
                job_id=row[5],
                job_pub_idx=row[6],
                counts=json.loads(row[7]) if row[7] else None,
                simulation_counts=json.loads(row[8]) if row[8] else None,
                trial_id=row[0],
            )
            for row in rows
        ]

    def _initialize_database(self):
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS clique_trials (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    graph_id INTEGER,
                    compile_type TEXT,
                    clique_size INTEGER,
                    grover_iterations INTEGER,
                    job_id TEXT,
                    job_pub_idx INTEGER,
                    counts TEXT,
                    simulation_counts TEXT,
                    FOREIGN KEY(graph_id) REFERENCES graphs(id)
                )
                """
            )
            conn.commit()

    def save(self, trial: Trial):
        graph_id = trial.graph_id
        compile_type = trial.compile_type
        clique_size = trial.clique_size
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
                    "INSERT INTO clique_trials (graph_id, compile_type, clique_size, grover_iterations, job_id, job_pub_idx, counts, simulation_counts) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        graph_id,
                        compile_type,
                        clique_size,
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
                    "UPDATE clique_trials SET graph_id = ?, compile_type = ?, clique_size = ?, grover_iterations = ?, job_id = ?, job_pub_idx = ?, counts = ?, simulation_counts =? WHERE id = ?",
                    (
                        graph_id,
                        compile_type,
                        clique_size,
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
            cursor.execute("DELETE FROM clique_trials WHERE id = ?", (trial_id,))
            conn.commit()

    def all(self):
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM clique_trials")
            return self._as_trials(cursor.fetchall())

    def get(
        self,
        graph_id=None,
        graph=None,
        n=None,
        compile_type=None,
        clique_size=None,
        grover_iterations=None,
        job_id=None,
        include_pending=False,
        trial_id=None,
    ):  # Added trial_id
        """
        Retrieves trials from the database based on specified criteria.

        If trial_id is provided, it fetches the specific trial by its ID,
        ignoring other filter parameters. Otherwise, it filters based on
        the other provided parameters.

        Args:
            graph_id (Optional[int]): Filter by graph_id.
            graph (Optional[Union[Graph, Any]]): Filter by graph object or its representation.
                                                 If a Graph object, its 'g' attribute is used.
            n (Optional[int]): Filter by the number of nodes in the graph.
            compile_type (Optional[str]): Filter by compile_type.
            clique_size (Optional[int]): Filter by clique_size.
            grover_iterations (Optional[int]): Filter by grover_iterations.
            job_id (Optional[str]): Filter by job_id.
            include_pending (bool): If False (default), trials with no 'counts' (empty string)
                                    are excluded. If True, all trials are included.
            trial_id (Optional[int]): Fetch a specific trial by its primary ID.

        Returns:
            List[Trial]: A list of Trial objects matching the criteria.
        """
        params = []
        query_parts = []
        graph_query_parts = []

        if trial_id is not None:
            # If trial_id is provided, construct a query to fetch by primary key.
            # Other filters are ignored in this case.
            query = "SELECT * FROM clique_trials WHERE id = ?"
            params.append(trial_id)
        else:
            # Existing logic for building query based on other parameters
            if graph_id is not None:
                query_parts.append(
                    "clique_trials.graph_id = ?"
                )  # Prefixed with table name for clarity in joins
                params.append(graph_id)

            if compile_type is not None:
                query_parts.append("clique_trials.compile_type = ?")
                params.append(compile_type)

            if clique_size is not None:
                query_parts.append("clique_trials.clique_size = ?")
                params.append(clique_size)

            if grover_iterations is not None:
                query_parts.append("clique_trials.grover_iterations = ?")
                params.append(grover_iterations)

            if job_id is not None:
                query_parts.append("clique_trials.job_id = ?")
                params.append(job_id)

            if not include_pending:
                # Exclude trials where 'counts' is an empty string (convention for pending/no counts)
                query_parts.append("NOT clique_trials.counts = ?")
                params.append("")

            if graph is not None:
                # Assuming Graph is a defined class with a 'g' attribute
                # if isinstance(graph, Graph): # Replace 'Graph' with the actual class name
                #     graph_data = graph.g
                # else:
                #     graph_data = graph # Assuming 'graph' can also be the direct data
                graph_data = (
                    graph.g
                    if hasattr(graph, "g") and not isinstance(graph, str)
                    else graph
                )  # More robust check
                graph_query_parts.append("graphs.g = ?")
                params.append(graph_data)

            if n is not None:
                graph_query_parts.append("graphs.n = ?")
                params.append(n)

            # Determine the query structure based on whether graph-related filters are present
            if len(graph_query_parts) > 0:
                base_query = "SELECT clique_trials.* FROM clique_trials JOIN graphs ON clique_trials.graph_id = graphs.id"
                all_conditions = query_parts + graph_query_parts
                if all_conditions:
                    query = f"{base_query} WHERE " + " AND ".join(all_conditions)
                else:
                    query = (
                        base_query  # Join but no specific WHERE conditions from params
                    )
            else:  # No graph_query_parts, so no JOIN needed unless forced by other logic
                base_query = "SELECT * FROM clique_trials"
                if query_parts:
                    query = f"{base_query} WHERE " + " AND ".join(query_parts)
                else:
                    # No trial_id, no query_parts, no graph_query_parts means select all
                    query = base_query

        with self._connect() as conn:
            cursor = conn.cursor()
            # print(f"Executing query: {query} with params: {params}") # For debugging
            cursor.execute(query, params)
            return self._as_trials(cursor.fetchall())

    def _use_job_results(self, job_id, results):
        """
        Given a job_id and the results of the job, this updates the counts for all trials with that job_id
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM clique_trials WHERE job_id = ? AND counts = ''",
                (job_id,),
            )
            trials = self._as_trials(cursor.fetchall())
            for trial in trials:
                trial.counts = results[trial.job_pub_idx].data.c.get_counts()
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
        query = "SELECT DISTINCT job_id FROM clique_trials WHERE counts = ''"
        jobs = []
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            jobs = cursor.fetchall()

        # we could try to parallelize this, but this at least conserves memory
        for job in jobs:
            await self.use_job_results(job[0])


class SortPairNode:
    def __init__(self, high, low):
        self.high = high
        self.low = low


def get_sort_statements(variables):
    num_variables = len(variables)
    statements = []

    nodes = [
        [SortPairNode(None, None) for _ in range(num_variables)]
        for _ in range(num_variables)
    ]
    for i in range(num_variables):
        nodes[i][0] = SortPairNode(variables[i], None)

    for i in range(1, num_variables):
        for j in range(1, i + 1):
            s_high = f"s_{i}_{j}_high"
            s_low = f"s_{i}_{j}_low"
            nodes[i][j] = SortPairNode(s_high, s_low)

            if j == i:
                statements.append(
                    f"{s_high} = {nodes[i - 1][j - 1].high} or {nodes[i][j - 1].high}"
                )
                statements.append(
                    f"{s_low} = {nodes[i - 1][j - 1].high} and {nodes[i][j - 1].high}"
                )
            else:
                statements.append(
                    f"{s_high} = {nodes[i - 1][j].low} or {nodes[i][j - 1].high}"
                )
                statements.append(
                    f"{s_low} = {nodes[i - 1][j].low} and {nodes[i][j - 1].high}"
                )

    outputs = [nodes[num_variables - 1][num_variables - 1].high] + [
        nodes[num_variables - 1][i].low for i in range(num_variables - 1, 0, -1)
    ]

    return statements, outputs1


def get_variables(num_vars):
    return ["x" + str(i) for i in range(num_vars)]


def construct_clique_verifier(
    graph: str, as_classical_function=False, clique_size=None
):
    """
    Given a graph in the form of binary string
    e_11 e_12 e_13 ... e_1n e_23 e_24 ... e_2n ... e_n-1n, returns the string of a python function that takes n boolean variables denoting vertices
    True if in the clique and False if not,
    and returns whether the input is a clique of size at least n/2 in the graph.

    if clique_size is unspecified, the default is to require at least n/2 vertices
    """
    n = int((1 + (1 + 8 * len(graph)) ** 0.5) / 2)
    variables = get_variables(n)
    statements, sort_outputs = get_sort_statements(variables)
    clique_size = clique_size or n // 2

    # count whether there are at least clique_size vertices in the clique
    statements.append("count = " + " and ".join(o for o in sort_outputs[:clique_size]))

    # whenever there is not an edge between two vertices, they cannot both be in the clique
    statements.append(
        f"edge_sat = {variables[0]} or not {variables[0]}"
    )  # this should be initialized to True, but qiskit classical function cannot yet parse True
    edge_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge = graph[edge_idx]
            edge_idx += 1
            if edge == "0":
                # TODO: we could reduce depth to log instead of linear by applying AND more efficiently
                # for now, we'll let tweedledum optimize this
                statements.append(
                    f"edge_sat = edge_sat and not ({variables[i]} and {variables[j]})"
                )

    statements.append("return count and edge_sat")
    if as_classical_function:
        output = (
            "@classical_function\ndef is_clique("
            + ", ".join([f"{v} : Int1" for v in variables])
            + ") -> Int1:\n    "
        )
    else:
        output = "def is_clique(" + ", ".join(variables) + "):\n    "
    output += "\n    ".join(statements)
    return output


def _classical_function_to_oracle(function_string):
    """
    given a classical function in string form (such as the output of get_classical_function), returns a quantum oracle circuit
    for that function
    """
    # For now, we write the function to a file and import it then delete the file, since the classical function synthesis wants source code to work with
    function_name = function_string.split("(")[0].split("def")[1].strip()
    required_imports = """
from qiskit.circuit.classicalfunction import classical_function
from qiskit.circuit.classicalfunction.types import Int1
"""
    with tempfile.TemporaryDirectory() as temp_dir:
        module_name = "temp_boolean_func"
        file_path = os.path.join(temp_dir, f"{module_name}.py")

        with open(file_path, "w") as f:
            f.write(required_imports)
            f.write(function_string)

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        temp_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(temp_module)

        classical_function = getattr(temp_module, function_name)
        oracle = classical_function.synth(registerless=False)

        return oracle


def run_grover(oracle, n, grover_iterations, shots=10**4):
    """
    Given oracle U_f that has m solutions, this runs a Grover's search circuit using U_f
    and returns the job_id, simulation_counts of the job that was submitted to the backend.
    """
    # this assertion does NOT work for XAG compiler
    # assert oracle.num_qubits in [n, n+1]
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
            optimization_level=3, backend=simulator
        )
        qc = pass_manager.run(search_circuit)
        result = simulator.run(qc, shots=shots).result()
        simulation_counts = result.get_counts()
    except Exception as e:
        debug_print(f"Error {e} while creating simulator circuit and running")

    return job.job_id(), simulation_counts


def _clique_size_to_search_for(graph: Graph, target_grover_iterations: int):
    """
    Given a graph and a target number of grover iterations, this returns the largest clique size
    to search for such that the number of iterations is equal to the target
    """
    clique_size = 1
    while (
        clique_size < len(graph.clique_counts)
        and num_grover_iterations(graph.n, graph.clique_counts[clique_size])
        < target_grover_iterations
    ):
        clique_size += 1

    return clique_size - 1


def clique_oracle_compiler_classical_function(graph: str, clique_size):
    return _classical_function_to_oracle(
        construct_clique_verifier(
            graph, as_classical_function=True, clique_size=clique_size
        )
    )


def run_benchmark_sample(
    graph: Graph,
    compile_type,
    clique_oracle,
    clique_size,
    grover_iterations,
    shots=10**4,
    include_existing_trials=False,
):
    """
    Given a number of variables and complexity, this generates num_function random functions of that number of variables and complexity,
    compiles each into a quantum circuit, and runs the circuit on num_inputs random inputs.
    If include_existing_trials is True, this will only generate enough trials to bring the current count up to the specified amounts.
    For instance, if there are already trials for 20 functions with the given num_vars and complexity, this will only generate trials for 80 more.
    Returns a list of the new Trial objects created for this sample.
    circuits_per_job specifies the maximum number of circuits to submit to a single job, if greater than or equal to num_inputs, then there will
    be a single job per function, if less then there will be multiple jobs per function.

    Note: this does not wait for the quantum jobs to complete, but does the trials' metadata to the database. Call Trials().load_results() to update
    the database with the results of the jobs.
    """
    debug_print(f"Running benchmark sample for {graph}")

    trials = Trials()

    if include_existing_trials:
        if (
            len(
                trials.get(
                    graph=graph,
                    clique_size=clique_size,
                    grover_iterations=grover_iterations,
                    include_pending=True,
                )
            )
            > 0
        ):
            return

    debug_print(
        f"Looking for a clique with {clique_size} vertices using {grover_iterations} iterations"
    )
    job_id, simulation_counts = run_grover(
        clique_oracle, graph.n, grover_iterations, shots=shots
    )

    trial = Trial(
        graph_id=graph.graph_id,
        compile_type=compile_type,
        clique_size=clique_size,
        grover_iterations=grover_iterations,
        job_id=job_id,
        job_pub_idx=0,
        simulation_counts=simulation_counts,
    )

    # XAG can produce very large graphs that the simulator cannot run.
    # This should be considered a failure and the trial is marked as such
    # simulation_counts is None if the simulator failed
    if simulation_counts is None:
        trial.mark_failure()

    trials.save(trial)
    return trial


def mark_job_failure(job_id):
    trials = Trials()
    for trial in trials.get(job_id=job_id):
        trial.mark_failure()
        trials.save(trial)


def create_compilation_failure_trial(num_vars, complexity, statement, trials=None):
    trial = Trial(
        num_vars=num_vars,
        complexity=complexity,
        job_id="_compilation_failed_",
        job_pub_idx=0,
        statement=statement,
    )
    trial.mark_failure()
    if trials is not None:
        trials.save(trial)

    return trial
