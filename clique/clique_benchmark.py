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
service = QiskitRuntimeService(channel="ibm_quantum", token=API_TOKEN, instance=API_INSTANCE)
backend = service.backend(name="ibm_rensselaer")
graph_db = Graphs()

DEBUG = os.getenv("DEBUG", False)

class CompileType:
    DIRECT = "DIRECT"
    CLASSICAL_FUNCTION = "CLASSICAL_FUNCTION"
    XAG = "XAG"

def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

def num_grover_iterations(n, m):
    return math.floor(
        math.pi / (4 * math.asin(math.sqrt(m / 2 ** n)))
    )

class Trial:
    def __init__(self, graph_id, compile_type, clique_size, grover_iterations, job_id, job_pub_idx, counts=None, simulation_counts=None, trial_id=None):
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

        q = (2*m) / (2**n)
        theta = math.atan(math.sqrt(q*(2-q))/(1-q))
        phi = math.atan(math.sqrt(q/(2-q)))
        return math.sin(self.grover_iterations * theta + phi)**2
    
    @property
    def success_rate(self):
        if self.counts is None:
            raise ValueError("Counts must be set before calculating success rate. Use get_counts() or get_counts_async() to update the counts.")
        
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
        self.counts = {"-1":1}

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
            "simulation_counts": self.simulation_counts
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
            counts = result[self.job_pub_idx].data.c.get_counts()
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
        return [Trial(
            graph_id=row[1],
            compile_type=row[2],
            clique_size=row[3],
            grover_iterations=row[4],
            job_id=row[5],
            job_pub_idx=row[6],
            counts=json.loads(row[7]) if row[7] else None,
            simulation_counts=json.loads(row[8]) if row[8] else None,
            trial_id=row[0]
        ) for row in rows]
        

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
        simulation_counts = json.dumps(trial.simulation_counts) if trial.simulation_counts is not None else ""
        trial_id = trial.trial_id

        with self._connect() as conn:
            cursor = conn.cursor()
            if trial_id is None:
                cursor.execute(
                    "INSERT INTO clique_trials (graph_id, compile_type, clique_size, grover_iterations, job_id, job_pub_idx, counts, simulation_counts) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (graph_id, compile_type, clique_size, grover_iterations, job_id, job_pub_idx, counts, simulation_counts)
                )
                trial.trial_id = cursor.lastrowid
            else:
                cursor.execute(
                    "UPDATE clique_trials SET graph_id = ?, compile_type = ?, clique_size = ?, grover_iterations = ?, job_id = ?, job_pub_idx = ?, counts = ?, simulation_counts =? WHERE id = ?",
                    (graph_id, compile_type, clique_size, grover_iterations, job_id, job_pub_idx, counts, simulation_counts, trial_id)
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
            cursor.execute("DELETE FROM clique_trials WHERE id = ?", (trial_id,))
            conn.commit()

    def all(self):
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM clique_trials")
            return self._as_trials(cursor.fetchall())
        
    def get(self, graph_id=None, graph=None, n=None, compile_type=None, clique_size=None, grover_iterations=None, job_id=None, include_pending=False):
        
        params = []
        query_parts = []
        graph_query_parts = []

        if graph_id is not None:
            query_parts.append("graph_id = ?")
            params.append(graph_id)            
            
        if compile_type is not None:
            query_parts.append("compile_type = ?")
            params.append(compile_type)

        if clique_size is not None:
            query_parts.append("clique_size = ?")
            params.append(clique_size)

        if grover_iterations is not None:
            query_parts.append("grover_iterations = ?")
            params.append(grover_iterations)

        if job_id is not None:
            query_parts.append("job_id = ?")
            params.append(job_id)

        if not include_pending:
            query_parts.append("NOT counts = ?")
            params.append("")

        if graph is not None:
            if isinstance(graph, Graph):
                graph = graph.g
            graph_query_parts.append("graphs.g = ?")
            params.append(graph)

        if n is not None:
            graph_query_parts.append("graphs.n = ?")
            params.append(n)


        # handle the different cases for joins and/or filtering
        if len(graph_query_parts) == 0 and len(query_parts) == 0:
            query = "SELECT * FROM clique_trials"

        elif len(graph_query_parts) > 0 and len(query_parts) > 0:
            query = "SELECT clique_trials.* FROM clique_trials JOIN graphs ON clique_trials.graph_id = graphs.id WHERE " + " AND ".join(query_parts + graph_query_parts)

        elif len(graph_query_parts) > 0:
            query = "SELECT clique_trials.* FROM clique_trials JOIN graphs ON clique_trials.graph_id = graphs.id WHERE " + " AND ".join(graph_query_parts)

        else:
            query = "SELECT * FROM clique_trials WHERE " + " AND ".join(query_parts)
            
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return self._as_trials(cursor.fetchall())
    
    def _use_job_results(self, job_id, results):
        """
        Given a job_id and the results of the job, this updates the counts for all trials with that job_id
        """
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM clique_trials WHERE job_id = ? AND counts = ''", (job_id,))
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
        
        debug_print(f"Found {len(jobs)} jobs with missing results")
        tasks = []
        for job_id in [job[0] for job in jobs]:
            tasks.append(self.use_job_results(job_id))
        
        # process tasks in batches to avoid overwhelming the API or memory
        batch_size = 20
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            debug_print(f"Processing batch {i//batch_size + 1} of {(len(tasks) + batch_size - 1)//batch_size}")
            await asyncio.gather(*batch)


class SortPairNode:
    def __init__(self, high, low):
        self.high = high
        self.low = low

def get_sort_statements(variables):
    num_variables = len(variables)
    statements = []

    nodes = [[SortPairNode(None, None) for _ in range(num_variables)] for _ in range(num_variables)]
    for i in range(num_variables):
        nodes[i][0] = SortPairNode(variables[i], None)

    for i in range(1, num_variables):
        for j in range(1, i+1):
            s_high = f"s_{i}_{j}_high"
            s_low = f"s_{i}_{j}_low"
            nodes[i][j] = SortPairNode(s_high, s_low)

            if j == i:
                statements.append(f"{s_high} = {nodes[i-1][j-1].high} or {nodes[i][j-1].high}")
                statements.append(f"{s_low} = {nodes[i-1][j-1].high} and {nodes[i][j-1].high}")
            else:
                statements.append(f"{s_high} = {nodes[i-1][j].low} or {nodes[i][j-1].high}")
                statements.append(f"{s_low} = {nodes[i-1][j].low} and {nodes[i][j-1].high}")

    outputs = [nodes[num_variables-1][num_variables-1].high] + [nodes[num_variables-1][i].low for i in range(num_variables-1, 0, -1)]

    return statements, outputs

def get_variables(num_vars):
    return ["x" + str(i) for i in range(num_vars)]

def construct_clique_verifier(graph : str, as_classical_function=False, clique_size=None):
    """ 
    Given a graph in the form of binary string 
    e_11 e_12 e_13 ... e_1n e_23 e_24 ... e_2n ... e_n-1n, returns the string of a python function that takes n boolean variables denoting vertices 
    True if in the clique and False if not,
    and returns whether the input is a clique of size at least n/2 in the graph.

    if clique_size is unspecified, the default is to require at least n/2 vertices
    """
    n = int((1 + (1 + 8*len(graph))**0.5) / 2)
    variables = get_variables(n)
    statements, sort_outputs = get_sort_statements(variables)
    clique_size = clique_size or n//2

    # count whether there are at least clique_size vertices in the clique
    statements.append("count = " + sort_outputs[clique_size-1])

    # whenever there is not an edge between two vertices, they cannot both be in the clique
    statements.append(f"edge_sat = {variables[0]} or not {variables[0]}") # this should be initialized to True, but qiskit classical function cannot yet parse True
    edge_idx = 0
    for i in range(n):
        for j in range(i+1, n):
            edge = graph[edge_idx]
            edge_idx += 1
            if edge == '0':
                # TODO: we could reduce depth to log instead of linear by applying AND more efficiently
                # for now, we'll let tweedledum optimize this
                statements.append(f"edge_sat = edge_sat and not ({variables[i]} and {variables[j]})")

    statements.append("return count and edge_sat")
    if as_classical_function:
        output = "@classical_function\ndef is_clique(" + ", ".join([f"{v} : Int1" for v in variables]) + ") -> Int1:\n    "
    else:
        output = "def is_clique(" + ", ".join(variables) + "):\n    "
    output += "\n    ".join(statements)
    return output

def direct_clique_oracle_circuit(graph, clique_size=None):
    """ 
    Given a graph in the form of binary string 
    e_11 e_12 e_13 ... e_1n e_23 e_24 ... e_2n ... e_n-1n, returns a quantum oracle circuit for the 
    verifier function of such a clique.

    if clique_size is unspecified, the default is to require at least n/2 vertices
    """
    n = int((1 + (1 + 8*len(graph))**0.5) / 2)
    ret_qubit = n
    edge_sat_qubit = n + 1
    count_sat_qubit = n + 2
    variables = get_variables(n)
    statements, sort_outputs = get_sort_statements(variables)
    clique_size = clique_size or n//2
    assert clique_size >= 1

    # map variable names to qubit indices
    var_map = {}
    for i in range(n):
        var_map[variables[i]] = i

    num_sort_temps = len(statements) - 1
    num_missing_edges = len(list(filter(lambda x: x == '0', graph)))

    qc = qiskit.QuantumCircuit(n + 3 + num_missing_edges + num_sort_temps, n)
    operations = []

    # whenever there is not an edge between two vertices, they cannot both be in the clique
    edge_idx = 0
    qubit_idx = n+3
    for i in range(n):
        for j in range(i+1, n):
            edge = graph[edge_idx]
            edge_idx += 1
            if edge == '0':
                operations.append((qc.mcx, [i, j], qubit_idx))
                qubit_idx += 1
    for i in range(n+3, n+3+num_missing_edges):
        operations.append((qc.x, [i], None))

    if num_missing_edges > 0:
        operations.append((qc.mcx, [i for i in range(n+3, n+3+num_missing_edges)], edge_sat_qubit))
    else:
        operations.append((qc.x, edge_sat_qubit))

    # count whether there are at least clique_size vertices in the clique
    for s in statements:
        var_map[s.split('=')[0].strip()] = qubit_idx
        qubit_idx += 1

    var_map[sort_outputs[clique_size-1]] = count_sat_qubit

    for s in statements:
        res = var_map[s.split('=')[0].strip()]
        if "or" in s:
            var1, var2 = s.split('=')[1].split('or')
            var1 = var_map[var1.strip()]
            var2 = var_map[var2.strip()]
            operations.append((qc.x, var1))
            operations.append((qc.x, var2))
            operations.append((qc.mcx, [var1, var2], res))
            operations.append((qc.x, res))
            operations.append((qc.x, var1))
            operations.append((qc.x, var2))
            continue

        elif "and" in s:
            var1, var2 = s.split('=')[1].split('and')
            var1 = var_map[var1.strip()]
            var2 = var_map[var2.strip()]
            operations.append((qc.mcx, [var1, var2], res))
            continue

    # apply operations in forward order
    for i in range(len(operations)):
        op = operations[i][0]
        op(*operations[i][1:])
    qc.mcx([edge_sat_qubit, count_sat_qubit], ret_qubit)
    # apply operations in reverse order
    for i in range(len(operations)-1, -1, -1):
        op = operations[i][0]
        op(*operations[i][1:])
    return qc


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
    #assert oracle.num_qubits in [n, n+1]
    uf_mode = oracle.num_qubits >= n+1
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

    try:
        simulator = AerSimulator(n_qubits=search_circuit.num_qubits)
        pass_manager = generate_preset_pass_manager(optimization_level=1, backend=simulator)
        qc = pass_manager.run(search_circuit)
        result = simulator.run(qc,shots=shots).result()
        simulation_counts = result.get_counts()
    except Exception as e:
        print(f"Error running simulation: {e}")
        simulation_counts = None

    return job.job_id(), simulation_counts

def _clique_size_to_search_for(graph: Graph, target_grover_iterations: int):
    """
    Given a graph and a target number of grover iterations, this returns the largest clique size
    to search for such that the number of iterations is equal to the target
    """
    clique_size = 1
    while clique_size < len(graph.clique_counts) and num_grover_iterations(graph.n, graph.clique_counts[clique_size]) < target_grover_iterations:
        clique_size += 1
    
    return clique_size - 1

def clique_oracle_compiler_classical_function(graph: str, clique_size):
    return _classical_function_to_oracle(construct_clique_verifier(graph, as_classical_function=True, clique_size=clique_size))

def run_benchmark_sample(graph : Graph, compile_type, clique_oracle, clique_size, grover_iterations, shots=10**4, include_existing_trials=False):
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
        if len(trials.get(graph=graph, compile_type=compile_type, clique_size=clique_size, grover_iterations=grover_iterations, include_pending=True)) > 0:
            return
        
    debug_print(f"Looking for a clique with {clique_size} vertices using {grover_iterations} iterations")
    job_id, simulation_counts = run_grover(clique_oracle, graph.n, grover_iterations, shots=shots)
        
    trial = Trial(
        graph_id=graph.graph_id,
        compile_type=compile_type,
        clique_size=clique_size,
        grover_iterations=grover_iterations,
        job_id=job_id,
        job_pub_idx=0,
        simulation_counts=simulation_counts
    )
    trials.save(trial)
    return trial
    

def mark_job_failure(job_id):
    trials = Trials()
    for trial in trials.get(job_id=job_id):
        trial.mark_failure()
        trials.save(trial)
        
def create_compilation_failure_trial(num_vars, complexity, statement, trials=None):
    trial = Trial(num_vars=num_vars, complexity=complexity, job_id="_compilation_failed_", job_pub_idx=0, statement=statement)
    trial.mark_failure()
    if trials is not None:
        trials.save(trial)

    return trial


