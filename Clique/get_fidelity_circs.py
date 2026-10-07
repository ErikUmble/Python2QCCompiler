import os
import json
import sys
import qiskit

from sqlalchemy import select, case
from sqlalchemy.orm import joinedload

from benchmarklib import BenchmarkDatabase
from clique import CliqueOracleTrial, CliqueProblem
from benchmarklib.compilers import XAGCompiler, QCFCompiler
from benchmarklib.runners.resource_management import run_with_resource_limits



benchmark_db = BenchmarkDatabase("clique.db", CliqueProblem, CliqueOracleTrial)
basis_gates = ["cx", "sx", "rz", "x", "h"]


def compile_problem(problem: CliqueProblem, compiler: "SynthesisCompiler", input_state: str) -> qiskit.QuantumCircuit:
    oracle = problem.oracle(compiler.name)
    qc = qiskit.QuantumCircuit(max(oracle.num_qubits, problem.nodes + 1), problem.nodes + 1)
    for i, bit in enumerate(input_state):
        if bit == '1':
            qc.x(qc.qubits[i])

    qc.compose(oracle, inplace=True)
    qc.measure(range(problem.nodes + 1), range(problem.nodes + 1))

    # just transform to basis gates
    qc = qiskit.transpile(qc, basis_gates=basis_gates)

    return qc

def get_save_data(trial: CliqueOracleTrial):
    if trial.circuit_pretranspile is None:
        return None

    expected_measurement = ("1" if trial.expected_output else "0") + trial.input_state[::-1]
    
    if trial.counts is None:
        return {
            "id": trial.id,
            "qasm": qiskit.qasm3.dumps(trial.circuit_pretranspile),
            "expected_measurement": expected_measurement,
            "fidelity": None,
            "counts": None,
            "job_id": trial.job_id,
            "job_pub_idx": trial.job_pub_idx
        }
    
    return {
        "id": trial.id,
        "qasm": qiskit.qasm3.dumps(trial.circuit_pretranspile),
        "expected_measurement": expected_measurement,
        "fidelity": trial.calculate_success_rate(),
        "counts": trial.counts,
        "job_id": trial.job_id,
        "job_pub_idx": trial.job_pub_idx
    }


problem_ids = benchmark_db.query(
    select(CliqueOracleTrial.problem_id).where(CliqueOracleTrial._circuit_pretranspile_qpy != None).distinct()
)

def collect(min_id):
    data = []

    try:
        os.mkdir("tmp")
    except FileExistsError:
        pass

    for problem in benchmark_db.query(select(CliqueProblem).where(CliqueProblem.id.in_(problem_ids)).where(CliqueProblem.id >= min_id).order_by(CliqueProblem.id)):
        
        trials = benchmark_db.query(
            select(CliqueOracleTrial)
            .where(CliqueOracleTrial.problem_id == problem.id, CliqueOracleTrial._circuit_pretranspile_qpy != None)
            # order so that any trials saved with circuit already show up first to avoid recompiling when possible
            #.order_by(case((RandomBooleanFunctionTrial._circuit_pretranspile_qpy == None, 1), else_=0))
            .limit(1)
            .options(joinedload(CliqueOracleTrial.problem))
        )


        if len(trials) == 0:
            print(f"No trials found for problem {problem.id}, skipping")
            continue

        trial = trials[0]

        if trial.is_failed:
            print(f"Trial already failed for problem {problem.id}, skipping")
            continue

        ops = trial.circuit_pretranspile.count_ops()

        if ops.get("cx", 0) == 0:
            # skip if there is no entanglement in either (this implies trivially 0 function)
            #print(f"No entanglement in either circuit for problem {problem.id}: {problem.statement}: {ops1}, skipping")
            continue

        data.append(get_save_data(trial))

        if len(data) >= 100:
            with open(os.path.join("tmp", f"fidelity_circs_{problem.id}.json"), "w") as f:
                json.dump(data, f)
            print(f"Saved data from problems {min_id} to {problem.id}")
            data = []
            min_id = problem.id + 1

    if len(data) > 0:
        with open(os.path.join("tmp", f"fidelity_circs_{min_id}.json"), "w") as f:
            json.dump(data, f)
        print(f"Saved data from problems {min_id} to {problem.id}")

def combine(dest_filepath):
    combined_data = []
    for filename in os.listdir("tmp"):
        if filename.startswith("fidelity_circs_") and filename.endswith(".json"):
            with open(os.path.join("tmp", filename), "r") as f:
                data = json.load(f)
                combined_data.extend(data)
    
    with open(dest_filepath, "w") as f:
        json.dump(combined_data, f)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        action = "collect"
    else:
        action = sys.argv[1]

    if action == "collect":
        min_id = 0
        if len(sys.argv) >= 3:
            min_id = int(sys.argv[2])
        collect(min_id)
    else:
        dest_filepath = "fidelity_circs.json"
        if len(sys.argv) >= 3:
            dest_filepath = sys.argv[2]
        combine(dest_filepath)