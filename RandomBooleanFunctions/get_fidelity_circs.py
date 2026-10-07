import os
import json
import sys
import qiskit

from sqlalchemy import select, case
from sqlalchemy.orm import joinedload

from benchmarklib import BenchmarkDatabase
from rbf import RandomBooleanFunctionTrial, RandomBooleanFunction
from benchmarklib.compilers import XAGCompiler, QCFCompiler
from benchmarklib.runners.resource_management import run_with_resource_limits



benchmark_db = BenchmarkDatabase("rbf.db", RandomBooleanFunction, RandomBooleanFunctionTrial)
basis_gates = ["cx", "sx", "rz", "x", "h"]


def compile_problem(problem: RandomBooleanFunction, compiler: "SynthesisCompiler", input_state: str) -> qiskit.QuantumCircuit:
    oracle = problem.oracle(compiler.name)
    qc = qiskit.QuantumCircuit(max(oracle.num_qubits, problem.num_vars + 1), problem.num_vars + 1)
    for i, bit in enumerate(input_state):
        if bit == '1':
            qc.x(qc.qubits[i])

    qc.compose(oracle, inplace=True)
    qc.measure(range(problem.num_vars + 1), range(problem.num_vars + 1))

    # just transform to basis gates
    qc = qiskit.transpile(qc, basis_gates=basis_gates)

    return qc

def get_save_data(trial: RandomBooleanFunctionTrial):
    if trial.circuit_pretranspile is None:
        return None
    
    if trial.counts is None:
        return {
            "id": trial.id,
            "qasm": qiskit.qasm3.dumps(trial.circuit_pretranspile),
            "expected_measurement": trial.total_expected_results(),
            "fidelity": None,
            "counts": None,
            "job_id": trial.job_id,
            "job_pub_idx": trial.job_pub_idx,
            "created_at": trial.created_at.isoformat()
        }
    
    return {
        "id": trial.id,
        "qasm": qiskit.qasm3.dumps(trial.circuit_pretranspile),
        "expected_measurement": trial.total_expected_results(),
        "fidelity": trial.calculate_success_rate(),
        "counts": trial.counts,
        "job_id": trial.job_id,
        "job_pub_idx": trial.job_pub_idx,
        "created_at": trial.created_at.isoformat()
    }



cf_compiler = QCFCompiler()
xag_compiler = XAGCompiler()

def collect(min_id):
    data = []

    try:
        os.mkdir("tmp")
    except FileExistsError:
        pass

    for problem in benchmark_db.query(select(RandomBooleanFunction).where(RandomBooleanFunction.id >= min_id).order_by(RandomBooleanFunction.id)):
        
        cf_trials = benchmark_db.query(
            select(RandomBooleanFunctionTrial)
            .where(RandomBooleanFunctionTrial.problem_id == problem.id)
            .where(RandomBooleanFunctionTrial.compiler_name == "CLASSICAL_FUNCTION")
            # order so that any trials saved with circuit already show up first to avoid recompiling when possible
            #.order_by(case((RandomBooleanFunctionTrial._circuit_pretranspile_qpy == None, 1), else_=0))
            .limit(1)
            .options(joinedload(RandomBooleanFunctionTrial.problem))
        )

        xag_trials = benchmark_db.query(
            select(RandomBooleanFunctionTrial)
            .where(RandomBooleanFunctionTrial.problem_id == problem.id)
            .where(RandomBooleanFunctionTrial.compiler_name == "XAG")
            # order so that any trials saved with circuit already show up first to avoid recompiling when possible
            #.order_by(case((RandomBooleanFunctionTrial._circuit_pretranspile_qpy == None, 1), else_=0))
            .limit(1)
            .options(joinedload(RandomBooleanFunctionTrial.problem))
        )

        if len(cf_trials) == 0 and len(xag_trials) == 0:
            print(f"No trials found for problem {problem.id}, skipping")
            continue

        if len(cf_trials) == 0:
            cf_trial = RandomBooleanFunctionTrial(problem=problem, compiler_name="CLASSICAL_FUNCTION", input_state=xag_trials[0].input_state)
        else:
            cf_trial = cf_trials[0]

        if len(xag_trials) == 0:
            xag_trial = RandomBooleanFunctionTrial(problem=problem, compiler_name="XAG", input_state=cf_trials[0].input_state)
        else:
            xag_trial = xag_trials[0]

        if cf_trial.is_failed or xag_trial.is_failed:
            print(f"Trial already failed for problem {problem.id}, skipping")
            continue

        compiler_run = run_with_resource_limits(
            compile_problem,
            kwargs={"problem": problem, "compiler": cf_compiler, "input_state": cf_trial.input_state},
            memory_limit_mb=2024,
            timeout_seconds = 120,
            silence_output=True
        )
        cf_trial.circuit_pretranspile = compiler_run.result if compiler_run.success else None

        compiler_run = run_with_resource_limits(
            compile_problem,
            kwargs={"problem": problem, "compiler": xag_compiler, "input_state": xag_trial.input_state},
            memory_limit_mb=2024,
            timeout_seconds = 120,
            silence_output=True
        )
        xag_trial.circuit_pretranspile = compiler_run.result if compiler_run.success else None

        
        if cf_trial.circuit_pretranspile is None or xag_trial.circuit_pretranspile is None:
            # skip trial if compiling fails
            print(f"Compilation failed for problem {problem.id}, skipping")
            continue

        ops1 = cf_trial.circuit_pretranspile.count_ops()
        ops2 = xag_trial.circuit_pretranspile.count_ops()

        if ops1.get("cx", 0) == 0 or ops2.get("cx", 0) == 0:
            # skip if there is no entanglement in either (this implies trivially 0 function)
            #print(f"No entanglement in either circuit for problem {problem.id}: {problem.statement}: {ops1}, skipping")
            continue

        # check that the circuits are not essentially the same (don't check x since that can vary due to input differences)
        if ops1.get("cz", 0) == ops2.get("cx", 0) and ops1.get("sx", 0) == ops2.get("sx", 0) and ops1.get("rz", 0) == ops2.get("rz", 0):
            # if the circuits are essentially the same, skip one to avoid duplicates in dataset
            data.append(get_save_data(xag_trial))
        else:
            data.append(get_save_data(cf_trial))
            data.append(get_save_data(xag_trial))

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