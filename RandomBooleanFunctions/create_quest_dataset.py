from sqlalchemy import select, func
from sqlalchemy.orm import joinedload
import qiskit
from qiskit_ibm_runtime import QiskitRuntimeService
import os
import shutil
import pickle

from benchmarklib import BenchmarkDatabase
from rbf import RandomBooleanFunctionTrial, RandomBooleanFunction

service = QiskitRuntimeService()
backend = service.backend(name="ibm_rensselaer")
rbf_db = BenchmarkDatabase("rbf.db", RandomBooleanFunction, RandomBooleanFunctionTrial)

GATE_DICT = {"rz": 0, "x": 1, "sx": 2, "cx": 3, "ecr": 3}  # add ecr for noise reporting
def build_my_noise_dict(prop):  # from torchquantum.examples.quest.utils.circ_dag_converter.py
    mydict = {}
    mydict["qubit"] = {}
    mydict["gate"] = {}
    for i, qubit_prop in enumerate(prop["qubits"]):
        mydict["qubit"][i] = {}
        for item in qubit_prop:
            if item["name"] == "T1":
                mydict["qubit"][i]["T1"] = item["value"]
            elif item["name"] == "T2":
                mydict["qubit"][i]["T2"] = item["value"]
            elif item["name"] == "prob_meas0_prep1":
                mydict["qubit"][i]["prob_meas0_prep1"] = item["value"]
            elif item["name"] == "prob_meas1_prep0":
                mydict["qubit"][i]["prob_meas1_prep0"] = item["value"]
    for gate_prop in prop["gates"]:
        if not gate_prop["gate"] in GATE_DICT:
            continue
        qubit_list = tuple(gate_prop["qubits"])
        if qubit_list not in mydict["gate"]:
            mydict["gate"][qubit_list] = {}
        for item in gate_prop["parameters"]:
            if item["name"] == "gate_error":
                mydict["gate"][qubit_list][gate_prop["gate"]] = item["value"]
    return mydict

properties_dict = backend.properties().to_dict()
noise_properties = build_my_noise_dict(properties_dict)
def convert(trial):
    circuit = trial.circuit
    circuit.remove_final_measurements(inplace=True)  # model does not support the measure gate
    qasm_str = qiskit.qasm2.dumps(circuit)
    return (qasm_str, noise_properties, trial.calculate_success_rate())


num_trials = 10000
dataset_filename = "quest_rbf.data"
BATCH_SIZE = 100

# first collect the random sample of trial ids to use
ids = rbf_db.query(
    select(RandomBooleanFunctionTrial.id)
    .where(RandomBooleanFunctionTrial._circuit_qpy != None)
    .where(RandomBooleanFunctionTrial.circuit_num_qubits <= 10)  # limit to 10 qubits due to model constraints
    .order_by(func.random())
    .limit(num_trials)
)

# load and convert data in batches and store temporarily on disk
if not os.path.exists(".temp_batches") and not os.path.isdir(".temp_batches"):
    os.mkdir(".temp_batches")

for batch_idx in range(0, num_trials, BATCH_SIZE):
    batch_ids = ids[batch_idx : batch_idx + BATCH_SIZE]
    trials = rbf_db.query(
        select(RandomBooleanFunctionTrial)
        .where(RandomBooleanFunctionTrial.id.in_(batch_ids))
        .options(joinedload(RandomBooleanFunctionTrial.problem))
    )
    print(f"Processing batch {batch_idx // BATCH_SIZE + 1} / {(num_trials + BATCH_SIZE - 1) // BATCH_SIZE}")
    raw_batch = list(map(convert, trials))
    with open(os.path.join(".temp_batches", f"batch_{batch_idx // BATCH_SIZE}.data"), "wb") as f:
        pickle.dump(raw_batch, f)

# now combine all batches into the final dataset file
raw = []
for filename in os.listdir(".temp_batches"):
    with open(os.path.join(".temp_batches", filename), "rb") as f:
        raw_batch = pickle.load(f)
        raw.extend(raw_batch)

with open("quest_rbf.data", "wb") as f:
    pickle.dump(raw, f)

shutil.rmtree(".temp_batches")