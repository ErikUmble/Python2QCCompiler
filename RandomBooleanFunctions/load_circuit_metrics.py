import os
from dotenv import load_dotenv
import json
import random
import matplotlib.pyplot as plt
import numpy as np
import asyncio
import numpy as np
from sqlalchemy.orm import joinedload

from benchmarklib import BenchmarkDatabase
from rbf import RandomBooleanFunctionTrial, RandomBooleanFunction
from benchmarklib.compilers import CompileType, XAGCompiler

from qiskit_ibm_runtime import QiskitRuntimeService, RuntimeJobNotFound

import logging
from typing import Iterable, List, Tuple, Dict, Any, Union, Optional
import qiskit
from qiskit.providers import Backend
from qiskit import QuantumCircuit, transpile
import random

from tweedledum.bool_function_compiler import circuit_input, QuantumCircuitFunction
from tweedledum import BitVec

from sqlalchemy import select, func
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

from benchmarklib import CompileType, BenchmarkDatabase
from benchmarklib import BatchQueue
from benchmarklib.compilers import SynthesisCompiler

from qiskit_ibm_runtime import RuntimeJobFailureError
from collections import defaultdict
from sqlalchemy import update

load_dotenv()
API_TOKEN_OLD = os.getenv("API_TOKEN_OLD")
API_INSTANCE_OLD = os.getenv("API_INSTANCE_OLD")
service = QiskitRuntimeService()  # default service with new credentials
service_old = QiskitRuntimeService(
    channel='ibm_quantum_platform',
    token=API_TOKEN_OLD,
    instance=API_INSTANCE_OLD
)
backend = service.backend("ibm_rensselaer")
benchmark_db = BenchmarkDatabase("rbf.db", RandomBooleanFunction, RandomBooleanFunctionTrial)

def _get_missing_circuit_job_ids(db_manager) -> List[str]:
    """Get all job IDs with pending results."""
    with db_manager.session() as session:
        query = (
            select(db_manager.trial_class.job_id)
            .where(
                db_manager.trial_class.job_id != None,
                db_manager.trial_class.circuit_depth == None,
                db_manager.trial_class.is_failed == False
            )
            .distinct()
            .order_by(func.random())
            .limit(1000)
        )
        results = session.execute(query).scalars().all()
        return list(results)
    
def get_missing_circuit_job_ids(db_manager) -> List[str]:
    """Return job ID for each (problem, compiler) tuple which does not yet have a circuit."""
    compilers = ["CLASSICAL_FUNCTION", "XAG"]
    with db_manager.session() as session:
        # first, get the problem/compiler combinations that have circuits
        existing_combinations = set(session.execute(
            select(db_manager.trial_class.problem_id, db_manager.trial_class.compiler_name)
            .distinct()
            .where(
                db_manager.trial_class._circuit_qpy != None
            )).all()
        )
        
        all_problems = session.execute(select(db_manager.problem_class.id)).scalars().all()

        missing = set()
        for problem_id in all_problems:
            for compiler_name in compilers:
                if (problem_id, compiler_name) not in existing_combinations:
                    # find a job ID for this problem/compiler
                    job_id = session.execute(
                        select(db_manager.trial_class.job_id)
                        .where(
                            db_manager.trial_class.problem_id == problem_id,
                            db_manager.trial_class.compiler_name == compiler_name,
                            db_manager.trial_class.job_id != None,
                            db_manager.trial_class._circuit_qpy == None,
                            db_manager.trial_class.is_failed == False
                        )
                        .order_by(func.random())
                        .limit(1)
                    ).scalar_one_or_none()
                    if job_id is not None:
                        missing.add(job_id)

        return list(missing)

async def update_job_results(db_manager, job_id: str, save_circuits: Optional[bool] = False) -> None:
    """
    Fetch and update results for a specific job.

    Args:
        job_id: IBM Quantum job ID
        service: QiskitRuntimeService instance
    """
    print(f"Fetching trials for job {job_id}")
    job = None
    for svc in [service, service_old]:  # try each service in order until job is found
        try:
            job = await asyncio.to_thread(svc.job, job_id)
            break
        except RuntimeJobNotFound:
            continue
    
    if job is None:
        # Handle the case where job wasn't found in any service
        print(f"Job not found in any service for trial {trial}")
        return
    
    trials = db_manager.query(
        select(db_manager.trial_class)
        .where(
            db_manager.trial_class.job_id == job_id,
            db_manager.trial_class._circuit_qpy == None,
            db_manager.trial_class.is_failed == False
        )
    )

    updated_count = 0
    with db_manager.session() as session:
        for trial in trials:
            circuit = job.inputs['pubs'][trial.job_pub_idx][0]
            trial.load_circuit_metrics(circuit=circuit)
            trial.created_at = job.creation_date
            if save_circuits:
                trial.circuit = circuit
            session.merge(trial)
            updated_count += 1
        session.commit()

        print(f"Updated {updated_count} trials for job {job_id}")


async def update_all_pending_results(db_manager, batch_size: int = 5, save_circuits: Optional[bool] = False) -> None:
    """
    Update all pending job results asynchronously.

    Args:
        service: QiskitRuntimeService instance
        batch_size: Number of concurrent job fetches
    """
    while True:
        pending_jobs = get_missing_circuit_job_ids(db_manager)
        if len(pending_jobs) == 0:
            break

        print(f"Updating circuits from {len(pending_jobs)} jobs")

        # Process jobs in batches to avoid overwhelming the API
        for i in range(0, len(pending_jobs), batch_size):
            batch = pending_jobs[i : i + batch_size]
            tasks = [update_job_results(db_manager, job_id, save_circuits=save_circuits) for job_id in batch]

            batch_num = i // batch_size + 1
            total_batches = (len(pending_jobs) + batch_size - 1) // batch_size
            print(f"Processing batch {batch_num}/{total_batches}")

            await asyncio.gather(*tasks, return_exceptions=True)

async def main():
    await update_all_pending_results(benchmark_db, batch_size=1, save_circuits=True)


if __name__ == "__main__":
    asyncio.run(main())
