import logging
import benchmarklib
from benchmarklib import (
    BenchmarkDatabase,
    CompileType,
    GroverRunner,
    GroverConfig,
    calculate_grover_iterations,
)
from sat import ThreeSat, ThreeSatTrial, parameterized_3sat, unit_propped_sat
from pysat.formula import CNF
from tweedledum.bool_function_compiler import QuantumCircuitFunction
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
import numpy as np

benchmarklib.setup_logging(logging.INFO)


# Unit propagation functions
def apply_assumptions(
    circuit: List[List[int]], assumptions: List[int]
) -> List[List[int]]:
    clauses = list()
    for clause in circuit:
        if any(assumption in clause for assumption in assumptions):
            continue
        kept_clause = clause.copy()
        for assumption in assumptions:
            if -assumption in kept_clause:
                kept_clause.remove(-assumption)
        if not kept_clause:
            return [[]]
        clauses.append(kept_clause)
    return clauses


def evaluate_circuit_metrics(
    n: int, clauses: List[List[int]], is_unit_prop: bool = False
):
    """Evaluate oracle synthesis metrics for a given circuit."""
    try:
        if is_unit_prop:
            classical_function = QuantumCircuitFunction(unit_propped_sat, n, clauses)
        else:
            classical_function = QuantumCircuitFunction(parameterized_3sat, n, clauses)

        oracle = classical_function.synthesize_quantum_circuit()
        return {
            "oracle_num_qubits": oracle.num_qubits,
            "oracle_depth": oracle.depth(),
            "oracle_area": oracle.num_qubits * oracle.depth(),
            "oracle": oracle,
        }
    except Exception as e:
        return {
            "oracle_num_qubits": float("inf"),
            "oracle_depth": float("inf"),
            "oracle_area": float("inf"),
            "oracle": None,
            "error": str(e),
        }


def find_best_configuration_for_problem(problem_instance: ThreeSat) -> Dict:
    """
    Worker function that evaluates all unit propagation options for a single problem.
    Returns the best configuration (original or unit propagated).
    """
    n = problem_instance.num_vars
    cnf = CNF()
    cnf.nv = n
    for clause in problem_instance.expr:
        cnf.append(clause)

    # Start with original circuit
    original_metrics = evaluate_circuit_metrics(
        n, problem_instance.expr, is_unit_prop=False
    )
    best_config = {
        "problem_id": problem_instance.instance_id,
        "type": "original",
        "v1": 0,
        "v2": 0,
        "clauses": problem_instance.expr,
        **original_metrics,
    }

    # Try all 2-variable unit propagations
    try:
        for var1 in range(1, n + 1):
            for var2 in range(1, n + 1):
                if var1 == var2:
                    continue

                # Generate all 4 combinations for this variable pair
                cnfs = []
                for a1 in [var1, -var1]:
                    for a2 in [var2, -var2]:
                        result_clauses = apply_assumptions(cnf.clauses, [a1, a2])

                        result_cnf = CNF()
                        result_cnf.append([a1])
                        result_cnf.append([a2])
                        for clause in result_clauses:
                            result_cnf.append(clause)
                        cnfs.append(result_cnf)

                # Evaluate this unit propagation variant
                circuit = [cnf.clauses for cnf in cnfs]
                metrics = evaluate_circuit_metrics(n, circuit, is_unit_prop=True)

                # Update best if this is better
                if metrics["oracle_area"] < best_config["oracle_area"]:
                    best_config = {
                        "problem_id": problem_instance.instance_id,
                        "type": "unit_prop",
                        "v1": var1,
                        "v2": var2,
                        "cnfs": cnfs,
                        "clauses": circuit,
                        **metrics,
                    }

    except Exception as e:
        logging.error(
            f"Error in unit propagation for problem {problem_instance.instance_id}: {e}"
        )

    logging.info(
        f"Problem {problem_instance.instance_id}: Best is {best_config['type']} "
        f"(v1={best_config['v1']}, v2={best_config['v2']}, area={best_config['oracle_area']:.0f})"
    )

    return best_config


def process_problems_parallel(
    problems: List[ThreeSat], max_workers: int = 4
) -> List[Dict]:
    """
    Process multiple problems in parallel using thread pool.
    Each worker handles complete unit propagation evaluation for one problem.
    """
    best_configs = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all problems to thread pool
        future_to_problem = {
            executor.submit(find_best_configuration_for_problem, problem): problem
            for problem in problems
        }

        # Collect results as they complete
        for future in as_completed(future_to_problem):
            problem = future_to_problem[future]
            try:
                config = future.result()
                best_configs.append(config)
            except Exception as e:
                logging.error(f"Problem {problem.instance_id} failed: {e}")
                # Add fallback original configuration
                best_configs.append(
                    {
                        "problem_id": problem.instance_id,
                        "type": "original",
                        "v1": 0,
                        "v2": 0,
                        "clauses": problem.expr,
                        "oracle_area": float("inf"),
                    }
                )

    # Sort by problem_id to maintain order
    best_configs.sort(key=lambda x: x["problem_id"])
    return best_configs


# Main benchmark with unit propagation
def run_3sat_benchmark_with_unit_prop(
    db: BenchmarkDatabase, service, backend, num_vars_range: range, max_workers: int = 4
):
    """Run 3SAT benchmark with automatic unit propagation selection."""

    config = GroverConfig(shots=10**4, optimization_level=3, dynamical_decoupling=True)
    runner = GroverRunner(
        db_manager=db, service=service, backend=backend, config=config
    )

    for compile_type in [CompileType.XAG]:
        runner.start_batch(compile_type)

        for num_vars in num_vars_range:
            logging.info(f"\nProcessing {num_vars} variables...")

            # Get all problems for this size
            problems = list(
                db.find_problem_instances(
                    size_filters={"num_vars": num_vars}, limit=20, random_sample=True
                )
            )
            logging.info(f"Found {len(problems)} problems")

            # Process all problems in parallel to find best configurations
            logging.info(f"Evaluating unit propagation with {max_workers} workers...")
            best_configs = process_problems_parallel(problems, max_workers=max_workers)

            # Create problem->config mapping
            config_map = {config["problem_id"]: config for config in best_configs}

            # Run Grover benchmark with best configurations
            for problem in problems:
                best_config = config_map[problem.instance_id]

                if best_config["oracle_area"] == float("inf"):
                    logging.warning(
                        f"Skipping problem {problem.instance_id} - synthesis failed"
                    )
                    continue

                logging.info(
                    f"Problem {problem.instance_id}: Using {best_config['type']} "
                    f"(area={best_config['oracle_area']:.0f})"
                )

                # Store unit propagation decision as metadata
                metadata = {
                    "unit_prop_type": best_config["type"],
                    "unit_prop_v1": best_config["v1"],
                    "unit_prop_v2": best_config["v2"],
                    "oracle_area": best_config["oracle_area"],
                }

                optimal_grover_iters = calculate_grover_iterations(
                    len(problem.solutions), 2**num_vars
                )

                for grover_iter in range(1, optimal_grover_iters + 1):
                    runner.run_grover_benchmark(
                        problem_instance=problem,
                        compile_type=compile_type,
                        grover_iterations=grover_iter,
                        metadata=metadata,  # Pass unit prop info if your runner supports it
                    )

            # Submit job for this num_vars group
            job_id = runner.submit_job()
            logging.info(
                f"Submitted job {job_id} for {compile_type.value}, {num_vars} vars"
            )

        runner.finish_batch()


# Utility function to save unit propagation analysis
def save_unit_prop_analysis(
    best_configs: List[Dict], filename: str = "unit_prop_analysis.csv"
):
    """Save unit propagation decisions and metrics for analysis."""
    import pandas as pd

    rows = []
    for config in best_configs:
        rows.append(
            {
                "problem_id": config["problem_id"],
                "type": config["type"],
                "v1": config["v1"],
                "v2": config["v2"],
                "oracle_qubits": config.get("oracle_num_qubits", -1),
                "oracle_depth": config.get("oracle_depth", -1),
                "oracle_area": config.get("oracle_area", -1),
                "error": config.get("error", ""),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    logging.info(f"Saved unit propagation analysis to {filename}")


# Main execution
if __name__ == "__main__":
    from qiskit_ibm_runtime import QiskitRuntimeService
    from dotenv import load_dotenv
    import os

    load_dotenv()
    API_TOKEN = os.getenv("API_TOKEN")
    API_INSTANCE = os.getenv("API_INSTANCE", None)
    service = QiskitRuntimeService(
        channel="ibm_cloud", token=API_TOKEN, instance=API_INSTANCE
    )
    backend = service.backend(name="ibm_rensselaer")

    db = BenchmarkDatabase("3SAT_UP.db", ThreeSat, ThreeSatTrial)

    num_vars_range = range(3, 6)

    run_3sat_benchmark_with_unit_prop(
        db=db,
        service=service,
        backend=backend,
        num_vars_range=num_vars_range,
        max_workers=5,  # Adjust based on your CPU cores
    )
