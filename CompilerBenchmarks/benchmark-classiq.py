#!/usr/bin/env python
"""
Pipeline Compiler Benchmarking using PipelineCompiler and CompilationResult

Benchmarks XAG, TruthTable, and Classiq synthesizers with Qiskit level 3 optimization.
"""

import argparse
import json
import logging
import random
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
from benchmarklib.databases import CliqueDatabase
from benchmarklib.pipeline import CompilationResult, PipelineCompiler
from benchmarklib.pipeline.synthesis import (
    ClassiqSynthesizer,
    TruthTableSynthesizer,
    XAGSynthesizer,
)
from qiskit_ibm_runtime import QiskitRuntimeService

# Import from benchmarklib
from benchmarklib import CliqueProblem

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Qiskit Runtime SETUP
# ============================================================================

import os

from dotenv import load_dotenv

# create XAG Compiler

load_dotenv()
API_TOKEN = os.getenv("API_TOKEN")
API_INSTANCE = os.getenv("API_INSTANCE", None)
service = QiskitRuntimeService(
    channel="ibm_cloud", token=API_TOKEN, instance=API_INSTANCE
)
backend = service.backend(name="ibm_rensselaer")

# ============================================================================
# COMPILER SETUP
# ============================================================================


def create_compilers(backend=None):
    """Create Pipeline compilers with XAG, TruthTable, and Classiq synthesizers."""
    # synthesizers = [XAGSynthesizer(), TruthTableSynthesizer(), ClassiqSynthesizer()]
    synthesizers = [ClassiqSynthesizer()]

    compilers = []
    for synthesizer in synthesizers:
        compiler = PipelineCompiler(
            synthesizer=synthesizer,
            steps=[],
            backend=backend,
            transpile_options={"optimization_level": 3},
        )
        compilers.append(compiler)

    return compilers


# ============================================================================
# WORKER FUNCTION
# ============================================================================


def compile_problem(args):
    """Worker function to compile a single problem."""
    (
        compiler_data,
        problem_data,
        problem_id,
        trial_params,
    ) = args

    try:
        problem = CliqueProblem.from_dict(problem_data, instance_id=problem_id)

        compiler = PipelineCompiler.from_dict(compiler_data, backend=backend)
        # Run compilation
        result = compiler.compile(problem, return_intermediate=False, **trial_params)

        # Add problem_id to result for tracking
        result.problem_id = problem_id

        return result

    except Exception as e:
        # Return failed result
        return CompilationResult(
            compiler_name=compiler.name,
            success=False,
            total_time=0,
            error_message=str(e),
            error_stage="worker_execution",
        )


# ============================================================================
# DATABASE STORAGE
# ============================================================================


def setup_results_database(db_path):
    """Create results database with appropriate schema."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create results table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS compilation_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            problem_id INTEGER NOT NULL,
            compiler_name TEXT NOT NULL,
            success BOOLEAN NOT NULL,
            
            -- Timing
            total_time REAL,
            synthesis_time REAL,
            pipeline_time REAL,  -- Sum of all pipeline step times
            transpilation_time REAL,
            
            -- Synthesis metrics
            synthesis_qubits INTEGER,
            synthesis_depth INTEGER,
            synthesis_gates INTEGER,
            synthesis_entangling INTEGER,
            synthesis_single_qubit INTEGER,
            
            -- High-level metrics (after pipeline, before transpilation)
            high_level_qubits INTEGER,
            high_level_depth INTEGER,
            high_level_gates INTEGER,
            high_level_entangling INTEGER,
            high_level_single_qubit INTEGER,
            
            -- Low-level metrics (after transpilation)
            low_level_qubits INTEGER,
            low_level_depth INTEGER,
            low_level_gates INTEGER,
            low_level_entangling INTEGER,
            low_level_single_qubit INTEGER,
            
            -- Trial parameters and metadata
            trial_params TEXT,
            error_message TEXT,
            error_stage TEXT,
            pipeline_config TEXT,
            timestamp TEXT,
            
            -- Create unique constraint to prevent duplicates
            UNIQUE(problem_id, compiler_name, trial_params)
        )
    """)

    # Create indices
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_problem ON compilation_results(problem_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_compiler ON compilation_results(compiler_name)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_success ON compilation_results(success)"
    )

    conn.commit()
    conn.close()

    logger.info(f"Results database ready at: {db_path}")
    return db_path


def save_results(results: List[CompilationResult], db_path: str, trial_params: dict):
    """Save compilation results to database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    saved_count = 0
    for result in results:
        try:
            # Extract metrics from CircuitMetrics objects
            synthesis_metrics = (
                result.synthesis_metrics.__dict__ if result.synthesis_metrics else {}
            )
            high_level_metrics = (
                result.high_level_metrics.__dict__ if result.high_level_metrics else {}
            )
            low_level_metrics = (
                result.low_level_metrics.__dict__ if result.low_level_metrics else {}
            )

            # Calculate total pipeline time
            pipeline_time = sum(result.pipeline_times) if result.pipeline_times else 0

            cursor.execute(
                """
                INSERT OR REPLACE INTO compilation_results (
                    problem_id, compiler_name, success,
                    total_time, synthesis_time, pipeline_time, transpilation_time,
                    synthesis_qubits, synthesis_depth, synthesis_gates, 
                    synthesis_entangling, synthesis_single_qubit,
                    high_level_qubits, high_level_depth, high_level_gates,
                    high_level_entangling, high_level_single_qubit,
                    low_level_qubits, low_level_depth, low_level_gates,
                    low_level_entangling, low_level_single_qubit,
                    trial_params, error_message, error_stage, pipeline_config, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    getattr(result, "problem_id", None),
                    str(result.compiler_name),
                    result.success,
                    result.total_time,
                    result.synthesis_time,
                    pipeline_time,
                    result.transpilation_time,
                    synthesis_metrics.get("num_qubits"),
                    synthesis_metrics.get("depth"),
                    synthesis_metrics.get("gate_count"),
                    synthesis_metrics.get("entangling_count"),
                    synthesis_metrics.get("single_qubit_count"),
                    high_level_metrics.get("num_qubits"),
                    high_level_metrics.get("depth"),
                    high_level_metrics.get("gate_count"),
                    high_level_metrics.get("entangling_count"),
                    high_level_metrics.get("single_qubit_count"),
                    low_level_metrics.get("num_qubits"),
                    low_level_metrics.get("depth"),
                    low_level_metrics.get("gate_count"),
                    low_level_metrics.get("entangling_count"),
                    low_level_metrics.get("single_qubit_count"),
                    json.dumps(trial_params),
                    result.error_message,
                    result.error_stage,
                    json.dumps(result.pipeline_config)
                    if result.pipeline_config
                    else None,
                    datetime.now().isoformat(),
                ),
            )
            saved_count += 1
        except Exception as e:
            logger.error(f"Failed to save result for {result.compiler_name}: {e}")

    conn.commit()
    conn.close()

    logger.info(f"Saved {saved_count} results to database")


def check_existing(
    db_path: str, compiler_name: str, problem_ids: List[int], trial_params: dict
):
    """Check which problem-compiler pairs already exist."""
    if not Path(db_path).exists():
        return set()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    params_json = json.dumps(trial_params)
    placeholders = ",".join("?" * len(problem_ids))

    cursor.execute(
        f"""
        SELECT problem_id FROM compilation_results
        WHERE compiler_name = ? 
        AND trial_params = ?
        AND problem_id IN ({placeholders})
        AND success = 1
    """,
        [compiler_name, params_json] + problem_ids,
    )

    existing = {row[0] for row in cursor.fetchall()}
    conn.close()

    return existing


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Pipeline Compilers with Qiskit Level 3 Optimization"
    )

    parser.add_argument(
        "--db", default="project_benchmarks/results.db", help="Results database path"
    )
    parser.add_argument("--vertices", type=int, help="Number of vertices")
    parser.add_argument("--vertex-range", nargs=2, type=int, metavar=("MIN", "MAX"))
    # parser.add_argument("--edge-prob", type=int, help="Edge probability (0-100)")
    parser.add_argument("--clique-size", type=int, help="Clique size k")
    parser.add_argument("--max-problems", type=int, default=10)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--backend", type=str, help="IBM backend name")
    parser.add_argument("--no-skip", action="store_true", help="Don't skip existing")

    args = parser.parse_args()

    # Setup database
    db_path = setup_results_database(args.db)

    compilers = create_compilers(backend)
    logger.info(f"Created {len(compilers)} compilers: {[c.name for c in compilers]}")

    # Determine vertex counts to benchmark
    if args.vertex_range:
        vertex_counts = list(range(args.vertex_range[0], args.vertex_range[1] + 1))
    elif args.vertices:
        vertex_counts = [args.vertices]
    else:
        vertex_counts = [5]  # Default

    # Process each vertex count
    all_results = []

    clique_db = CliqueDatabase.shared()

    for n_vertices in vertex_counts:
        # Get problems
        filters = {"num_vertices": n_vertices}
        # if args.edge_prob:
        #   filters["edge_probability"] = args.edge_prob

        # problems = clique_db.random_sample(**filters, limit=args.max_problems)
        problems = clique_db.find(**filters)
        print(len(problems))
        # filter by edge probability and randomly sample
        problems = random.sample(
            [p for p in problems if p.edge_probability >= 30], args.max_problems
        )

        logger.info(f"Processing {len(problems)} problems with {n_vertices} vertices")

        # Set trial parameters
        trial_params = {"clique_size": args.clique_size or (n_vertices // 2)}

        # Generate jobs for each compiler
        for compiler in compilers:
            # Check existing if not skipping
            if not args.no_skip:
                problem_ids = [p.instance_id for p in problems]
                existing = check_existing(
                    db_path, compiler.name, problem_ids, trial_params
                )
                problems_to_run = [p for p in problems if p.instance_id not in existing]

                if len(existing) > 0:
                    logger.info(
                        f"Skipping {len(existing)} existing results for {compiler.name}"
                    )
            else:
                problems_to_run = problems

            if not problems_to_run:
                continue

            # Prepare jobs
            jobs = [
                (
                    compiler.to_dict(),
                    p.to_dict(),
                    p.instance_id,
                    trial_params,
                )
                for p in problems_to_run
            ]

            # Run compilation
            logger.info(f"Running {len(jobs)} jobs for {compiler.name}")

            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(compile_problem, job) for job in jobs]

                results = []
                for i, future in enumerate(as_completed(futures), 1):
                    result = future.result()
                    results.append(result)

                    status = "✓" if result.success else "✗"
                    print(
                        f"[{i}/{len(jobs)}] {status} {compiler.name} - {result.total_time:.2f}s"
                    )

                all_results.extend(results)

                # Save batch to database
                save_results(results, db_path, trial_params)

    # Display summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    by_compiler = {}
    for r in all_results:
        if r.compiler_name not in by_compiler:
            by_compiler[r.compiler_name] = []
        by_compiler[r.compiler_name].append(r)

    for name, results in by_compiler.items():
        successful = [r for r in results if r.success]
        print(f"\n{name}:")
        print(f"  Success: {len(successful)}/{len(results)}")
        if successful:
            times = [r.total_time for r in successful]
            print(f"  Time: {np.mean(times):.3f} ± {np.std(times):.3f}s")

            if successful[0].low_level_metrics:
                depths = [r.low_level_metrics.depth for r in successful]
                cx_counts = [r.low_level_metrics.entangling_count for r in successful]
                print(f"  Depth: {np.mean(depths):.0f} ± {np.std(depths):.0f}")
                print(f"  CX: {np.mean(cx_counts):.0f} ± {np.std(cx_counts):.0f}")


if __name__ == "__main__":
    main()
