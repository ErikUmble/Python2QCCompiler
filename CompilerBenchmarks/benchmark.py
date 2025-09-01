#!/usr/bin/env python
"""
Parallelized Synthesis Benchmarking Script

Runs synthesis benchmarks in parallel across multiple compilers and problem sizes.
Each job is defined as a (compiler, graph_size) pair, dynamically distributed to workers.
"""

import argparse
import json
import logging
import math
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# Import your existing modules
from benchmarklib import BenchmarkDatabase
from benchmarklib.compilers import (  # Adjust imports
    SynthesisBenchmark,
    SynthesisCompiler,
    SynthesisTrial,
    TruthTableCompiler,
    XAGCompiler,
)
from benchmarklib.problems import CliqueProblem  # Adjust import as needed


@dataclass
class CompilerConfig:
    """Configuration for a compiler that can be pickled."""

    name: str
    compiler_type: str


@dataclass
class BenchmarkJob:
    """A single benchmarking job."""

    job_id: int
    compiler_config: CompilerConfig
    n_vertices: int
    clique_size: int
    db_path: str
    edge_prob_threshold: int
    max_problems: Optional[int] = None


@dataclass
class JobResult:
    """Result from a benchmarking job."""

    job_id: int
    compiler_name: str
    n_vertices: int
    clique_size: int
    num_problems: int
    success_count: int
    avg_metrics: Dict[str, float]
    total_time: float
    error: Optional[str] = None


def create_compiler(config: CompilerConfig) -> SynthesisCompiler:
    """Create a compiler instance from configuration."""
    # Map compiler types to classes
    COMPILER_MAP = {
        "XAG": XAGCompiler,
        "TruthTable": TruthTableCompiler,
        # Add more compiler mappings here
    }

    compiler_class = COMPILER_MAP.get(config.compiler_type)
    if not compiler_class:
        raise ValueError(f"Unknown compiler type: {config.compiler_type}")

    return compiler_class()


def run_benchmark_job(job: BenchmarkJob) -> JobResult:
    """
    Worker function to run a single benchmark job.
    This runs in a separate process.
    """
    start_time = time.time()

    # Setup logging for this worker
    logging.basicConfig(
        level=logging.INFO,
        format=f"[Worker-{job.job_id}] %(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Starting job: {job.compiler_config.name} for n={job.n_vertices}")

    try:
        # Create database connection (each process needs its own)
        db_manager = BenchmarkDatabase(
            db_name=job.db_path,
            problem_class=CliqueProblem,
            trial_class=SynthesisTrial,
        )

        # Create compiler instance
        compiler = create_compiler(job.compiler_config)

        # Create benchmark runner with just this compiler
        benchmark = SynthesisBenchmark(
            db_manager=db_manager,
            compilers=[compiler],
            save_to_db=True,
            backend=None,  # Add backend support if needed
        )

        # Find problems for this vertex count
        problems = db_manager.find_problem_instances(
            size_filters={"num_vertices": job.n_vertices},
            choose_untested=True,
            random_sample=True,
        )

        # Filter by edge probability
        problems = [p for p in problems if p.edge_probability > job.edge_prob_threshold]

        # Limit number of problems if specified
        if job.max_problems and len(problems) > job.max_problems:
            problems = problems[: job.max_problems]

        logger.info(f"Found {len(problems)} problems for n={job.n_vertices}")

        if not problems:
            return JobResult(
                job_id=job.job_id,
                compiler_name=job.compiler_config.name,
                n_vertices=job.n_vertices,
                clique_size=job.clique_size,
                num_problems=0,
                success_count=0,
                avg_metrics={},
                total_time=time.time() - start_time,
                error="No suitable problems found",
            )

        # Run benchmarks
        results = benchmark.run_benchmarks(
            problems=problems,
            clique_size=job.clique_size,
            skip_existing=True,
        )

        # Calculate summary statistics
        compiler_results = results.get(compiler.name, [])
        successful = [r for r in compiler_results if r.success]

        avg_metrics = {}
        if successful:
            avg_metrics = {
                "avg_qubits": sum(r.num_qubits for r in successful) / len(successful),
                "avg_depth": sum(r.circuit_depth for r in successful) / len(successful),
                "avg_cx_count": sum(r.cx_count for r in successful) / len(successful),
                "avg_synthesis_time": sum(r.synthesis_time for r in successful)
                / len(successful),
            }

        total_time = time.time() - start_time
        logger.info(
            f"Completed job in {total_time:.2f}s: {job.compiler_config.name} n={job.n_vertices}"
        )

        return JobResult(
            job_id=job.job_id,
            compiler_name=job.compiler_config.name,
            n_vertices=job.n_vertices,
            clique_size=job.clique_size,
            num_problems=len(problems),
            success_count=len(successful),
            avg_metrics=avg_metrics,
            total_time=total_time,
        )

    except Exception as e:
        logger.error(f"Job failed: {e}")
        return JobResult(
            job_id=job.job_id,
            compiler_name=job.compiler_config.name,
            n_vertices=job.n_vertices,
            clique_size=job.clique_size,
            num_problems=0,
            success_count=0,
            avg_metrics={},
            total_time=time.time() - start_time,
            error=str(e),
        )


def create_jobs(
    compiler_configs: List[CompilerConfig],
    n_range: Tuple[int, int],
    db_path: str,
    edge_prob_threshold: int = 50,
    max_problems_per_n: Optional[int] = None,
) -> List[BenchmarkJob]:
    """Create all benchmark jobs."""
    jobs = []
    job_id = 0

    for n in range(n_range[0], n_range[1] + 1):
        clique_size = math.ceil(n / 2)

        for compiler_config in compiler_configs:
            job = BenchmarkJob(
                job_id=job_id,
                compiler_config=compiler_config,
                n_vertices=n,
                clique_size=clique_size,
                db_path=db_path,
                edge_prob_threshold=edge_prob_threshold,
                max_problems=max_problems_per_n,
            )
            jobs.append(job)
            job_id += 1

    return jobs


def print_results_summary(
    results: List[JobResult], compiler_configs: List[CompilerConfig]
):
    """Print a formatted summary of all results."""
    print("\n" + "=" * 80)
    print("PARALLEL BENCHMARK RESULTS SUMMARY")
    print("=" * 80)

    # Group results by compiler
    by_compiler = {}
    for result in results:
        if result.compiler_name not in by_compiler:
            by_compiler[result.compiler_name] = []
        by_compiler[result.compiler_name].append(result)

    # Print per-compiler summary
    for compiler_name in by_compiler:
        print(f"\n{compiler_name} Results:")
        print("-" * 60)
        print("n\tClique\tProblems\tSuccess\tQubits\tDepth\tCX\tTime(s)")
        print("-" * 60)

        compiler_results = sorted(
            by_compiler[compiler_name], key=lambda r: r.n_vertices
        )

        for result in compiler_results:
            if result.error:
                print(
                    f"{result.n_vertices}\t{result.clique_size}\tERROR: {result.error}"
                )
            elif result.success_count > 0:
                metrics = result.avg_metrics
                success_rate = (
                    (result.success_count / result.num_problems * 100)
                    if result.num_problems > 0
                    else 0
                )
                print(
                    f"{result.n_vertices}\t{result.clique_size}\t{result.num_problems}\t"
                    f"{success_rate:.0f}%\t"
                    f"{metrics.get('avg_qubits', 0):.1f}\t"
                    f"{metrics.get('avg_depth', 0):.0f}\t"
                    f"{metrics.get('avg_cx_count', 0):.0f}\t"
                    f"{result.total_time:.2f}"
                )
            else:
                print(
                    f"{result.n_vertices}\t{result.clique_size}\t{result.num_problems}\t0%\t-\t-\t-\t{result.total_time:.2f}"
                )

    # Print timing summary
    print("\n" + "=" * 80)
    print("EXECUTION TIME SUMMARY")
    print("-" * 60)
    total_time = sum(r.total_time for r in results)
    avg_time = total_time / len(results) if results else 0
    print(f"Total job time (sum): {total_time:.2f}s")
    print(f"Average job time: {avg_time:.2f}s")
    print(f"Jobs completed: {len(results)}")


def main():
    parser = argparse.ArgumentParser(description="Parallel Synthesis Benchmarking")

    # Database arguments
    parser.add_argument(
        "--db", type=str, default="BenchmarkTest.db", help="Path to benchmark database"
    )

    # Range arguments
    parser.add_argument(
        "--n-min", type=int, default=3, help="Minimum number of vertices"
    )
    parser.add_argument(
        "--n-max", type=int, default=12, help="Maximum number of vertices"
    )

    # Compiler selection
    parser.add_argument(
        "--compilers",
        nargs="+",
        choices=["xag", "xag-opt", "truth-table", "all"],
        default=["all"],
        help="Compilers to benchmark",
    )

    # Problem selection
    parser.add_argument(
        "--edge-prob-threshold",
        type=int,
        default=50,
        help="Minimum edge probability for problems",
    )
    parser.add_argument(
        "--max-problems", type=int, default=None, help="Maximum problems per n value"
    )

    # Parallelization
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Run sequentially (no parallelization)",
    )

    # Output
    parser.add_argument(
        "--output", type=str, default=None, help="Save results to JSON file"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Configure compilers based on arguments
    compiler_configs = []

    if "all" in args.compilers or "xag" in args.compilers:
        compiler_configs.append(CompilerConfig(name="XAG", compiler_type="XAG"))

    if "all" in args.compilers or "truth-table" in args.compilers:
        compiler_configs.append(
            CompilerConfig(name="TruthTable", compiler_type="TruthTable")
        )

    # Create jobs
    jobs = create_jobs(
        compiler_configs=compiler_configs,
        n_range=(args.n_min, args.n_max),
        db_path=args.db,
        edge_prob_threshold=args.edge_prob_threshold,
        max_problems_per_n=args.max_problems,
    )

    logger.info(f"Created {len(jobs)} benchmark jobs")
    logger.info(f"Compilers: {[c.name for c in compiler_configs]}")
    logger.info(f"Vertex range: {args.n_min} to {args.n_max}")

    # Run benchmarks
    results = []
    start_time = time.time()

    if args.sequential:
        logger.info("Running benchmarks sequentially...")
        for job in jobs:
            result = run_benchmark_job(job)
            results.append(result)
            print(
                f"Completed job {job.job_id + 1}/{len(jobs)}: "
                f"{job.compiler_config.name} n={job.n_vertices}"
            )
    else:
        # Determine number of workers
        num_workers = args.workers or mp.cpu_count()
        logger.info(f"Running benchmarks in parallel with {num_workers} workers...")

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all jobs
            future_to_job = {
                executor.submit(run_benchmark_job, job): job for job in jobs
            }

            # Process completed jobs as they finish
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(
                        f"Completed job {len(results)}/{len(jobs)}: "
                        f"{job.compiler_config.name} n={job.n_vertices} "
                        f"(time: {result.total_time:.2f}s)"
                    )
                except Exception as e:
                    logger.error(f"Job failed: {e}")
                    results.append(
                        JobResult(
                            job_id=job.job_id,
                            compiler_name=job.compiler_config.name,
                            n_vertices=job.n_vertices,
                            clique_size=job.clique_size,
                            num_problems=0,
                            success_count=0,
                            avg_metrics={},
                            total_time=0,
                            error=str(e),
                        )
                    )

    elapsed_time = time.time() - start_time
    logger.info(f"All benchmarks completed in {elapsed_time:.2f}s")

    # Print summary
    print_results_summary(results, compiler_configs)

    # Calculate speedup if parallel
    if not args.sequential and results:
        total_sequential_time = sum(r.total_time for r in results)
        speedup = total_sequential_time / elapsed_time
        print(f"\nParallel speedup: {speedup:.2f}x")
        print(f"Efficiency: {speedup / num_workers * 100:.1f}%")

    # Save results if requested
    if args.output:
        output_data = {
            "config": {
                "compilers": [c.__dict__ for c in compiler_configs],
                "n_range": [args.n_min, args.n_max],
                "edge_prob_threshold": args.edge_prob_threshold,
                "max_problems": args.max_problems,
                "workers": num_workers if not args.sequential else 1,
            },
            "results": [r.__dict__ for r in results],
            "timing": {
                "total_wall_time": elapsed_time,
                "total_cpu_time": sum(r.total_time for r in results),
            },
        }

        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
