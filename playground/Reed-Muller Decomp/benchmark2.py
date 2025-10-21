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
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List

import networkx as nx
import pygraphviz
import sympy
from benchmarklib.databases import CliqueDatabase
from benchmarklib.pipeline import (
    CompilationResult,
    PipelineCompiler,
    PipelineStep,
    StepRegistry,
)
from benchmarklib.pipeline.synthesis import Synthesizer, XAGSynthesizer
from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService

# Import from benchmarklib
from benchmarklib import CliqueProblem

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =====================================
# Special Compiler Steps
# =====================================


@StepRegistry.register
class CreateNXGraph(PipelineStep):
    @property
    def name(self) -> str:
        return self.__class__.__name__

    def transform(
        self, synthesizer: Synthesizer, circuit: QuantumCircuit, **kwargs
    ) -> QuantumCircuit:
        if isinstance(synthesizer, XAGSynthesizer):
            dot_str = synthesizer.compilation_artifacts["xag_graphviz"]
            A = pygraphviz.AGraph()
            A.from_string(dot_str)
            G = nx.drawing.nx_agraph.from_agraph(A)

            for node, data in G.nodes.data():
                try:
                    node_label = int(data["label"])
                    if node_label != 0:
                        data["label"] = f"input{node_label}"
                        data["color"] = "green"
                    else:
                        data["color"] = "cyan"
                except:
                    clr_map = {"XOR": "blue", "AND": "red"}
                    if node != "po0":
                        data["color"] = clr_map[data["label"]]
                    pass

                del data["fillcolor"]
                del data["shape"]
                del data["style"]

            for u, v, data in G.edges.data():
                data["negated"] = data["style"] == "dashed"
                del data["style"]

            synthesizer.compilation_artifacts["xag"] = G
        else:
            print("WARNING: Not XAG Synthesizer, cannot perform graph operations")

        return circuit


@StepRegistry.register
class BoolFunctionFromNXGraph(PipelineStep):
    @property
    def name(self) -> str:
        return self.__class__.__name__

    def transform(
        self, synthesizer: Synthesizer, circuit: QuantumCircuit, **kwargs
    ) -> QuantumCircuit:
        if isinstance(synthesizer, XAGSynthesizer):
            G = synthesizer.compilation_artifacts["xag"]

            if not isinstance(G, nx.MultiDiGraph):
                raise ValueError("Expected nx XAG")

            G_rev = G.to_directed().reverse()
            nodes = list(nx.topological_sort(G_rev))

            print("\nSTARTING EXPRESSION BUILDER")
            expr = self.build_expression(G_rev, nodes[2])

            # check final edge to see if we need to negate our expr
            # first edge is always the one we want, nx was giving me issues so we put it in a loop
            for u, v, data in G.in_edges("po0", data=True):
                if data["negated"]:
                    expr = sympy.Not(expr)
                synthesizer.compilation_artifacts["reduced_expr"] = expr
                return circuit
        else:
            print("WARNING: Not XAG Synthesizer, cannot perform graph operations")
            return circuit

    def build_expression(self, graph: nx.MultiDiGraph, node, memo=None):
        """
        Recursively builds a sympy boolean expression from a graph,
        starting from the output node.

        Args:
            graph (nx.MultiDiGraph): The graph to traverse.
            node_id: The ID of the current node to process.
            memo (dict): A dictionary to store results and avoid recomputing.

        Returns:
            A sympy boolean expression for the subtree
            rooted at the given node.
        """
        if memo is None:
            memo = {}

        # Check if we have already computed this node's expression
        if node in memo:
            return memo[node]

        # print(graph[node], graph.nodes[node])
        node_label = graph.nodes[node]["label"]

        # Handle input nodes which don't have an operation
        if "input" in node_label:
            # There should only be incoming edges to the reversed graph input nodes.
            for u, v, key, data in graph.in_edges(node, keys=True, data=True):
                is_negated = data.get("negated", False)
                input_expr = sympy.Symbol(
                    f"x{node_label[5:]}"
                )  # name variable in sympy xNUM

                if is_negated:
                    result = sympy.Not(input_expr)
                else:
                    result = input_expr
                memo[node] = result
                return result

        # Recursive step: The node is a logic gate.
        gate_type = node_label.upper()
        sub_expressions = []

        # Iterate through the incoming edges to get the parent nodes and edge data.
        for u, v, key, data in graph.out_edges(node, keys=True, data=True):
            is_negated = data.get("negated", False)

            # Recursively build the expression for the parent node.
            parent_expression = self.build_expression(graph, v, memo)

            # Apply negation if the edge is dashed.
            if is_negated:
                sub_expressions.append(sympy.Not(parent_expression))
            else:
                sub_expressions.append(parent_expression)

        # Join the sub-expressions with the appropriate operator.
        if gate_type == "XOR":
            result = sympy.Xor(*sub_expressions)
        elif gate_type == "AND":
            result = sympy.And(*sub_expressions)
        else:
            # Fallback for other gate types
            result = f"({gate_type} {str(sub_expressions)})"

        # Store the result in the memoization dictionary before returning.
        memo[node] = result

        return result


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

        # compiler = PipelineCompiler.from_dict(compiler_data, backend=backend)
        compiler = PipelineCompiler.from_dict(compiler_data, backend=None)
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
                   
            -- Boolean Function Metrics
            expr TEXT,
            expr_num_args INTEGER,
            expr_clause_len_dict TEXT,                   
            expr_symbol_counts TEXT,
            multiplicative_complexity INTEGER,
            additive_complexity INTEGER,
            max_width_level INTEGER,
            max_width INTEGER,
            longest_path_len INTEGER,
            and_dist_by_level TEXT,
            xor_dist_by_level TEXT,
            and_count INTEGER,
            xor_count INTEGER,
            num_nodes INTEGER,
                    

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


def count_and_operators(expr):
    """
    Recursively counts the number of And operators in a SymPy boolean expression.
    """
    and_count = 0
    xor_count = 0
    if isinstance(expr, sympy.And):
        and_count += 1
    elif isinstance(expr, sympy.Xor):
        xor_count += 1

    for arg in expr.args:
        new_and, new_xor = count_and_operators(arg)
        and_count += new_and
        xor_count += new_xor

    return and_count, xor_count


def compute_graph_stats(G: nx.MultiDiGraph):
    """Compute graph statistics for XAG
    - width of graph (maximum nodes at a level)
    - distribution of ANDs at different levels of the graph
    - Longest path from any input to the output
    - AND and XOR node counts
    """
    queue = deque()

    node_level = defaultdict(int)
    in_degree = defaultdict(int)

    for node in G.nodes:
        in_deg = G.in_degree(node)
        if node != "0" and in_deg == 0:
            queue.append(node)
            node_level[node] = 0
        else:
            in_degree[node] = in_deg

    # topological search to find levels
    while queue:
        parent = queue.popleft()

        for u, v in G.out_edges(parent):
            # level of a node is the longest path from an input
            node_level[v] = max(node_level[v], node_level[u] + 1)
            in_degree[v] -= 1

            if in_degree[v] == 0:
                queue.append(v)

    width_at_level = defaultdict(int)
    for level in node_level.values():
        width_at_level[level] += 1

    longest_path = nx.dag_longest_path_length(G)
    and_dist = [0 for _ in range(longest_path)]
    xor_dist = [0 for _ in range(longest_path)]

    xor_count = 0
    and_count = 0
    for node, data in G.nodes(data=True):
        try:
            if data["label"] == "XOR":
                xor_count += 1
                xor_dist[node_level[node]] += 1
            elif data["label"] == "AND":
                and_count += 1
                and_dist[node_level[node]] += 1
        except:
            continue

    return {
        # find the level
        "max_width_level": max(width_at_level, key=width_at_level.get),
        "max_width": max(width_at_level.values()),
        "longest_path_len": longest_path,
        "and_dist_by_level": and_dist,
        "xor_dist_by_level": xor_dist,
        "and_count": and_count,
        "xor_count": xor_count,
        "num_nodes": G.number_of_nodes(),
    }


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

            # boolean function conversion and statistics
            expr = result.artifacts["reduced_expr"]
            expr_num_clauses = len(expr.args)
            expr_clause_len_dict = defaultdict(int)
            for clause in expr.args:
                expr_clause_len_dict[len(clause.args)] += 1

            expr_symbol_counts = {}
            for sym in expr.free_symbols:
                expr_symbol_counts[str(sym)] = expr.count(sym)

            mutliplicative_complexity, additive_complexity = count_and_operators(expr)

            xag = result.artifacts["xag"]

            graph_stats = compute_graph_stats(xag)

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
                    expr, expr_num_args, expr_clause_len_dict, expr_symbol_counts,
                    multiplicative_complexity, additive_complexity,
                    max_width_level, max_width, longest_path_len, and_dist_by_level, xor_dist_by_level, and_count, xor_count, num_nodes,
                    trial_params, error_message, error_stage, pipeline_config, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    str(expr),
                    expr_num_clauses,
                    json.dumps(expr_clause_len_dict),
                    json.dumps(expr_symbol_counts),
                    mutliplicative_complexity,
                    additive_complexity,
                    graph_stats["max_width_level"],
                    graph_stats["max_width"],
                    graph_stats["longest_path_len"],
                    json.dumps(graph_stats["and_dist_by_level"]),
                    json.dumps(graph_stats["xor_dist_by_level"]),
                    graph_stats["and_count"],
                    graph_stats["xor_count"],
                    graph_stats["num_nodes"],
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
        description="Benchmark Boolean ANF from reduced XAG"
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

    compiler = PipelineCompiler(
        synthesizer=XAGSynthesizer(),
        steps=[CreateNXGraph(), BoolFunctionFromNXGraph()],
    )

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

        # Check existing if not skipping
        if not args.no_skip:
            problem_ids = [p.instance_id for p in problems]
            existing = check_existing(db_path, compiler.name, problem_ids, trial_params)
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

                save_results([result], db_path, trial_params)

                status = "✓" if result.success else "✗"
                print(
                    f"[{i}/{len(jobs)}] {status} {compiler.name} - {result.total_time:.2f}s"
                )

            all_results.extend(results)

            # Save batch to database
            # save_results(results, db_path, trial_params)


if __name__ == "__main__":
    main()
