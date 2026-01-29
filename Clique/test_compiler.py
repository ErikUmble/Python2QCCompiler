from sqlalchemy import func, select
from benchmarklib.pipeline import PipelineCompiler
from benchmarklib.pipeline.synthesis import QuantumMPC
from benchmarklib.algorithms.grover import verify_oracle

import os
from dotenv import load_dotenv
import json
import random
import matplotlib.pyplot as plt
import numpy as np
import asyncio

from benchmarklib import BenchmarkDatabase
from clique import CliqueProblem, CliqueTrial
from benchmarklib.compilers import CompileType, XAGCompiler

from qiskit_ibm_runtime import QiskitRuntimeService

import logging
from typing import Iterable, List, Tuple, Dict, Any, Union, Optional
import qiskit
from qiskit.providers import Backend
from qiskit import QuantumCircuit, transpile
import random

from tweedledum.bool_function_compiler import circuit_input, QuantumCircuitFunction
from tweedledum import BitVec

from benchmarklib import CompileType, BenchmarkDatabase
from benchmarklib import BatchQueue
from benchmarklib.compilers import SynthesisCompiler

from sqlalchemy import func, select

# Load Qiskit
from qiskit_ibm_runtime import QiskitRuntimeService, Batch
from dotenv import load_dotenv, find_dotenv
import os

service = QiskitRuntimeService()
backend = service.backend(name="ibm_rensselaer")

benchmark_db = BenchmarkDatabase("clique.db", CliqueProblem, CliqueTrial)

compiler_qmpc = PipelineCompiler(
    synthesizer = QuantumMPC(),
    # Transpilation is an optional step, but passing a backend & transpile options will do it automatically
    #steps = [QiskitTranspile(backend=backend, optimization_level=3)]
    steps = [], 
    backend = backend, 
    transpile_options = {"optimization_level": 3}
)

problems = benchmark_db.query(
    select(CliqueProblem).where(CliqueProblem.nodes <= 5).order_by(func.random()).limit(100)
)

def get_verifier(problem: CliqueProblem):
    n = problem.nodes
    k = max(problem.nodes // 2, 2)
    # naive brute force
    solutions = set()
    for bits in range(2**n):
        bitvec = [(bits >> i) & 1 for i in range(n)]
        if sum(bitvec) == k:
            valid = True
            edge_idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if bitvec[i] and bitvec[j]:
                        if problem.graph[edge_idx] == '0':
                            valid = False
                    edge_idx += 1
            if valid:
                solutions.add(tuple(bitvec))
    def verifier(Vertices_ex_0: List[bool]) -> bool:
        return tuple(Vertices_ex_0) in solutions
    return verifier

for p in problems:
    print(f"Verifying problem {p.graph}")
    #p.verify_solution = get_verifier(p)
    compilation_result = compiler_qmpc.compile(p, clique_size=max(p.nodes//2, 2))
    if not verify_oracle(compilation_result.synthesis_circuit, p):
        print(f"Verification failed for problem {p.graph}")
        print(compilation_result.synthesis_circuit.draw())