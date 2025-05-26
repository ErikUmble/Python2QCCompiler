import os
from dotenv import load_dotenv
import json
import random
import matplotlib.pyplot as plt
import numpy as np
import asyncio

from benchmark import run_benchmark_sample, Trials, mark_job_failure, create_compilation_failure_trial

crashes = []
def run_benchmark(include_existing_trials=True):
    for complexity in range(8, 21):
        for num_vars in range(2, 16):
            if (num_vars, complexity) in crashes:
                print(f"Skipping {num_vars} vars and {complexity} complexity")
                continue
            # lowering shots to 10**3 to meet new system constraint (it is either that or reduce num_inputs / split into more batches)
            run_benchmark_sample(num_vars, complexity, num_functions=5, num_inputs=100, shots=10**3, include_existing_trials=include_existing_trials, circuits_per_job=25)

run_benchmark()