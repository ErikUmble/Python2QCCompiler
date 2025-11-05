from rbf import RandomBooleanFunction, RandomBooleanFunctionTrial
from benchmarklib import BenchmarkDatabase
from benchmarklib.algorithms.grover import verify_oracle
from benchmarklib.compilers import CompileType, XAGCompiler, QCFCompiler

from sqlalchemy import select, func

compilers = [QCFCompiler(), XAGCompiler()]
num_samples = 10
max_vars = 6

benchmark_db = BenchmarkDatabase("rbf.db", RandomBooleanFunction, RandomBooleanFunctionTrial)

# problems = benchmark_db.find_problem_instances(random_sample=True, limit=10)
problems = benchmark_db.query(
    select(RandomBooleanFunction)
    .where(RandomBooleanFunction.num_vars <= max_vars)
    .order_by(func.random())
    .limit(num_samples)
)
for compiler in compilers:

    for problem in problems:
        if problem.num_vars > max_vars:
            continue
        print(verify_oracle(problem.oracle(compiler.name), problem))