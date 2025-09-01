# Import the research library 
from benchmarklib import BenchmarkDatabase

import random
from pysat.solvers import Glucose3 as Glucose
from pysat.formula import CNF
from sat import ThreeSat 

# generate a random 3SAT and return the expression and list of solutions
def generate_planted_3sat(num_vars, num_clauses, seed: int) -> ThreeSat:
    """
    Generates a planted 3SAT instance.

    Args:
        num_vars (int): The number of Boolean variables.
        num_clauses (int): The target number of 3-literal clauses.
        seed: For reproducability

    Returns:
        ThreeSat: The ProblemInstance for our BenchmarkDatabase
    """
    rng_instance = random.Random(seed)
    
    # 1. Choose a unique solution (planted_solution)
    random_assignment = [rng_instance.randint(0, 1) for _ in range(num_vars)]
    plant = [(i + 1) if bit == 1 else -(i + 1) for (i, bit) in enumerate(random_assignment)]
    planted_solution_set = set(plant)

    formula = CNF()
    current_clauses = 0

    # 2. Generate clauses that are satisfied by the planted_solution
    while current_clauses < num_clauses:
        # Generate a random 3-literal clause
        # Ensure distinct variables in the clause
        variables_in_clause = rng_instance.sample(range(1, num_vars + 1), 3)
        clause = []
        for var in variables_in_clause:
            # Randomly decide polarity (positive or negative literal)
            literal = var if rng_instance.random() < 0.5 else -var
            clause.append(literal)

        # Check if the generated clause is satisfied by the planted_solution
        is_satisfied_by_planted_solution = False
        for literal in clause:
            if literal in planted_solution_set:
                is_satisfied_by_planted_solution = True
                break
            elif -literal not in planted_solution_set:
                is_satisfied_by_planted_solution = True
                break

        if is_satisfied_by_planted_solution:
            formula.append(clause)
            current_clauses += 1

    with Glucose(bootstrap_with=formula) as solver: 
        solutions = [model for model in solver.enum_models()]

    # Build ThreeSat
    return ThreeSat(expr=formula.clauses, num_vars=num_vars, solutions=solutions, seed=seed)


def populate_3sat_problem_db(db: BenchmarkDatabase, vars_range, problems_per_class = 100, clause_var_ratio = 4.2 ):
    count = 0
    for num_vars in vars_range: 
        for _ in range(problems_per_class):
            target_num_clauses = int(num_vars * clause_var_ratio)
            sat_problem = generate_planted_3sat(num_vars, target_num_clauses, seed=count)
            count += 1

            db.save_problem_instance(sat_problem)
            

