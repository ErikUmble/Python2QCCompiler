import random

operators = ("and", "or", "not", "^")

def random_statement(num_vars, complexity, hop_limit=50, best_statement="", best_variable_count=0):
    global num_errors
    if num_vars < 2:
        raise ValueError("num_vars must be at least 2")

    if complexity < 1:
        raise ValueError("complexity must be at least 1")

    variables = ["x" + str(i) for i in range(num_vars)]
    statements = list(variables)

    statement = random.choice(statements)
    for i in range(complexity):
        # in each iteration, generate a new statement for the collection of available statements
        # and also build on to the current statement, increasing it's complexity by 1

        operator = random.choice(operators)
        s1 = random.choice(statements)
        s2 = random.choice(statements)
        s3 = random.choice(statements)
        
        if operator == "not":
            statements.append(f"not ({s1})")
            statement = f"not ({statement})"
        else:
            statements.append(f"({s1}) {operator} ({s2})")
            statement = f"({statement}) {operator} ({s3})"


    variables_used = 0
    for v in variables:
        if v in statement:
            variables_used += 1

    if variables_used == num_vars: 
        return statement, variables
    
    if variables_used > best_variable_count:
        best_variable_count = variables_used
        best_statement = statement

    if hop_limit == 0:
        print(f"Warning: statement ({num_vars},{complexity}) only uses {best_variable_count} variables; skipping")
        return None, None
    
    return random_statement(num_vars, complexity, hop_limit - 1, best_statement, best_variable_count)

def get_classical_function(statement, variables, name="f"):
    return f"""
@classical_function
def {name}({", ".join([f"{var} : Int1" for var in variables])}) -> Int1:
    return {statement}
    """

def get_python_function(statement, variables, name="f"):
    return f"""
def {name}({", ".join(variables)}):
    return {statement}
    """

def generate_benchmark_functions(filename="benchmark_functions.py"):
    with open(filename, "w") as f:
        f.write("from qiskit.circuit.classicalfunction import classical_function\n")
        f.write("from qiskit.circuit.classicalfunction.types import Int1\n")

        index = {}
        for num_vars in range(2, 20):
            for complexity in range(num_vars - 1, num_vars + 15):
                statement, variables = random_statement(num_vars, complexity)
                if statement is None:
                    continue
                f.write(get_classical_function(statement, variables, name=f"cf_v{num_vars}_c{complexity}"))
                f.write(get_python_function(statement, variables, name=f"pf_v{num_vars}_c{complexity}"))
                f.write("\n")

                index[f"({num_vars},{complexity})"] = {
                    "variables": variables,
                    "statement": statement,
                    "classical_function": f"cf_v{num_vars}_c{complexity}",
                    "python_function": f"pf_v{num_vars}_c{complexity}"
                }
        f.write("benchmark_functions = " + str(index))

if __name__ == "__main__":
    generate_benchmark_functions()