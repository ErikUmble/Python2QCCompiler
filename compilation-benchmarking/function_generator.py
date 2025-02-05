import random
import queue


class Node:
    def __init__(self, value=None, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right


def generate_random_boolean_expression_iterative(num_inputs, depth):
    """
    Generates a random boolean expression represented by a tree (iterative approach).

    Args:
      num_inputs: The number of input variables to use in the expression.
      depth: The maximum depth of the tree.

    Returns:
      A Node object representing the root of the expression tree.
    """
    if depth == 0:
        return Node(random.choice([f"x{i}" for i in range(num_inputs)]))

    root = Node(random.choice(["AND", "OR", "NOT"]))
    nodes = queue.Queue()
    # Queue that store the current level and operation of this node
    nodes.put((0, root))

    while nodes:
        current_node = nodes.get()
        if current_node.value == "NOT":
            # NOT is a unary operator
            current_node.left = generate_random_boolean_expression_iterative(
                num_inputs, depth - 1
            )
        else:
            # Binary operator
            current_node.left = generate_random_boolean_expression_iterative(
                num_inputs, depth - 1
            )
            current_node.right = generate_random_boolean_expression_iterative(
                num_inputs, depth - 1
            )

    return root


def to_string(node):
    """
    Converts an expression tree to a string representation.

    Args:
      node: The root node of the expression tree.

    Returns:
      A string representation of the expression.
    """
    if node.left and node.right:
        return f"({to_string(node.left)} {node.value} {to_string(node.right)})"
    elif node.left:
        return f"({node.value} {to_string(node.left)})"
    else:
        return node.value


def get_classical_function(expression_tree, variables, name="f"):
    """
    Generates a classical function string from an expression tree.

    Args:
      expression_tree: The root node of the expression tree.
      variables: A list of variable names.
      name: The name of the function.

    Returns:
      A string representation of the classical function.
    """
    expression_str = (
        to_string(expression_tree)
        .replace("AND", "and")
        .replace("OR", "or")
        .replace("NOT", "not")
    )  # Adjust operators
    return f"""
@classical_function
def {name}({", ".join([f"{var} : Int1" for var in variables])}) -> Int1:
    return {expression_str}
    """


def get_python_function(expression_tree, variables, name="f"):
    """
    Generates a Python function string from an expression tree.

    Args:
      expression_tree: The root node of the expression tree.
      variables: A list of variable names.
      name: The name of the function.

    Returns:
      A string representation of the Python function.
    """
    expression_str = (
        to_string(expression_tree)
        .replace("AND", "and")
        .replace("OR", "or")
        .replace("NOT", "not")
    )  # Adjust operators
    return f"""
def {name}({", ".join(variables)}):
    return {expression_str}
    """


def generate_benchmark_functions(filename="benchmark_functions.py"):
    """
    Generates benchmark functions using the tree-based approach.

    Args:
      filename: The name of the file to write the functions to.
    """
    with open(filename, "w") as f:
        f.write("from qiskit.circuit.classicalfunction import classical_function\n")
        f.write("from qiskit.circuit.classicalfunction.types import Int1\n")

        index = {}
        for num_vars in range(2, 20):
            for complexity in range(1, 10):  # Adjust complexity range as needed
                expression_tree = generate_random_boolean_expression_iterative(
                    num_vars, complexity
                )
                variables = [f"x{i}" for i in range(num_vars)]
                f.write(
                    get_classical_function(
                        expression_tree, variables, name=f"cf_v{num_vars}_c{complexity}"
                    )
                )
                f.write(
                    get_python_function(
                        expression_tree, variables, name=f"pf_v{num_vars}_c{complexity}"
                    )
                )
                f.write("\n")

                index[f"({num_vars},{complexity})"] = {
                    "variables": variables,
                    "statement": to_string(expression_tree),
                    "classical_function": f"cf_v{num_vars}_c{complexity}",
                    "python_function": f"pf_v{num_vars}_c{complexity}",
                }
        f.write("benchmark_functions = " + str(index))


if __name__ == "__main__":
    generate_benchmark_functions()
