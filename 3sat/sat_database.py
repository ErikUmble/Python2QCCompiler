import json
import math
import random
import sqlite3
from typing import Any, List, Optional, Tuple, Dict


# --- ThreeSat Class ---


class ThreeSat:
    def __init__(
        self,
        expr: List[Tuple[int, int, int]],
        num_vars: int,
        solutions: Optional[List[List[int]]] = None,
        seed: Optional[int] = None,
        sat_id: Optional[int] = None,
    ):
        """
        3SAT Circuit Representation -- List of Clauses List[Tuple[int, int, int]]
        the absolute value is the index, the negation means apply not
        indices start at 1
        [(1, 2, 3), (-4, 1, 2)] = (x1 V x2 V x3) ^ (~x4 V x1 V x2)

        Args:
            expr: The 3SAT expression as a list of 3-literal clauses.
            solutions: Optional pre-computed list of satisfying assignments (as lists of integers).
            seed: Optional seed used for generation.
            sat_id: Optional database ID.
        """
        if not all(len(clause) == 3 for clause in expr):
            raise ValueError(
                "All clauses in the expression must have exactly 3 literals."
            )

        # flatten expression and collect the unique variables into a sorted list
        # Sorting ensures a consistent order for solutions
        self.number_of_vars = num_vars
        self.vars: List[int] = sorted(
            list(set(abs(var) for clause in expr for var in clause))
        )
        self.expr: List[Tuple[int, int, int]] = expr
        self._solutions: Optional[List[List[int]]] = solutions
        self.seed: Optional[int] = seed
        self.sat_id: Optional[int] = sat_id
        self._pretty_printed_strings = None

    @property
    def num_vars(self) -> int:
        """Number of variables in the expression (including any that may not show up)."""
        return self.number_of_vars

    @property
    def num_clauses(self) -> int:
        """Number of clauses in the expression."""
        return len(self.expr)

    @property
    def solutions(self) -> List[List[int]]:
        """
        List of satisfying assignments. Computes them if not already available.
        Each solution is a list of booleans corresponding to the sorted variable list `self.vars`.
        """
        if self._solutions is None:
            self.compute_solutions()
        # We check None above, so assert tells type checkers it's not None here
        assert self._solutions is not None
        return self._solutions

    def to_dict(self) -> Dict[str, Any]:
        """Converts the ThreeSat instance to a JSON-serializable dictionary."""
        # Make sure solutions are computed if they are intended to be part of the serialization
        return {
            "expr": self.expr,  # Tuples will be converted to lists by json.dumps
            "num_vars": self.num_vars,
            # Serialize _solutions, which might be None or List[List[int]]
            "solutions": self._solutions,
            "seed": self.seed,
            "sat_id": self.sat_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThreeSat":
        """Creates a ThreeSat instance from a dictionary."""
        expr_from_json = data.get("expr")
        # JSON loads tuples as lists, so convert inner lists back to tuples
        expr = [tuple(clause) for clause in expr_from_json] if expr_from_json else []

        return cls(
            expr=expr,
            num_vars=data.get("num_vars", 0), # Default num_vars if not present
            solutions=data.get("solutions"),   # This will be List[List[int]] or None
            seed=data.get("seed"),
            sat_id=data.get("sat_id"),
        )

    def __str__(self) -> str:
        """String representation of the 3SAT expression."""

        def format_literal(lit):
            return f"~x{abs(lit)}" if lit < 0 else f"x{lit}"

        clause_strs = []
        for clause in self.expr:
            clause_strs.append(
                "(" + " V ".join(format_literal(lit) for lit in clause) + ")"
            )
        return " ^ ".join(clause_strs)

    def compute_solutions(self) -> List[List[int]]:
        """
        Computes all satisfying assignments using a truth table approach.
        Stores the result in self._solutions and returns it.
        """
        clauses = [list(clause) for clause in self.expr]
        solver = Glucose(bootstrap_with=clauses)
        self._solutions = [sol for sol in solver.enum_models()]
        return self._solutions

    def optimal_grover_iterations(self) -> int:
        """
        Calculate the optimal number of Grover iterations to find a satisfying assignment.
        Raises Exception if there are no solutions.
        """
        num_solutions = len(self.solutions)  # Ensures solutions are computed if needed
        if num_solutions == 0:
            # Allow returning 0 or raising an error depending on use case
            # raise Exception(f"No solutions exist for this 3SAT instance (ID: {self.sat_id}). Cannot calculate Grover iterations.")
            return 0  # Or handle as appropriate if Grover shouldn't run

        total_states = 2**self.num_vars
        if num_solutions == total_states:
            return 0  # All states are solutions, no search needed.

        # Grover's algorithm optimal iterations formula
        return math.floor(
            math.pi / (4 * math.asin(math.sqrt(num_solutions / total_states)))
        )

    def print_summary(self):
        """
        Prints a formatted summary of the 3SAT instance.

        Args:
            max_expr_length: Maximum characters to display for the expression string.
                             Set to None for unlimited length.
        """
        print("--- 3SAT Instance Summary ---")

        # --- Basic Info ---
        print(
            f"{'Database ID (sat_id):':<25} {self.sat_id if self.sat_id is not None else 'N/A (unsaved)'}"
        )
        print(
            f"{'Generation Seed:':<25} {self.seed if self.seed is not None else 'Unknown'}"
        )

        # --- Problem Size ---
        n_vars = self.num_vars
        n_clauses = self.num_clauses
        print(f"{'Number of Variables:':<25} {n_vars}")
        print(f"{'Number of Clauses:':<25} {n_clauses}")

        # --- Expression ---
        display_expr = str(self)  # Get the formatted expression string
        print(f"{'Expression:':<25} {display_expr}")
        # Optionally show variables if needed:
        # print(f"{'Variables (ordered):':<25} {self.vars}")

        # --- Solution Info ---
        print("\n--- Solution Information ---")
        # Check _solutions directly to avoid triggering computation just for summary
        if self._solutions is None:
            print(f"{'Status:':<25} Solutions not yet computed.")
            print(f"{'':<25} (Access instance.solutions to compute)")
        else:
            num_sols = len(self._solutions)
            print(f"{'Status:':<25} Solutions computed.")
            print(f"{'Number Found:':<25} {num_sols}")

            if n_vars >= 0:  # Avoid issues if somehow n_vars is negative
                total_assignments = 2**n_vars
                if total_assignments > 0:
                    percentage = (num_sols / total_assignments) * 100
                    print(
                        f"{'Percentage Solutions:':<25} {percentage:.4f}% ({num_sols}/{total_assignments})"
                    )
                else:  # Only happens if n_vars = 0
                    print(f"{'Total Assignments:':<25} {total_assignments} (for n=0)")

                # --- Grover Iterations ---
                # Check if solutions exist and not all possibilities are solutions
                if 0 < num_sols < total_assignments:
                    try:
                        grover_iters = self.optimal_grover_iterations()
                        print(f"{'Optimal Grover Iterations:':<25} {grover_iters}")
                    except Exception as e:
                        # Catch potential errors from optimal_grover_iterations if any
                        print(
                            f"{'Optimal Grover Iterations:':<25} Error calculating ({e})"
                        )
                elif num_sols == total_assignments:
                    print(
                        f"{'Optimal Grover Iterations:':<25} 0 (all states are solutions)"
                    )
                else:  # num_sols == 0
                    print(f"{'Optimal Grover Iterations:':<25} N/A (no solutions)")
            else:
                print(f"{'Solution Percentage:':<25} N/A (invalid number of variables)")
                print(
                    f"{'Optimal Grover Iterations:':<25} N/A (invalid number of variables)"
                )

        print("-----------------------------")

    def pretty_print_solutions(self):
        """
        Prints the list of solutions with the binary representation

                                     |------x3
                                     v   v------x1
        x1 = 0, x2 = 0, x3 = 1 ----> 1 0 0
                                       ^-----x2
        """
        if self._pretty_printed_strings is None:
            if self._solutions is None:
                print(f"{'Status:':<25} Solutions not yet computed.")
                print(f"{'':<25} (Access instance.solutions to compute)")
            else:
                strings = list()
                for sol in self._solutions:
                    string = "".join(["1" if val else "0" for val in sol])
                    strings.append(string)
                self._pretty_printed_strings = strings
        print(self._pretty_printed_strings)


# --- Database Manager Class ---


class ThreeSatDB:
    def __init__(self, db_name="3sat.db"):
        """Initializes the database connection and creates the table if it doesn't exist."""
        self.db_name = db_name
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        """Establishes a connection to the SQLite database."""
        return sqlite3.connect(self.db_name)

    def _initialize_database(self):
        """Creates the 'three_sat' table if it's not already present."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS three_sat (
                    sat_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    expr TEXT NOT NULL,           -- Store JSON representation of List[Tuple[int, int, int]]
                    num_vars INTEGER NOT NULL,
                    num_clauses INTEGER NOT NULL, -- Added for easier querying
                    seed INTEGER,                 -- Seed used for generation (optional)
                    solutions TEXT                -- Store JSON representation of List[List[bool]] (optional)
                )
                """
            )
            # Optional: Create indices for faster lookups
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_num_vars ON three_sat (num_vars)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_num_clauses ON three_sat (num_clauses)"
            )
            conn.commit()

    def _as_3sat_instances(self, rows: list) -> List[ThreeSat]:
        """Converts raw database rows into a list of ThreeSat objects."""
        instances = []
        for row in rows:
            sat_id, expr_json, num_vars, num_clauses, seed, solutions_json = row

            # Safely load JSON, handling potential None values
            expr = json.loads(expr_json) if expr_json else []
            # Convert inner lists back to tuples if necessary (depends on how saved)
            # Assuming json saves tuples as lists, convert back
            expr = [tuple(clause) for clause in expr]

            solutions = json.loads(solutions_json) if solutions_json else None

            instance = ThreeSat(
                expr=expr,
                num_vars=num_vars,
                solutions=solutions,  # Pass pre-computed solutions if available
                seed=seed,
                sat_id=sat_id,
            )
            # Optional verification: Check if loaded num_vars matches computed one
            # assert instance.num_vars == num_vars, f"Mismatch in num_vars for sat_id {sat_id}"
            # assert instance.num_clauses == num_clauses, f"Mismatch in num_clauses for sat_id {sat_id}"
            instances.append(instance)
        return instances

    def save(self, sat_instance: ThreeSat):
        """Saves or updates a ThreeSat instance in the database."""
        # Ensure solutions are computed if they are to be saved (optional, saves compute later)
        # Depending on policy, you might compute solutions here or save None
        # solutions_list = sat_instance.solutions # Compute if needed

        # Serialize complex fields to JSON
        expr_json = json.dumps(sat_instance.expr)
        # Save solutions only if they have been computed, otherwise save NULL
        solutions_json = json.dumps(sat_instance.solutions())

        with self._connect() as conn:
            cursor = conn.cursor()
            if sat_instance.sat_id is None:
                # Insert new record
                cursor.execute(
                    "INSERT INTO three_sat (expr, num_vars, num_clauses, seed, solutions) VALUES (?, ?, ?, ?, ?)",
                    (
                        expr_json,
                        sat_instance.num_vars,
                        sat_instance.num_clauses,
                        sat_instance.seed,
                        solutions_json,
                    ),
                )
                sat_instance.sat_id = (
                    cursor.lastrowid
                )  # Get the new ID and assign it back
            else:
                # Update existing record
                cursor.execute(
                    """UPDATE three_sat 
                       SET expr = ?, num_vars = ?, num_clauses = ?, seed = ?, solutions = ? 
                       WHERE sat_id = ?""",
                    (
                        expr_json,
                        sat_instance.num_vars,
                        sat_instance.num_clauses,
                        sat_instance.seed,
                        solutions_json,
                        sat_instance.sat_id,
                    ),
                )
            conn.commit()

    def delete(
        self, sat_instance: Optional[ThreeSat] = None, sat_id: Optional[int] = None
    ):
        """Deletes a ThreeSat instance from the database by object or ID."""
        if sat_instance is None and sat_id is None:
            raise ValueError("Either sat_instance or sat_id must be provided")

        if sat_instance is not None:
            # Ensure consistency if both are provided
            if sat_id is not None and sat_id != sat_instance.sat_id:
                raise ValueError("Provided sat_id does not match sat_instance.sat_id")
            id_to_delete = sat_instance.sat_id
            if id_to_delete is None:
                # Should we try to find it based on content? Or just raise error?
                raise ValueError("Cannot delete instance: sat_instance has no sat_id.")
        else:
            # Assert sat_id is not None (due to initial check)
            assert sat_id is not None
            id_to_delete = sat_id

        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM three_sat WHERE sat_id = ?", (id_to_delete,))
            conn.commit()
            # Optionally, clear the id from the passed object if deletion was successful
            # if sat_instance is not None and conn.total_changes > 0:
            #     sat_instance.sat_id = None

    def get(
        self,
        sat_id: Optional[int] = None,
        num_vars: Optional[int] = None,
        num_clauses: Optional[int] = None,
        seed: Optional[int] = None,
        expr: Optional[List[Tuple[int, int, int]]] = None,
    ) -> List[ThreeSat]:
        """Retrieves ThreeSat instances based on specified criteria."""

        query_parts = []
        params: List[Any] = []  # Specify type for params

        if sat_id is not None:
            query_parts.append("sat_id = ?")
            params.append(sat_id)
        if num_vars is not None:
            query_parts.append("num_vars = ?")
            params.append(num_vars)
        if num_clauses is not None:
            query_parts.append("num_clauses = ?")
            params.append(num_clauses)
        if seed is not None:
            query_parts.append("seed = ?")
            params.append(seed)
        if expr is not None:
            # Careful: Comparing serialized JSON in SQL can be inefficient/tricky
            # It's better to query by other indexed fields if possible.
            query_parts.append("expr = ?")
            params.append(json.dumps(expr))  # Serialize for comparison

        if not query_parts:
            # Fetch all if no criteria given? Or raise error? Let's fetch all for now.
            query = "SELECT * FROM three_sat"
            # raise ValueError("At least one search criterion must be provided")
        else:
            query = "SELECT * FROM three_sat WHERE " + " AND ".join(query_parts)

        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return self._as_3sat_instances(rows)


# --- Generation Functions ---


def get_random_3sat(
    num_vars: int, num_clauses: int, seed: Optional[int] = None
) -> ThreeSat:
    """
    Generates a random 3SAT instance.
    Ensures exactly 3 distinct variables per clause and random negation.
    """
    if num_vars < 3:
        raise ValueError(
            "Number of variables must be at least 3 to form 3-literal clauses."
        )

    if seed is not None:
        random.seed(seed)

    variables = list(range(1, num_vars + 1))
    expr = []
    for _ in range(num_clauses):
        # Sample 3 distinct variables
        clause_vars = random.sample(variables, 3)

        # Randomly negate each variable
        clause = tuple(v * random.choice([-1, 1]) for v in clause_vars)
        expr.append(clause)

    return ThreeSat(expr=expr, num_vars=num_vars, seed=seed)


def generate_3sat_database(
    num_vars_range: range,
    num_clauses_range: range,  # Use a range or list for num_clauses
    num_instances_per_config: int,
    compute_solutions: bool = False,
    db_name="3sat.db",
    include_existing: bool = True,
):
    """
    Populates the database with randomly generated 3SAT instances.

    Args:
        num_vars_range: Range of number of variables for instances.
        num_clauses_range: Range of number of clauses for instances.
        num_instances_per_config: How many instances to generate for each (n_vars, n_clauses) pair.
        compute_solutions: If True, compute and store solutions (can be slow for large instances).
        db_name: Name of the database file.
        include_existing: If True, checks existing count and only generates missing ones.
                          If False, generates `num_instances_per_config` new instances regardless.
    """
    db = ThreeSatDB(db_name)

    for n_vars in num_vars_range:
        # Ensure n_vars >= 3 for valid 3SAT generation
        if n_vars < 3:
            print(f"Skipping n_vars={n_vars} as it's less than 3.")
            continue

        for n_clauses in num_clauses_range:
            # print(f"Generating for n_vars={n_vars}, n_clauses={n_clauses}...")

            num_to_generate = num_instances_per_config
            if include_existing:
                # Check how many exist for this specific configuration
                # Note: Doesn't check for specific seeds, just the counts
                existing_instances = db.get(num_vars=n_vars, num_clauses=n_clauses)
                num_existing = len(existing_instances)
                num_to_generate = max(0, num_instances_per_config - num_existing)
                # print(f"  Found {num_existing} existing. Need to generate {num_to_generate} more.")

            if num_to_generate == 0:
                continue

            for i in range(num_to_generate):
                # Generate a unique seed perhaps based on config and index? Or use system random.
                # Using system random default here. Provide a seed if reproducibility needed.
                instance_seed = random.randint(
                    0, 2**32 - 1
                )  # Example of generating a seed

                sat_instance = get_random_3sat(n_vars, n_clauses, seed=instance_seed)

                if compute_solutions:
                    print(
                        f"  Computing solutions for instance {i + 1}/{num_to_generate} (seed={instance_seed})..."
                    )
                    sat_instance.compute_solutions()  # Compute solutions before saving
                    print(f"    Found {len(sat_instance.solutions)} solutions.")

                db.save(sat_instance)
                print(
                    f"  Saved instance {i + 1}/{num_to_generate} with ID {sat_instance.sat_id}"
                )


def get_sat_with_one_solution(num_vars: int, seed: Optional[int] = None) -> ThreeSat:
    """
    Generate a random SAT problem with a unique solution.

    The idea comes from the idea of arithmetic equations with a single solution.
    3x + 2 - 5 = 0 is a SAT instance with ONE solution x = 1
    If we can encode this into a boolean expression and then reduce that expression to CNF
    then we have a SAT problem with one solution. Hopefully this produces better results with Grover

    As a bonus we can start to test the capability of tranlating basic arithmetic to quantum
    """
