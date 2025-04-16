import itertools
import json
import numpy as np
import random
import sqlite3
import math

"""
Graph Schema:
    graph_id: integer primary key
    g: edge representation e_12 e_13 ... e_1n e_23 e_24 ... e_2n ... e_(n-1)n string of 0s and 1s
    n: number of vertices
    p: integer percentage (using this convention rather than float to avoid problematic floating point imprecision in the database)
    clique_counts: list of integers, where the ith element is the number of n-bit strings for which the clique verifier for the graph returns True
        NOTE: this is not the same thing as the number of cliques for each size; rather the number of cliques of size at least i (with the exception of index 0, since any string is a clique of size 0)
        Use num_cliques_of_size to get the number of distinct cliques of a given size
"""

def verify_clique(g, clique, clique_size):
    """
    given g: edge representation of a graph, and a clique: binary string denoting vertices in the clique, return True if the clique is valid
    """
    n = int((1 + (1 + 8*len(g))**0.5) / 2)
    assert len(clique) == n

    # verify clique size
    if sum([1 for v in clique if v == '1']) < clique_size:
        return False

    # verify that two vertices are only in the clique if their edge is in the graph
    edge_idx = 0
    for i in range(n):
        for j in range(i+1, n):
            if g[edge_idx] == '0':
                if clique[i] == '1' and clique[j] == '1':
                    return False
            edge_idx += 1
    return True



class Graph:
    def __init__(self, g, p=None, clique_counts=None, graph_id=None):
        """
        g is edge representation e_12 e_13 ... e_1n e_23 e_24 ... e_2n ... e_(n-1)n string of 0s and 1s
        """
        self.g = g
        self.p = p
        self._clique_counts = clique_counts or []
        self.graph_id = graph_id

        if len(g) != self.n * (self.n - 1) / 2:
            raise ValueError("Invalid edge representation")
        
    @property 
    def n(self):
        return int((1 + (1 + 8*len(self.g))**0.5) / 2)
    
    @property
    def clique_counts(self):
        if not self._clique_counts:
            self.compute_clique_counts()
        return self._clique_counts
    
    def __str__(self):
        return self.g
    
    def num_cliques_of_size(self, k):
        if k <= 0 or k > self.n:
            raise ValueError("Invalid clique size")
        
        clique_counts = self.clique_counts
        if k == len(clique_counts)-1:
            return clique_counts[k]
        return self.clique_counts[k] - self.clique_counts[k+1]

    def as_adjacency_matrix(self):
        adjacency_matrix = np.zeros((self.n, self.n))
        edge_idx = 0
        for i in range(self.n):
            for j in range(i+1, self.n):
                edge = self.g[edge_idx]
                edge_idx += 1
                if edge == '1':
                    adjacency_matrix[i, j] = 1
                    adjacency_matrix[j, i] = 1

        return adjacency_matrix
    
    def compute_clique_counts(self):
        adjacency_matrix = self.as_adjacency_matrix()
        n = self.n
        clique_counts = [0 for _ in range(n+1)]
        clique_counts[0] = 2**n
        clique_counts[1] = n
        clique_counts[2] = sum([1 for e in self.g if e == "1"])

        for i in range(3, n + 1):
            for clique in itertools.combinations(range(n), i):
                if all(adjacency_matrix[u, v] for u, v in itertools.combinations(clique, 2)):
                    clique_counts[i] += 1

        # adjust counts to be cumulative
        for i in range(n-1, 0, -1):
            clique_counts[i] += clique_counts[i+1]

        self._clique_counts = clique_counts
        return clique_counts

    def optimal_grover_iterations(self, clique_size):
        """
        Given a clique size, return the optimal number of Grover iterations to find a clique of at least that size
        """
        if clique_size <= 0 or clique_size > self.n:
            raise ValueError("Invalid clique size")
        
        m = self.clique_counts[clique_size]
        if m == 0:
            raise Exception(f"No cliques of size {clique_size}")

        return math.floor(
            math.pi / (4 * math.asin(math.sqrt(m / 2 ** self.n)))
        )

class Graphs:
    def __init__(self, db_name="graphs.db"):
        self.db_name = db_name
        self._initialize_database()

    def _connect(self):
        return sqlite3.connect(self.db_name)
    
    def _as_graphs(self, rows: list):
        return [Graph(
            g = row[1],
            p = row[2],
            clique_counts = json.loads(row[4]),
            graph_id = row[0]
        ) for row in rows]
    
    def _initialize_database(self):
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS graphs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    g TEXT,
                    p INTEGER,
                    n INTEGER,
                    clique_counts TEXT
                )
                """
            )
            conn.commit()

    def save(self, graph: Graph):
        clique_counts = json.dumps(graph.clique_counts)
        with self._connect() as conn:
            cursor = conn.cursor()
            if graph.graph_id is None:
                cursor.execute(
                    "INSERT INTO graphs (g, p, n, clique_counts) VALUES (?, ?, ?, ?)",
                    (graph.g, graph.p, graph.n, clique_counts)
                )
                graph.graph_id = cursor.lastrowid
            else:
                cursor.execute(
                    "UPDATE graphs SET g = ?, p = ?, n = ?, clique_counts = ? WHERE id = ?",
                    (graph.g, graph.p, graph.n, clique_counts, graph.graph_id)
                )
            conn.commit()

    def delete(self, graph: Graph = None, graph_id=None):
        if graph is None and graph_id is None:
            raise ValueError("Either graph or graph_id must be provided")
        if graph is not None:
            assert graph_id is None or graph_id == graph.graph_id
            graph_id = graph.graph_id

        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM graphs WHERE id = ?",
                (graph_id,)
            )
            conn.commit()

    def get(self, graph_id=None, g=None, p=None, n=None, clique_counts=None):
        if graph_id is None and g is None and p is None and n is None and clique_counts is None:
            raise ValueError("At least one of graph_id, g, p, n, or clique_counts must be provided")

        query_parts = []
        params = []

        if graph_id is not None:
            query_parts.append("id = ?")
            params.append(graph_id)

        if g is not None:
            query_parts.append("g = ?")
            params.append(g)

        if p is not None:
            query_parts.append("p = ?")
            params.append(p)

        if n is not None:
            query_parts.append("n = ?")
            params.append(n)

        if clique_counts is not None:
            query_parts.append("clique_counts = ?")
            params.append(json.dumps(clique_counts))

        query = "SELECT * FROM graphs WHERE " + " AND ".join(query_parts)

        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return self._as_graphs(rows)
        
def get_random_graph(n, p):
    """
    Given n vertices, generate a random graph with edge probability p (as integer percentage)
    """
    num_edges = n*(n-1)//2
    g = "".join(['1' if random.random()*100 < p else '0' for _ in range(num_edges)])
    return Graph(g, p=p)

def generate_database(n_range, p_range, num_graphs, compute_clique_counts=False, db_name="graphs.db", include_existing=True):
    graphs = Graphs(db_name)
    for n in n_range:
        for p in p_range:
            num_existing = len(graphs.get(n=n, p=p)) if include_existing else 0
            for _ in range(num_graphs - num_existing):
                graph = get_random_graph(n, p)
                if compute_clique_counts:
                    graph.compute_clique_counts()
                graphs.save(graph)

if __name__ == "__main__":
    generate_database(range(3, 21), np.linspace(1, 99, 50).tolist(), 100, compute_clique_counts=True)
