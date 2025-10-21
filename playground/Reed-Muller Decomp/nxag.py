"""
Networkx implementation of XAG for easily converting to/from tweedledum LogicNetwork
"""

import tempfile
from typing import List

import networkx as nx
import pygraphviz
import sympy
from tweedledum.classical import LogicNetwork, optimize
from tweedledum.synthesis import xag_cleanup2
from tweedledum.utils import xag_export_dot
from collections import defaultdict, deque


def _assign_color_map(G: nx.MultiDiGraph) -> List:
    color_map = []
    for node in G.nodes:
        color_map.append(G.nodes[node].get("color", "gray"))  # Use 'gray' as default
    return color_map


def _build_expression(graph: nx.MultiDiGraph, node, memo=None):
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
    elif "0" == node_label:
        for u, v, key, data in graph.in_edges(node, keys=True, data=True):
            is_negated = data.get("negated", False)
            input_expr = sympy.false

            if is_negated:
                result = sympy.Not(input_expr)
            else:
                result = input_expr
            memo[node] = result
    else:
        # Recursive step: The node is a logic gate.
        gate_type = node_label.upper()
        sub_expressions = []

        # Iterate through the incoming edges to get the parent nodes and edge data.
        for u, v, key, data in graph.out_edges(node, keys=True, data=True):
            is_negated = data.get("negated", False)

            # Recursively build the expression for the parent node.
            parent_expression = _build_expression(graph, v, memo)

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

    return memo[node]


class PyXAG:
    def __init__(self):
        self.graph_stats = None
        self.nodes_at_level = None
        self.longest_path = None
        self.and_count = None
        self.and_dist = None
        self.xor_count = None
        self.xor_dist = None
        self.input_nodes = None

    def optimize_xag(self):
        """
        Convert PyXAG to tweedledum logic network, run optimization, and convert back
        """

        xag = self.to_tweedledum_logicnetwork()
        xag = xag_cleanup2(xag)
        optimize(xag)
        new_pyxag = PyXAG().from_tweedledum_xag(xag)
        self = new_pyxag

    def from_nx(self, G: nx.MultiDiGraph):
        self.G = G
        self.color_map = _assign_color_map(self.G)

        if self.G.nodes and "style" in next(iter(self.G.nodes.values())):
            # this graph needs cleaning -- freshly converted from tweedledum graphviz
            self.cleanup_graph_data()

        self.collect_input_nodes()

        return self

    def from_tweedledum_xag(self, xag: LogicNetwork):
        """
        Build XAG in python from tweedledum xag

        NOTE: IT seems that tweedledum modifies
        their graphs so this memory COULD be corrupted
        """

        fp = tempfile.NamedTemporaryFile()
        xag_export_dot(xag, fp.name)
        dot_str = fp.read()
        # print(dot_str)

        A = pygraphviz.AGraph()
        A.from_string(dot_str)
        self.G = nx.drawing.nx_agraph.from_agraph(A)

        self.cleanup_graph_data()
        self.collect_input_nodes()

        return self
    
    def collect_input_nodes(self):
        self.input_nodes = set()
        for node, data in self.G.nodes(data=True):
            if 'label' in data and 'input' in data['label']:
                self.input_nodes.add(node)

    def cleanup_graph_data(self):
        for node, data in self.G.nodes.data():
            try:
                node_label = int(data["label"])
                if node_label != 0:
                    data["label"] = f"input{node_label}"
                    data["color"] = "green"
                else:
                    data["label"] = "0"
                    data["color"] = "cyan"
            except:
                clr_map = {"XOR": "blue", "AND": "red"}
                if node != "po0":
                    data["color"] = clr_map[data["label"]]
                pass

            del data["fillcolor"]
            del data["shape"]
            del data["style"]

        for _, _, data in self.G.edges.data():
            data["negated"] = data["style"] == "dashed"
            del data["style"]

        self.color_map = _assign_color_map(self.G)

    def to_tweedledum_logicnetwork(self) -> LogicNetwork:
        logic_network = LogicNetwork()

        signals = {}

        for node in nx.topological_sort(self.G):
            # print(node, self.G.nodes[node])

            if "label" in self.G.nodes[node]:
                node_label = self.G.nodes[node]["label"]
                if "input" in node_label:
                    signals[node] = logic_network.create_pi()
                    continue

                if node_label == "0":
                    signals[node] = logic_network.get_constant(False)
                    continue

                # gather parents of this node and the incoming signals
                parents = [
                    (u, v, data) for (u, v, data) in self.G.in_edges(node, data=True)
                ]
                parent_signals = []
                for parent, _, data in parents:
                    if data["negated"]:
                        # create an inverted signal coming from the parent
                        parent_signals.append(logic_network.create_not(signals[parent]))
                    else:
                        parent_signals.append(signals[parent])

                if node_label == "XOR":
                    signals[node] = logic_network.create_xor(
                        parent_signals[0], parent_signals[1]
                    )
                elif node_label == "AND":
                    signals[node] = logic_network.create_and(
                        parent_signals[0], parent_signals[1]
                    )
                else:
                    raise ValueError("Node contained unexpected label")

            else:
                # primary output node
                # gather parent of this node and the incoming signals
                parent, _, data = [
                    (u, v, data) for (u, v, data) in self.G.in_edges(node, data=True)
                ][0]
                if data["negated"]:
                    # create an inverted signal coming from the parent
                    parent_signal = logic_network.create_not(signals[parent])
                else:
                    parent_signal = signals[parent]

                signals[node] = logic_network.create_po(parent_signal)

        return logic_network

    def sympy_expr(self):
        G_rev = self.G.to_directed().reverse()

        nodes = list(nx.topological_sort(G_rev))

        if not nodes:
            # nodes is empty
            return None

        starting_node = 1
        if nodes[starting_node] == "po0":
            # if the graph has not been edited then 0 has no connections and
            # the expression builder will terminate early
            starting_node = 2

        # print("\nSTARTING EXPRESSION BUILDER")
        expr = _build_expression(G_rev, node=nodes[starting_node])

        # check final edge to see if we need to negate our expr
        # first edge is always the one we want, nx was giving me issues so we put it in a loop
        for u, v, data in self.G.in_edges("po0", data=True):
            if data["negated"]:
                expr = sympy.Not(expr)
            return expr

    def compute_graph_stats(self):
        """Compute graph statistics for XAG
        - width of graph (maximum nodes at a level)
        - distribution of ANDs at different levels of the graph
        - Longest path from any input to the output
        - AND and XOR node counts
        """

        if self.graph_stats:
            return self.graph_stats

        queue = deque()

        self.node_level = defaultdict(int)
        in_degree = defaultdict(int)

        for node in self.G.nodes:
            in_deg = self.G.in_degree(node)
            if node != "0" and in_deg == 0:
                queue.append(node)
                self.node_level[node] = 0
            else:
                in_degree[node] = in_deg

        # topological search to find levels
        while queue:
            parent = queue.popleft()

            for u, v in self.G.out_edges(parent):
                # level of a node is the longest path from an input
                self.node_level[v] = max(self.node_level[v], self.node_level[u] + 1)
                in_degree[v] -= 1

                if in_degree[v] == 0:
                    queue.append(v)

        self.nodes_at_level = defaultdict(list)
        for node, level in self.node_level.items():
            self.nodes_at_level[level].append(node)

        self.longest_path = nx.dag_longest_path_length(self.G)
        self.and_dist = [0 for _ in range(self.longest_path)]
        self.xor_dist = [0 for _ in range(self.longest_path)]

        self.xor_count = 0
        self.and_count = 0
        for node, data in self.G.nodes(data=True):
            try:
                if data["label"] == "XOR":
                    self.xor_count += 1
                    self.xor_dist[self.node_level[node]] += 1
                elif data["label"] == "AND":
                    self.and_count += 1
                    self.and_dist[self.node_level[node]] += 1
            except:
                continue

        max_width_level = 0
        max_width = 0
        for level, nodes in self.nodes_at_level.items():
            if len(nodes) > max_width:
                max_width = len(nodes)
                max_width_level = level

        self.graph_stats = {
            # find the level
            "nodes_at_level": [len(self.nodes_at_level.get(lvl, [])) for lvl in range(self.longest_path)],
            "max_width_level": max_width_level,
            "max_width": max_width,
            "longest_path_len": self.longest_path,
            "and_dist_by_level": self.and_dist,
            "xor_dist_by_level": self.xor_dist,
            "and_count": self.and_count,
            "xor_count": self.xor_count,
            "num_nodes": self.G.number_of_nodes(),
        }

        return self.graph_stats

    def draw(self, ax=None):
        self.color_map = _assign_color_map(self.G)
        pos = nx.drawing.nx_agraph.graphviz_layout(self.G, prog="dot")
        nx.draw(self.G, pos, node_color=self.color_map, with_labels=True, ax=ax)
