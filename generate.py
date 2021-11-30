import numpy as np
import networkx as nx
import random

NUM_NODES = 32
DENSITY = 0.5
SEED = 42


operations = {
    "sum": np.sum,
    "product": np.prod
}


def random_calman_tree(n, p, ops=operations):
    # init graph
    tree = nx.DiGraph()
    leaves = [i for i in range(NUM_NODES)]
    leaves_nodes = [(x, {"type": "leaf", "val": None}) for x in leaves]
    tree.add_nodes_from(leaves_nodes)  # add leaves as graph nodes

    # updating variables
    n = NUM_NODES
    curr_layer = leaves
    next_layer = []

    # tree construction
    while len(curr_layer) > 1:
        while len(curr_layer):
            node = curr_layer[0]
            # add random operation node to next layer
            op = random.choice(list(ops.keys()))
            tree.add_nodes_from([(n, {"type": op, "val": None})])
            tree.add_edge(n, node)  # each node has at least one connection
            next_layer.append(n)

            # find connections from same layer
            neighbors = [0]
            for i, candidate in enumerate(curr_layer[1:]):
                if random.uniform(0, 1) < p:
                    tree.add_edge(n, candidate)
                    neighbors += [i + 1]

            # remove redundancy by forcing at least one neighbor
            if len(neighbors) == 1 and len(curr_layer) > 1:
                candidates = [j for j in range(len(curr_layer))][1:]
                neighbor = random.choice(candidates)
                tree.add_edge(n, curr_layer[neighbor])
                neighbors += [neighbor]

            # update current layer
            for idx in reversed(neighbors):
                del curr_layer[idx]

            n += 1

        # progress to next layer
        curr_layer = next_layer
        next_layer = []

    return tree


def reset(graph: nx.DiGraph):
    for i in range(len(graph.nodes)):
        graph.nodes[i]['val'] = None


def _compute(tree: nx.DiGraph, node_idx: int):
    node = tree.nodes[node_idx]
    if node['type'] == 'leaf':
        return node['val']

    # recursively compute values for children nodes
    children = tree.successors(node_idx)
    child_values = np.array([_compute(tree, i) for i in children])

    # apply graph operation on accumulated values
    op = operations.get(node['type'])
    val = op(child_values)
    node['val'] = val
    return val


def compute(tree: nx.DiGraph, x: np.array):
    leaves = [n for n, d in tree.nodes(data=True) if d['type'] == 'leaf']
    assert len(x) == len(leaves)

    # assign input values to leaves
    for i, val in enumerate(x):
        tree.nodes[i]['val'] = val

    # recursively compute the tree source
    source_idx = max(tree.nodes)  # by construction
    return _compute(tree, source_idx)


if __name__ == "__main__":
    t = random_calman_tree(NUM_NODES, DENSITY)
