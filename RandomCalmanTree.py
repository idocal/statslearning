import numpy as np
import networkx as nx
import random

NUM_NODES = 16
DENSITY = 0.5
SEED = 42


class RandomCalmanTree:

    operations = {
        "sum": np.sum,
        "max": np.max,
        "prod": np.prod
    }

    def __init__(self, n, p):
        # init graph
        tree = nx.DiGraph()
        leaves = [i for i in range(n)]
        leaf_props = {"type": "leaf", "val": None, "layer": 0}
        leaves_nodes = [(x, leaf_props) for x in leaves]
        tree.add_nodes_from(leaves_nodes)  # add leaves as graph nodes

        # updating variables
        curr_layer = leaves
        num_nodes = n
        n_layer = 1

        # tree construction
        while len(curr_layer) > 1:
            nodes_map = {}
            edges = []
            next_layer_size = max(1, int(len(curr_layer) * p))

            # define candidates from next layer
            next_layer = []
            next_layer_last = num_nodes + next_layer_size
            candidates = [x for x in range(num_nodes, next_layer_last)]

            # random choice for connections between layers
            connections = np.random.choice(candidates, len(curr_layer))
            for j, src in enumerate(connections):
                dest = int(curr_layer[j])
                edges.append((src, dest))

            # create tree nodes only for nodes with children
            # add edges to tree
            for src, dest in edges:
                if src not in nodes_map.keys():
                    nodes_map[src] = num_nodes
                    op = random.choice(list(self.operations.keys()))
                    node_props = {"type": op, "val": None, "layer": n_layer}
                    tree.add_nodes_from([(num_nodes, node_props)])
                    next_layer.append(num_nodes)
                    num_nodes += 1
                tree.add_edge(nodes_map[src], dest)

            # proceed to next layer
            curr_layer = next_layer
            n_layer += 1

        self.tree = tree

    def reset(self):
        for i in range(len(self.tree.nodes)):
            self.tree.nodes[i]['val'] = None

    def _compute(self, node_idx: int):
        node = self.tree.nodes[node_idx]
        if node.get('type') == 'leaf':
            return node['val']

        # recursively compute values for children nodes
        children = self.tree.successors(node_idx)
        c_vals = np.array([self._compute(i) for i in children])

        # apply graph operation on accumulated values
        op = self.operations.get(node['type'])
        val = op(c_vals)
        node['val'] = val
        return val

    def compute(self, x: np.array):
        leaves = [n for n, d in self.tree.nodes(data=True)
                  if d.get('type') == 'leaf']
        assert len(x) == len(leaves), f"num_leaves: {len(leaves)}, x: {len(x)}"

        # assign input values to leaves
        for i, val in enumerate(x):
            self.tree.nodes[i]['val'] = val

        # recursively compute the tree source
        source_idx = max(self.tree.nodes)  # by construction
        return self._compute(source_idx)

    @property
    def leaves(self):
        return [n for n, d in self.tree.nodes(data=True)
                if d.get('type') == 'leaf']

    @property
    def layers(self):
        last = self.tree.nodes[max(self.tree.nodes)]['layer']
        layers = [0] * (last + 1)
        for node in self.tree.nodes:
            layers[node['layer']] += 1


if __name__ == "__main__":
    t = RandomCalmanTree(NUM_NODES, DENSITY)
