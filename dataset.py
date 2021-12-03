import os
import json
import argparse
import numpy as np
import pandas as pd
from RandomCalmanTree import RandomCalmanTree
from tqdm import tqdm
import networkx as nx


TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
META_FILE = "meta.json"
NUM_FEATURES = 16
TREE_DENSITY = 0.5
NUM_EXAMPLES = 100_000
TRAIN_TEST_RATIO = 0.9


def generate_data(n, features, tree, filepath):
    data = None
    for i in tqdm(range(n)):
        x = np.random.uniform(low=0, high=2, size=features)
        y = tree.compute(x)
        tree.reset()
        row = np.append(x, y)
        if data is None:
            data = row
        else:
            data = np.row_stack((data, row))
    pd.DataFrame(data).to_csv(filepath)
    print(f"Dataset has been successfully saved to: {filepath}")


if __name__ == "__main__":
    # parse user args
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str)
    parser.add_argument('-f', '--features', default=NUM_FEATURES, type=int)
    parser.add_argument('-n', '--examples', default=NUM_EXAMPLES, type=int)
    parser.add_argument('-d', '--density', default=TREE_DENSITY, type=float)

    args = parser.parse_args()

    # create a new experiment
    exp_path = os.path.join('experiments', args.name)
    if os.path.exists(exp_path):
        raise AttributeError(f"Experiment {args.name} already exists.")
    os.makedirs(exp_path)

    # generate a random computational tree
    tree = RandomCalmanTree(args.features, args.density)

    # generate train and test data
    n_train = int(TRAIN_TEST_RATIO * args.examples)
    train_path = os.path.join(exp_path, TRAIN_FILE)
    generate_data(n_train, args.features, tree, train_path)
    n_test = int(args.examples - n_train)
    test_path = os.path.join(exp_path, TEST_FILE)
    generate_data(n_test, args.features, tree, test_path)

    # store the random tree
    tree_path = os.path.join(exp_path, 'f.gpickle')
    nx.write_gpickle(tree, tree_path)
    print(f"Function tree has been saved to: {tree_path}")

    # store metadata
    metadata = {
        'features': args.features,
        'examples': args.examples,
        'density': args.density
    }

    meta_path = os.path.join(exp_path, META_FILE)
    json.dump(metadata, open(meta_path, 'w'))
    print(f"Metadata has been saved to: {meta_path}")
