import os
import argparse
import numpy as np
import pandas as pd
from RandomCalmanTree import RandomCalmanTree
from tqdm import tqdm
import networkx as nx


TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
NUM_FEATURES = 300
TREE_DENSITY = 0.5
NUM_EXAMPLES = 10_000
TRAIN_TEST_RATIO = 0.9


def generate_data(n, tree, filepath):
    data = None
    for i in tqdm(range(n)):
        x = np.random.rand(NUM_FEATURES)
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
    args = parser.parse_args()

    # create a new experiment
    exp_path = os.path.join('experiments', args.name)
    if os.path.exists(exp_path):
        raise AttributeError(f"Experiment {args.name} already exists.")
    os.makedirs(exp_path)

    # generate a random computational tree
    tree = RandomCalmanTree(NUM_FEATURES, TREE_DENSITY)

    # generate train and test data
    n_train = int(TRAIN_TEST_RATIO * NUM_EXAMPLES)
    train_path = os.path.join(exp_path, TRAIN_FILE)
    generate_data(n_train, tree, train_path)
    n_test = int(NUM_EXAMPLES - n_train)
    test_path = os.path.join(exp_path, TEST_FILE)
    generate_data(n_test, tree, test_path)

    # store the random tree
    tree_path = os.path.join(exp_path, 'f.gpickle')
    nx.write_gpickle(tree, tree_path)
    print(f"Function tree has been saved to: {tree_path}")
