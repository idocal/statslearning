import numpy as np
import pandas as pd
from RandomCalmanTree import RandomCalmanTree
from tqdm import tqdm


DATA_FILE = "data.csv"
NUM_FEATURES = 300
TREE_DENSITY = 0.5
NUM_EXAMPLES = 10_000


if __name__ == "__main__":
    tree = RandomCalmanTree(NUM_FEATURES, TREE_DENSITY)
    data = None
    for i in tqdm(range(NUM_EXAMPLES)):
        x = np.random.rand(NUM_FEATURES)
        y = tree.compute(x)
        tree.reset()
        row = np.append(x, y)
        if data is None:
            data = row
        else:
            data = np.row_stack((data, row))
    pd.DataFrame(data).to_csv(DATA_FILE)
    print(f"Dataset has been successfully saved to: {DATA_FILE}")
