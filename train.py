import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import networkx as nx
from DNN import DNN
from GPDNN import GPDNN
from CNN import CNN
import pandas as pd
import argparse


TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
META_FILE = "meta.json"
TREE_FILE = "f.gpickle"
LOG_DIR = "lightning_logs"


class RCTDataset(Dataset):

    def __init__(self, csv_file):
        df = pd.read_csv(csv_file, header=None)
        self.df = df.iloc[1:, 1:]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y = self.df.iloc[idx, -1]
        x = torch.FloatTensor(row[:-1].to_numpy())
        return x, y


class RCTDataModule(LightningDataModule):

    def __init__(self, train_path, test_path, batch_size=32):
        super().__init__()
        self.batch_size = batch_size
        self.train_path = train_path
        self.test_path = test_path

    def prepare_data(self) -> None:
        return

    def train_dataloader(self):
        train_dataset = RCTDataset(self.train_path)
        return DataLoader(train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        test_dataset = RCTDataset(self.test_path)
        return DataLoader(test_dataset, batch_size=self.batch_size)


if __name__ == "__main__":
    # parse args from user
    parser = argparse.ArgumentParser()
    parser.add_argument('names', type=str, nargs='+')
    args = parser.parse_args()

    for name in args.names:
        exp_path = os.path.join('experiments', name)
        if not os.path.exists(exp_path):
            raise AttributeError(f"Experiment {name} does not exist")

        # define training and test data
        train_path = os.path.join(exp_path, TRAIN_FILE)
        test_path = os.path.join(exp_path, TEST_FILE)
        meta = json.load(open(os.path.join(exp_path, META_FILE)))
        # n_feats = len(pd.read_csv(test_path, header=None).columns) - 2
        n_feats = meta['features']

        # train shallow neural network the model
        dm = RCTDataModule(train_path=train_path, test_path=test_path)
        logger = TensorBoardLogger(LOG_DIR, version=name, name="shallow")
        trainer = Trainer(max_epochs=50, logger=logger)
        shallow = DNN(n_feats=n_feats, layers=[])
        trainer.fit(shallow, dm)

        # graph prior dnn
        tree_path = os.path.join(exp_path, TREE_FILE)
        rct = nx.read_gpickle(tree_path)
        logger = TensorBoardLogger(LOG_DIR, version=name, name="gpdnn")
        trainer = Trainer(max_epochs=50, logger=logger)
        gpdnn = GPDNN(rct)
        trainer.fit(gpdnn, dm)

        # train one layer dnn
        one_dnn = DNN(n_feats=n_feats, layers=[n_feats*2])
        logger = TensorBoardLogger(LOG_DIR, version=name, name="one_layer")
        trainer = Trainer(max_epochs=50, logger=logger)
        trainer.fit(one_dnn, dm)

        # two layer dnn
        one_dnn = DNN(n_feats=n_feats, layers=[n_feats*2])
        logger = TensorBoardLogger(LOG_DIR, version=name, name="two_layer")
        trainer = Trainer(max_epochs=50, logger=logger)
        two_dnn = DNN(n_feats=n_feats, layers=[n_feats, n_feats])
        trainer.fit(two_dnn, dm)

        # CNN
        one_dnn = DNN(n_feats=n_feats, layers=[n_feats*2])
        logger = TensorBoardLogger(LOG_DIR, version=name, name="cnn")
        trainer = Trainer(max_epochs=50, logger=logger)
        cnn = CNN(n_feats=n_feats)
        trainer.fit(cnn, dm)
