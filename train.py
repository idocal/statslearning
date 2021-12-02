import os
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule, Trainer
from DNN import DNN
import pandas as pd
import argparse


TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"


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
    parser.add_argument('name', type=str)
    args = parser.parse_args()
    exp_path = os.path.join('experiments', args.name)
    if not os.path.exists(exp_path):
        raise AttributeError(f"Experiment {args.name} does not exist")

    train_path = os.path.join(exp_path, TRAIN_FILE)
    test_path = os.path.join(exp_path, TEST_FILE)

    trainer = Trainer(max_epochs=100)
    dm = RCTDataModule(train_path=train_path, test_path=test_path)
    model = DNN(n_feats=300)
    trainer.fit(model, dm)
