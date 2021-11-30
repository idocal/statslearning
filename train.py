import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule, Trainer
from DNN import DNN
import pandas as pd


TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"


class RCTDataset(Dataset):

    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file, header=None)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        y = self.df.iloc[idx, -1]
        x = torch.FloatTensor(row[1:-1].to_numpy())
        return x, y


class RCTDataModule(LightningDataModule):

    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        return

    def train_dataloader(self):
        train_dataset = RCTDataset(TRAIN_FILE)
        return DataLoader(train_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        test_dataset = RCTDataset(TEST_FILE)
        return DataLoader(test_dataset, batch_size=self.batch_size)


if __name__ == "__main__":
    trainer = Trainer()
    dm = RCTDataModule()
    model = DNN(n_feats=300)
    trainer.fit(model, dm)
