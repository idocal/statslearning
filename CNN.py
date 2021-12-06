import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl


class CNN(pl.LightningModule):

    def __init__(self, n_feats, channels=[], k=3):
        super().__init__()
        self.conv1 = nn.Conv1d(1, n_feats, kernel_size=k)
        self.fc1 = nn.Linear(n_feats * (n_feats - k + 1), 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.type(torch.FloatTensor)  # necessary for MSE
        x = x.view(x.size(0), -1)
        y_hat = self.forward(x)
        y_hat = torch.squeeze(y_hat)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y = y.type(torch.FloatTensor)  # necessary for MSE
        x = x.view(x.size(0), -1)
        y_hat = self.forward(x)
        y_hat = torch.squeeze(y_hat)
        val_loss = F.mse_loss(y_hat, y)
        self.log("val_loss", val_loss)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
