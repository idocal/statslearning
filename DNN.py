import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl


class DNN(pl.LightningModule):

    def __init__(self, n_feats, layers=None):
        super().__init__()
        if layers is None:
            layers = [n_feats * 2]
        modules = []
        if not len(layers):  # perceptron
            modules = [nn.Linear(n_feats, 1)]
        else:
            modules += [nn.Linear(n_feats, layers[0]), nn.ReLU()]
            for i in range(1, len(layers)):
                modules += [nn.Linear(layers[i-1], layers[i]), nn.ReLU()]
            modules += [nn.Linear(layers[-1], 1)]
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y = y.type(torch.FloatTensor)  # necessary for MSE
        x = x.view(x.size(0), -1)
        y_hat = self.net(x)
        y_hat = torch.squeeze(y_hat)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y = y.type(torch.FloatTensor)  # necessary for MSE
        x = x.view(x.size(0), -1)
        y_hat = self.net(x)
        y_hat = torch.squeeze(y_hat)
        val_loss = F.mse_loss(y_hat, y)
        self.log("val_loss", val_loss)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
