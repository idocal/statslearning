import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl


class DNN(pl.LightningModule):

    def __init__(self, n_feats, layers=[50, 20]):
        super().__init__()
        modules = [nn.Linear(n_feats, layers[0]), nn.ReLU()]
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
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
