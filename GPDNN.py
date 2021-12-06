import networkx as nx
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from RandomCalmanTree import RandomCalmanTree
from typing import List


class GPDNN(pl.LightningModule):

    def __init__(self, rct: RandomCalmanTree):
        super().__init__()

        # construct linear ReLU architecture
        modules = []
        for i in range(len(rct.layers) - 1):
            in_features = rct.layers[i]
            out_features = rct.layers[i+1]
            modules += [nn.Linear(in_features, out_features)]
            if i < len(rct.layers) - 2:
                modules += [nn.ReLU()]
        self.net = nn.Sequential(*modules)

        # mask by connectivity
        self.masks = self.masks(rct)
        assert len(self.masks) == (len(self.net) // 2 + 1)
        for j, mask in enumerate(self.masks):
            matrix_idx = j * 2  # we do not mask the bias weights
            masked = self.net[matrix_idx].weight * mask
            with torch.no_grad():
                self.net[matrix_idx].weight = nn.Parameter(masked)

    @staticmethod
    def masks(rct: RandomCalmanTree) -> List[torch.tensor]:
        masks = []
        curr_node = 0
        for i in range(len(rct.layers) - 1):
            # define layer mask shape
            in_features = rct.layers[i]
            out_features = rct.layers[i+1]
            mask = torch.zeros(out_features, in_features)

            # detect connectivity from node to successors
            for node in range(out_features):
                node_idx = node + curr_node + in_features
                children = [x for x in rct.tree.successors(node_idx)]
                for child in children:
                    child_idx = child - curr_node
                    mask[node, child_idx] = 1

            # proceed to next layer
            curr_node += in_features
            masks.append(mask)

        return masks

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
