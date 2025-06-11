import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
from opts import parser
import complex_functions as complex_functions
import utils
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchmetrics.functional import accuracy
import pytorch_lightning as pl
from skimage.util import random_noise
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score

args = parser.parse_args()


class Model(pl.LightningModule):

    def __init__(self, in_channels=16, channels=32, kernel_size=3, lr=args.lr):
        pl.LightningModule.__init__(self)
        self.channels = channels
        self.padding = kernel_size // 2
        self.acc = BinaryAccuracy()
        self.f1 = BinaryF1Score()

        if args.dataset == 'multi_mnist':
            if args.gabors:
                self.downsampling = nn.Conv2d(1, in_channels, kernel_size, stride=1, padding=5 // 2, bias=False)
                self.downsampling.weight.data = torch.load('../pt_utils/gabors.pt',
                                                           map_location='cuda')  # .repeat(1,3,1,1)
            else:
                self.downsampling = nn.Conv2d(1, in_channels, kernel_size, stride=1, padding=3 // 2, bias=False)
                self.downsampling.weight.data = torch.load('../pt_utils/weights_conv1.pt',
                                                           map_location='cuda')  # .repeat(1,3,1,1)
        else:
            self.downsampling = nn.Conv2d(3, in_channels, kernel_size, stride=1, padding=5 // 2, bias=False)
            self.downsampling.weight.data = torch.load('../pt_utils/gabors.pt', map_location='cuda').repeat(1, 3, 1, 1)

        self.learning_rate = lr

        self.out_file = None

    def bce_loss(self, logits, labels, mode):
        if mode == 'bce':
            return F.binary_cross_entropy_with_logits(logits, labels)
        else:
            return F.nll_loss(logits, labels)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def evaluate(self, batch, stage=None):

        x, y, m = batch

        if x.shape[-1] != 32:
            x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=True)

        if args.add_noise:
            if args.dataset == 'object_0' or args.dataset == 'object_025' or args.dataset == 'object_05':
                x = torch.tensor(random_noise(x.cpu(), mode='gaussian', mean=0, var=0.1, clip=True), dtype=torch.float)
                # x = torch.tensor(random_noise(x.cpu(), mode = 's&p', amount = 0.1, clip=True), dtype=torch.float)
            else:
                x = torch.tensor(random_noise(x.cpu(), mode='gaussian', mean=0, var=0.05, clip=True), dtype=torch.float)
                # x = torch.tensor(random_noise(x.cpu(), mode = 's&p', amount = 0.1, clip=True), dtype=torch.float)

        logits = self.forward(x.to(self.device))

        loss = self.bce_loss(logits, y, 'bce')
        acc = utils.accuracy_multi(y, torch.sigmoid(logits), display=False, n_obj=args.num_objects)
        binary_acc = self.acc(torch.sigmoid(logits), y)
        f1 = self.f1(torch.sigmoid(logits), y)
        if stage:
            self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True)
            self.log(f"{stage}_acc", acc, on_step=True, on_epoch=True)
            self.log(f"{stage}_binary_acc", binary_acc, on_step=True, on_epoch=True)
            self.log(f"{stage}_f1", f1, on_step=True, on_epoch=True)

        return acc

    def training_step(self, train_batch, batch_idx):
        setattr(self, 'test_mode', False)

        x, y, m = train_batch

        if x.shape[-1] != 32:
            x = F.interpolate(x, size=(32, 32), mode='bilinear', align_corners=True)

        logits = self.forward(x.to(self.device))

        loss = self.bce_loss(logits, y, 'bce')
        acc = utils.accuracy_multi(y, torch.sigmoid(logits), display=False, n_obj=args.num_objects)
        b_acc = self.acc(torch.sigmoid(logits), y)
        f1 = self.f1(torch.sigmoid(logits), y)

        self.log(f"train/loss", loss, on_step=True, on_epoch=True)
        self.log(f"train/acc", acc, on_step=True, on_epoch=True)
        self.log(f"train/binary_acc", b_acc, on_step=True, on_epoch=True)
        self.log(f"train/f1", f1, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.evaluate(batch, "Val")

    def test_step(self, batch, batch_idx):
        return self.evaluate(batch, "Test")
