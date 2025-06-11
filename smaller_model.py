import torch
import torch.nn as nn
from opts import parser
from model import Model
from utils import Flatten
import torch.nn.functional as F


import pytorch_lightning as pl

args = parser.parse_args()

class SmallModel(Model):

    def __init__(self, in_channels=16, channels=32, kernel_size=3, stride=2, biases=True, num_classes=10, lr=args.lr):
        Model.__init__(self, in_channels, channels, kernel_size, lr)

        in_channels = 8
        channels = 9

        if args.kuramoto_mode == 'endtoend':
            if args.dataset == 'multi_mnist' or args.in_repo == 'multi_mnist_cifar_greyscale':
                if args.gabors:
                    self.downsampling = nn.Conv2d(1, in_channels, kernel_size, 1, padding=5 // 2, bias=False)
                    self.downsampling.weight = nn.Parameter(torch.load('../pt_utils/gabors.pt', map_location='cuda'),
                                                            requires_grad=True)  # .repeat(1,3,1,1)

                else:
                    self.downsampling = nn.Conv2d(1, in_channels, kernel_size, 1, padding=3 // 2, bias=False)

            else:
                self.downsampling = nn.Conv2d(3, in_channels, kernel_size, 1, padding=5 // 2, bias=False)
                self.downsampling.weight = nn.Parameter(
                    torch.load('../pt_utils/gabors.pt', map_location='cuda').repeat(1, 3, 1, 1), requires_grad=True)

        elif not args.kuramoto_mode == 'input':
            self.downsampling.weight.requires_grad = False

        self.convA = nn.Conv2d(in_channels, channels, kernel_size=kernel_size, stride=stride, padding=self.padding, bias=biases)
        self.normA = nn.InstanceNorm2d(channels, affine=True)

        self.classif = nn.Sequential(Flatten(), nn.Linear(channels * 16 * 16, 46, bias=biases), nn.LayerNorm(46),
                                 nn.ReLU(), nn.Linear(46, num_classes, bias=biases))

    def forward(self, input):
        input = input.to(torch.float)
        out = torch.relu(self.downsampling(input))

        outA = torch.relu(self.normA(self.convA(out)))
        pred = self.classif(outA)

        return pred






