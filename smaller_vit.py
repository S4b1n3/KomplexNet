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
        embed_dim = 64
        patch_size=4
        num_heads = 2
        mlp_dim = 128
        num_layers = 3
        image_size = 32
        assert image_size % patch_size == 0
        num_patches = (image_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)


        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = x.to(torch.float)

        x = self.patch_embed(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        x = x + self.pos_embed

        x = self.transformer(x)  # (B, num_patches, embed_dim)

        # Pooling: mean over tokens
        x = x.mean(dim=1)
        return self.head(x)






