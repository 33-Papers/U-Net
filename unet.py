import torch.nn as nn
import torchvision.transforms.functional as tf
from double_conv import DoubleConv
from encoder import Encoder
from decoder import Decoder


class UNET(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
