import torch.nn as nn
from double_conv import DoubleConv


class Encoder(nn.Module):

    def __init__(self, in_channels=3, channels=(64, 128, 256, 512)):
        super(Encoder, self).__init__()
        self.down = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for channel in channels:
            self.down.append(DoubleConv(in_channels, channel))
            in_channels = channel

    def forward(self, x):
        residual_connections = []

        for down in self.down:
            x = down(x)
            residual_connections.append(x)
            x = self.pool(x)

        return x, residual_connections
