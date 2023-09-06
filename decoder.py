import torch
import torch.nn as nn
import torchvision.transforms.functional as tf
from double_conv import DoubleConv


class Decoder(nn.Module):

    def __init__(self, out_channel=1, channels=(512, 256, 128, 64)):
        super(Decoder, self).__init__()

        self.up_tp = nn.ModuleList()
        self.dc = DoubleConv(512, 1024)

        for channel in channels:
            self.up_tp.append(
                self.up_transpose(channel * 2, channel),
            )
            self.up_tp.append(DoubleConv(channel * 2, channel))

        self.final_conv = nn.Conv2d(channels[-1], out_channel, kernel_size=1)

    @staticmethod
    def up_transpose(in_channel, out_channel):
        return nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=2, stride=2)

    def forward(self, x):
        out, residual_connections = x

        residual_connections = residual_connections[::-1]

        out = self.dc(out)

        for i in range(0, len(self.up), 2):
            out = self.up[i](out)
            residual_connection = residual_connections[i // 2]

            if out.shape != residual_connection:
                out = tf.resize(out, size=residual_connection.shape[2:])

            concat_residue = torch.cat((residual_connection, out), dim=1)

            out = self.up_tp[i + 1](concat_residue)

        return self.final_conv(out)
