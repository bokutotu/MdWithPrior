import torch
import torch.nn as nn


class CnnBlock(nn.Module):

    def __init__(self, layer_num, hidden_dim):
        super().__init__()
        self.net = nn.Sequential()
        for i in range(layer_num):
            dilation = 2 ** (i)
            cnn = nn.Conv1d(
                in_channels=hidden_dim, out_channels=hidden_dim,
                kernel_size=3, dilation=dilation, stride=1, padding=dilation)
            self.net.add_module("cnn-{}".format(i), cnn)

    def forward(self, x):
        return self.net(x) + x


class WaveNet(nn.Module):

    def __init__(self, block_layers, block_num, input_size, hidden_dim, out_dim):
        super().__init__()
        self.input_cnn = nn.Conv1d(
            in_channels=input_size, out_channels=hidden_dim, kernel_size=1)
        self.output_cnn = nn.Conv1d(
            in_channels=hidden_dim, out_channels=out_dim, kernel_size=1)

        self.res = nn.Sequential()
        for i in range(block_num):
            block = CnnBlock(block_layers, hidden_dim)
            self.res.add_module("cnnblock-{}".format(i), block)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.input_cnn(x)
        x = self.res(x)
        x = self.output_cnn(x)
        x = torch.transpose(x, 1, 2)
        return x
