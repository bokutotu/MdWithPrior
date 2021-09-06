import torch
import torch.nn as nn


class MaskedCNN(nn.Conv1d):
    """
    Implementation of Masked CNN Class as explained in A Oord et. al.
    Taken from https://github.com/jzbontar/pixelcnn-pytorch
    """

    def __init__(self, *args, **kwargs):
        super(MaskedCNN, self).__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())

        _, _, kernel_size = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kernel_size//2+1] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedCNN, self).forward(x)


class CnnBlock(nn.Module):

    def __init__(self, layer_num, hidden_dim, activate_func):
        super().__init__()
        self.net = nn.Sequential()
        for i in range(layer_num):
            dilation = 2 ** (i)
            cnn = MaskedCNN(
                in_channels=hidden_dim, out_channels=hidden_dim, bias=False,
                kernel_size=3, dilation=dilation, stride=1, padding=dilation)
            self.net.add_module("cnn-{}".format(i), cnn)
            func = getattr(nn, activate_func)()
            self.net.add_module(activate_func+"{}".format(i), func)

    def forward(self, x):
        return self.net(x) + x


class WaveNet(nn.Module):

    def __init__(self, block_layers, block_num, input_size, hidden_dim, out_dim, activate_func):
        super().__init__()
        self.input_cnn = nn.Conv1d(
            in_channels=input_size, out_channels=hidden_dim, kernel_size=1)
        self.output_cnn = nn.Conv1d(
            in_channels=hidden_dim, out_channels=out_dim, kernel_size=1)

        self.res = nn.Sequential()
        for i in range(block_num):
            block = CnnBlock(block_layers, hidden_dim, activate_func)
            self.res.add_module("cnnblock-{}".format(i), block)

    def forward(self, x):
        x = torch.transpose(x, 1, 2)
        x = self.input_cnn(x)
        x = self.res(x)
        x = self.output_cnn(x)
        x = torch.transpose(x, 1, 2)
        return x
