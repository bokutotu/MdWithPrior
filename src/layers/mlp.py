import torch.nn as nn


class MLP(nn.Module):

    def __init__(
        self, num_layers, dropout, input_size, hidden_size, output_size,
        activate_function
    ):
        super().__init__()
        self.net = nn.Sequential()
        for i in range(num_layers-1):
            in_features = hidden_size if i != 0 else input_size
            self.net.add_module("layer_{}".format(i+1),
                                nn.Linear(in_features=in_features,
                                          out_features=hidden_size, ))
            self.net.add_module("activation_{}".format(i+1),
                                getattr(nn, activate_function)())
        self.net.add_module("layer_{}".format(num_layers),
                            nn.Linear(in_features=hidden_size, out_features=output_size))

    def forward(self, x):
        return self.net(x)
