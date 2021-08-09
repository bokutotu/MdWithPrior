import torch.nn as nn


class LSTM(nn.Module):

    def __init__(self, ):
        super().__init__(num_layers, dropout, input_size, hidden_size, output_size, )

        self.net = nn.LSTM(num_layers=num_layers-1, batch_first=True,
                           dropout=dropout, input_size=input_size, hidden_size=hidden_size)
        self.last = nn.LSTM(num_layers=1, batch_first=True, dropout=dropout,
                            input_size=hidden_size, output_size=output_size)

    def forward(self, x):
        x, _ = self.net(x)
        return self.last(x)[0]
