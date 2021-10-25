import torch
import torch.nn as nn


class GRU(nn.Module):

    def __init__(self, input_size, num_layers, dropout, hidden_size, output_size):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers-1, dropout=dropout)
        self.last_gru = nn.GRU(
            input_size=hidden_size, hidden_size=output_size,
            num_layers=1, dropout=dropout)

    def forward(self, x):
        x = torch.transpose(x, 0, 1)
        x, _ = self.gru(x)
        x, _ = self.last_gru(x)
        x = torch.transpose(x, 0, 1)
        return x
