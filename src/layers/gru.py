import torch
import torch.nn as nn


class GRU(nn.Module):

    def __init__(self, input_size, num_layers, dropout, hidden_size, output_size):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, dropout=dropout)
        self.k = torch.nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, x):
        x = torch.transpose(x, 0, 1)
        x, _ = self.gru(x)
        x = torch.sum(x, dim=-1)
        x = torch.transpose(x, 0, 1)
        return torch.abs(x.unsqueeze(dim=-1) * self.k)
