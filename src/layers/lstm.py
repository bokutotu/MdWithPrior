import torch
import torch.nn as nn


class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ih = nn.Linear(input_size, 4 * hidden_size)
        self.hh = nn.Linear(hidden_size, 4 * hidden_size)
        # self.hh = nn.Parameter(
        #     torch.randn(4 * hidden_size, hidden_size))
        # self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        # self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))

    def forward(self, input, state):
        hx, cx = state
        gates = self.ih(input) + self.hh(hx)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, cy


class LSTM(nn.Module):

    def __init__(self, num_layers, dropout, input_size, hidden_size, output_size):
        super().__init__()
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        layer_list = []
        for layer_idx in range(self.num_layers):
            input_size = hidden_size if layer_idx != 0 else self.input_size
            hidden_size = hidden_size if layer_idx != self.num_layers - 1 else output_size
            tmp_layer = LSTMCell(input_size=input_size,
                                 hidden_size=hidden_size)
            layer_list.append(tmp_layer)

        self.dropout = None
        if dropout != 0.0:
            self.dropout = nn.Dropout(dropout)

        self.layer = nn.Sequential(*layer_list)

    def forward(self, x, state=None):
        """
        x: torch.Tensor
            (batch, time, dim)
        """
        batch_size = x.size()[0]
        time_len = x.size()[1]

        device = "cpu" if not x.is_cuda else "cuda:" + str(x.get_device())

        state = [[] for _ in range(self.num_layers)]
        hc = [[] for _ in range(self.num_layers)]
        init_state = []
        for i in range(self.num_layers):
            dim = self.hidden_size if i != self.num_layers - 1 else self.output_size
            init_state.append(
                torch.zeros((batch_size, dim), device=device,
                            requires_grad=True))

        for time_idx in range(time_len):
            for layer_idx in range(self.num_layers):
                if time_idx == 0:
                    if layer_idx == 0:
                        output, hidden = self.layer[layer_idx](
                            x[::, time_idx], (init_state[layer_idx], init_state[layer_idx]))
                    else:
                        output, hidden = self.layer[layer_idx](
                            output, (init_state[layer_idx], init_state[layer_idx]))
                else:
                    if layer_idx == 0:
                        output, hidden = self.layer[layer_idx](
                            x[::, time_idx], (state[layer_idx][-1], hc[layer_idx][-1]))
                    else:
                        output, hidden = self.layer[layer_idx](
                            output, (state[layer_idx][-1], hc[layer_idx][-1]))
                state[layer_idx] += torch.unsqueeze(output, 0)
                hc[layer_idx] += hidden

        return torch.cat(state[-1], ).transpose(0, 1)
        # return torch.cat(state[-1], ).view(batch_size, time_len, 1)
