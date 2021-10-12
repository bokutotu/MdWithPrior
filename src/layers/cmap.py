import math

import torch


class _cmap(torch.autograd.Function):

    @staticmethod
    def forward(ctx, idx_x, idx_y, grad, grad_grad):
        grid = cmap.size()[0]
        flatten_idx = idx_x * grid + idx_y
        energy = torch.index_select(grad, 0, flatten_idx)
        ctx.save_for_backward(flatten_idx, grad_grad)
        return energy

    @staticmethod
    def backward(ctx, grad_output):
        flatten_idx, grad_grad = ctx.saved_tensors
        grad_input = grad_output.clone()
        return grad_input * torch.index_select(grad_input, 0, flatten_idx)


def prepare_grad(cmap):
    grids = cmap.size()[0]

    grad = torch.zeros(grids, grids, 2)
    grad_grad = torch.zeros(grids, grids, 2)

    delta = 2 * math.pi / grids

    for i, j in zip(range(grids), range(grids)):
        before_i = grids - 1 if i == 0 else i - 1
        after_i = 0 if i == grids - 1 else i + 1

        before_j = grids - 1 if j == 0 else j - 1
        after_j = 0 if j == grids - 1 else j + 1

        grad[i, j, 0] = cmap[after_i, j] - cmap[before_i, j]
        grad[i, j, 1] = cmap[i, after_j] - cmap[i, before_j]

        grad_grad[i, j, 0] = cmap[after_i, j] + \
            cmap[before_i, j] - 2 * cmap[i, j]

        grad_grad[i, j, 0] = cmap[i, after_j] + \
            cmap[i, before_j] - 2 * cmap[i, j]

    grad = grad / (2 * delta)
    grad_grad = grad_grad / (delta ** 2)
    return grad, grad_grad


class CMAP(torch.nn.Module):

    def __init__(self, cmap):
        grad, grad_grad = prepare_grad(cmap)
        self.register_buffer("cmap", cmap.view(-1))
        self.register_buffer("grad", grad.view(-1))
        self.register_buffer("grad_grad", grad_grad.view(-1))

    def forward(self, psi, phi):
        delta = 2 * math.pi / cmap.size()[0]

        psi = int(psi / delta)
        phi = int(phi / delta)

        return _cmap.apply(psi, phi, self.grad, self.grad_grad)
