import math

import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import multivariate_normal

from src.features.dihedral import DihedralLayer


class CMAPFowardFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, energy, force_x, force_y, grad_x, grad_y, psi, phi):
        grid = int(math.sqrt(energy.size()[0]))
        delta = 360 / grid

        psi = torch.rad2deg(psi) + 179
        phi = torch.rad2deg(phi) + 179

        psi_grid_index = psi // delta
        phi_grid_index = phi // delta
        index = psi_grid_index * grid + phi_grid_index
        index = index.int()
        ctx.save_for_backward(force_x, force_y, grad_x, grad_y, index)
        return torch.index_select(energy, 0, index.reshape(-1)).reshape(index.size())

    @staticmethod
    def backward(ctx, grad_output):
        force_x, force_y, grad_x, grad_y, index = ctx.saved_tensors
        backward_func = CMAPBackwardFunc.apply
        force_x, force_y = backward_func(
                force_x, force_y, grad_x, grad_y, index)
        return torch.zeros_like(force_x), torch.zeros_like(force_x), \
                torch.zeros_like(force_x), torch.zeros_like(force_x), \
                torch.zeros_like(force_x), (grad_output * force_x).requires_grad_(True), \
                (grad_output * force_y).requires_grad_(True)


class CMAPBackwardFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, force_x, force_y, grad_x, grad_y, index):
        size = index.size()
        ctx.save_for_backward(grad_x, grad_y, index, size)
        force_x = torch.index_select(force_x, 0, index.reshape(-1)).reshape(size)
        force_y = torch.index_select(force_y, 0, index.reshape(-1)).reshape(size)
        return force_x.reshape(size), force_y.reshape(size)

    @staticmethod
    def backward(ctx, grad_output):
        grad_x, grad_y, index = ctx.saved_tensors
        size = index.size()
        grad_x = torch.index_select(grad_x, 0, index.reshape(-1)).reshape(size)
        grad_y = torch.index_select(grad_y, 0, index.reshape(-1)).reshape(size)
        return torch.zeros_like(), (grad_output * grad_x).reshape(size), \
                (grad_output * grad_y).reshape(size)


class CMAP(torch.nn.Module):

    def __init__(self, energy, force, grad):
        super().__init__()
        self.register_buffer("energy", energy.reshape(-1))
        self.register_buffer("force_x", force[0].reshape(-1))
        self.register_buffer("force_y", force[1].reshape(-1))
        self.register_buffer("grad_x", grad[0].reshape(-1))
        self.register_buffer("grad_y", grad[1].reshape(-1))
        self.func = CMAPFowardFunc.apply

    def forward(self, psi, phi):
        return self.func(self.energy, self.force_x, 
                self.force_y, self.grad_x, self.grad_y, psi, phi)


def cal_cmap(psi, phi, grid_size):
    cmap = np.zeros((grid_size, grid_size))

    delta = 360 / grid_size

    for i in range(psi.shape[0]):
        psi_index = int(psi[i] / delta)
        phi_index = int(phi[i] / delta)

        cmap[psi_index, phi_index] = cmap[psi_index, phi_index] + 1
    return cmap


def cal_energy(cmap_, epsilon=1e-4):
    cmap_ = cmap_ + epsilon
    energy = -1 * np.log(cmap_)
    energy = energy + np.abs(np.min(energy)) + epsilon
    return energy


def cal_force(energy, grid):
    force_x = np.zeros_like(energy)
    force_y = np.zeros_like(energy)

    delta = 360 / grid

    for i in range(grid):
        for j in range(grid):
            x_max = 0 if i == grid - 1 else i + 1
            x_min = i - 1

            y_max = 0 if j == grid - 1 else j + 1
            y_min = j - 1

#             print(x_max, x_min, y_max, y_min)

            force_x[i, j] = (energy[x_max, j] - energy[x_min, j]) / delta
            force_y[i, j] = (energy[i, y_max] - energy[i, y_min]) / delta

    return force_x, force_y


def cal_force_grad(energy, grid):
    force_grad_x = np.zeros_like(energy)
    force_grad_y = np.zeros_like(energy)

    delta = 360 / grid

    for i in range(grid):
        for j in range(grid):
            x_max = 0 if i == grid - 1 else i + 1
            x_min = i - 1

            y_max = 0 if j == grid - 1 else j + 1
            y_min = j - 1

            # F = -∇E
            # ∇F = ∇(-∇E)
            # ∇F = - ∇**2E
            force_grad_x[i, j] = (energy[x_max, j]+energy[x_min, j]-2*energy[i, j])/(delta ** 2)
            force_grad_y[i, j] = (energy[i, y_max]+energy[i, y_min]-2*energy[i, j])/(delta ** 2)

    return force_grad_x, force_grad_y

def prepare_cmap_force_grad(coord_npy, grid_size, sigma=2, truncate=4):
    _, (psi, phi) = DihedralLayer(coord_npy.shape[1])(torch.tensor(coord_npy))

    psi = torch.rad2deg(psi) + 179
    phi = torch.rad2deg(phi) + 179

    psi = torch.flatten(psi).numpy()
    phi = torch.flatten(phi).numpy()

    cmap = cal_cmap(psi, phi, grid_size)
    cmap = gaussian_filter(cmap, sigma=2, truncate=4)

    energy = cal_energy(cmap)
    force = cal_force(energy, grid_size)
    grad = cal_force_grad(energy, grid_size)

    cmap = torch.tensor(cmap)
    energy = torch.tensor(energy)
    grad_x = torch.tensor(grad[0])
    grad_y = torch.tensor(grad[1])
    force_x = torch.tensor(force[0])
    force_y = torch.tensor(force[1])

    return cmap, energy, (force_x, force_y), (grad_x, grad_y)
