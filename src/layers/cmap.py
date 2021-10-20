import math

import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import multivariate_normal

from src.features.dihedral import DihedralLayer


class CMAPFowardFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, energy, force, grad, psi, phi):
        num_grid = int(energy.size() ** 0.5)

        psi = torch.rad2deg(psi) + 179
        phi = torch.rad2deg(phi) + 179

        psi_grid_index = psi // grid
        phi_grid_index = phi // grid
        index = psi_grid_index * num_grid + phi_grid_index
        ctx.save_for_backward(force, grad, index)
        return torch.index_select(energy, 0, index)

    @staticmethod
    def backward(ctx, grad_output):
        force, grad, index = ctx.saved_tensors
        backward_func = CMAPBackwardFunc.apply
        return grad_output * backward_func(force, grad, index)


class CMAPBackwardFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, force, grad, index):
        ctx.save_for_backward(grad, index)
        return torch.index_select(force, 0, index)

    @staticmethod
    def backward(ctx, grad_output):
        grad, index = ctx.saved_tensors
        return grad_output * torch.index_select(grad, 0, index)


class CMAP(torch.nn.Module):

    def __init__(self, energy, force, grad, atom_num):
        super().__init__()
        self.register_buffer("energy", energy)
        self.register_buffer("force", force)
        self.register_buffer("grad", grad)
        self.func = CMAPFowardFunc.apply

    def forward(self, psi, phi):
        return self.func(self.energy, self.force, self.grad, psi, phi)


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

            force_x[i, j] = -1 * (energy[x_max, j] - energy[x_min, j]) / delta
            force_y[i, j] = -1 * (energy[i, y_max] - energy[i, y_min]) / delta

    return force_x, force_y


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

    return cmap, energy, force
