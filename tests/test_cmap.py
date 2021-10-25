import math

import numpy as np
import torch

from src.layers.cmap import CMAP, prepare_cmap_force_grad
from src.features.dihedral import DihedralLayer


def test_cmap_select_energy():
    coord_path = "./tests/data/c_test.npy"
    coord_npy = np.load(coord_path)

    cmap, energy, force, grad = prepare_cmap_force_grad(coord_npy, 50)
    cmap_layer = CMAP(energy, force, grad)

    psi_deg = -50.4
    phi_deg = -64.8

    psi = math.radians(psi_deg)
    phi = math.radians(phi_deg)

    psi = torch.tensor(psi).reshape(1,) # batch, number of time step,  psi_num
    phi = torch.tensor(phi).reshape(1,)

    energy = cmap_layer(psi, phi)


def test_cmap_get_forces():
    coord_path = "./tests/data/c_test.npy"
    coord_npy = np.load(coord_path)

    cmap, energy, force, grad = prepare_cmap_force_grad(coord_npy, 50)
    cmap_layer = CMAP(energy, force, grad)

    psi_deg = -179.
    phi_deg = -179.

    psi = math.radians(psi_deg)
    phi = math.radians(phi_deg)

    psi = torch.tensor(psi, requires_grad=True).reshape(1,) # batch, number of time step,  psi_num
    phi = torch.tensor(phi, requires_grad=True).reshape(1,)

    energy = cmap_layer(psi, phi)
    force_psi = torch.autograd.grad(-torch.sum(energy), 
            (psi, phi), 
            create_graph=True, 
            retain_graph=True)


def test_cmap_grad_grad():

    coord_path = "./tests/data/c_test.npy"
    coord_npy = np.load(coord_path)

    cmap, energy, force, grad = prepare_cmap_force_grad(coord_npy, 50)
    cmap_layer = CMAP(energy, force, grad)

    psi_deg = -179.
    phi_deg = -179.

    psi = math.radians(psi_deg)
    phi = math.radians(phi_deg)

    psi = torch.tensor(psi, requires_grad=True).reshape(1,) # batch, number of time step,  psi_num
    phi = torch.tensor(phi, requires_grad=True).reshape(1,)

    energy = cmap_layer(psi, phi)
    force = torch.autograd.grad(-torch.sum(energy), 
            (psi, phi), 
            create_graph=True, 
            retain_graph=True)

    grad_grad = torch.autograd.grad(force[0] + force[1], (psi, phi), 
            create_graph=True, retain_graph=True, allow_unused=True)


def test_cmap_grad_grad_2():
    coord_path = "./tests/data/c_test.npy"
    coord_npy = np.load(coord_path)

    cmap, energy, force, grad = prepare_cmap_force_grad(coord_npy, 50)
    cmap_layer = CMAP(energy, force, grad)

    dihedral = DihedralLayer(coord_npy.shape[1])

    coords = torch.tensor(coord_npy, requires_grad=True)

    dihedrals,rad = dihedral(coords)
    energy = cmap_layer(rad[0], rad[1])

    force = torch.autograd.grad(-torch.sum(energy), 
            coords, 
            create_graph=True, 
            retain_graph=True)

    grad_grad = torch.autograd.grad(torch.sum(force[0]), coords, 
            create_graph=True, retain_graph=True, allow_unused=True)


def test_cmap_force():
    energy = np.load("tests/data/energy.npy")
    force_psi = np.load("tests/data/force_psi.npy")
    force_phi = np.load("tests/data/force_phi.npy")
    grad_psi = np.load("tests/data/grad_psi.npy")
    grad_phi = np.load("tests/data/grad_phi.npy")

    energy = torch.tensor(energy)
    force_psi = torch.tensor(force_psi)
    force_phi = torch.tensor(force_phi)
    grad_psi = torch.tensor(grad_psi)
    grad_phi = torch.tensor(grad_phi)

    cmap = CMAP(energy, (force_psi, force_phi), (grad_psi, grad_phi))

    psi_deg = 37.
    phi_deg = -35.

    psi = math.radians(psi_deg)
    phi = math.radians(phi_deg)

    psi = torch.tensor(psi, requires_grad=True).reshape(1,) # batch, number of time step,  psi_num
    phi = torch.tensor(phi, requires_grad=True).reshape(1,)

    psi = math.radians(psi_deg)
    phi = math.radians(phi_deg)

    psi = torch.tensor(psi, requires_grad=True).reshape(1,) # batch, number of time step,  psi_num
    phi = torch.tensor(phi, requires_grad=True).reshape(1,)

    energy = cmap(psi, phi)
    force = torch.autograd.grad(-torch.sum(energy), 
            (psi, phi), 
            create_graph=True, 
            retain_graph=True)
    np.testing.assert_allclose(force[0].detach().numpy(), np.array(-0.2516679777619106, dtype=np.float32), atol=1e-7)
    np.testing.assert_allclose(force[1].detach().numpy(), np.array(-0.4071115362695653, dtype=np.float32), atol=1e-7)


class _cmap(torch.nn.Module):

    def __init__(self, layer, size):
        super().__init__()
        self.layer = layer
        self.k = torch.nn.parameter.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, psi, phi):
        energy = self.layer(psi, phi)
        return self.k * energy


def test_cmap_force_loss_is_becoming_small():
    energy = np.load("tests/data/energy.npy")
    force_psi = np.load("tests/data/force_psi.npy")
    force_phi = np.load("tests/data/force_phi.npy")
    grad_psi = np.load("tests/data/grad_psi.npy")
    grad_phi = np.load("tests/data/grad_phi.npy")

    energy = torch.tensor(energy)
    force_psi = torch.tensor(force_psi)
    force_phi = torch.tensor(force_phi)
    grad_psi = torch.tensor(grad_psi)
    grad_phi = torch.tensor(grad_phi)

    cmap = CMAP(energy, (force_psi, force_phi), (grad_psi, grad_phi))
    cmap_ = _cmap(CMAP(energy, (force_psi, force_phi), (grad_psi, grad_phi)), energy.size()[0])
    real = 3

    optim = torch.optim.Adam(cmap_.parameters(), lr=1e-4)
    loss = torch.nn.MSELoss()
    dihedral = DihedralLayer(force_phi.size()[0])

    input = torch.randn((10, force_phi.size()[0], 3), requires_grad=True)
    _, rads = dihedral(input)
    out = cmap(*rads)
    force = torch.autograd.grad(-torch.sum(out), 
            input, 
            create_graph=True, 
            retain_graph=True)
    init_loss = loss(out * real, cmap_(*rads))

    for i in range(1000):
        input = torch.randn((10, force_phi.size()[0], 3), requires_grad=True)
        _, rads = dihedral(input)
        out = cmap(*rads)
        force = torch.autograd.grad(-torch.sum(out), 
                input, 
                create_graph=True, 
                retain_graph=True)
        _loss = loss(out * real, cmap_(*rads))
        _loss.backward()
        optim.step()

    assert _loss < init_loss
        
