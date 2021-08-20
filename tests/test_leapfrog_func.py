import numpy as np
from numpy.testing import assert_allclose
import torch

from src.simulater.leapfrog import (cal_coord,
                                    cal_v,
                                    get_weight_tensor,
                                    cal_next_coord_using_pred_forces)

MASS = {'CA': 12.01100, 'CB': 12.01100, 'C': 12.01100, 'O': 15.99900, 'N': 14.00700}


def test_cal_coord():
    coordinates = np.load("tests/data/c_test.npy")
    velocity = np.load("tests/data//v_test.npy")

    coordinates_tensor = torch.tensor(coordinates)
    velocity = torch.tensor(velocity)

    for i in range(coordinates.shape[0] - 1):
        res = cal_coord(coordinates_tensor[i], velocity[i+1])
        res = res.numpy()
        assert_allclose(res, coordinates[i+1], )


def test_cal_velocity():
    velocities = np.load("tests/data/v_test.npy")
    forces = np.load("tests/data/f_test.npy")
    weight = get_weight_tensor(velocities.shape[1])
    weight = weight.numpy()

    # velocities_tensor = torch.tensor(velocities)
    # forces = torch.tensor(forces)

    for i in range(velocities.shape[0] - 1):
        res = cal_v(velocities[i], weight, forces[i])
        assert_allclose(res, velocities[i+1], atol=1e-3)


def test_get_weight():
    atom_num = 79
    mass = []
    for idx in range(atom_num):
        if idx % 4 == 0:
            mass.append(MASS["N"])
        elif idx % 4 == 1 or idx % 4 == 2:
            mass.append(MASS["C"])
        else:
            mass.append(MASS["O"])
    mass = np.array(mass).reshape(-1,1)
    mass = np.concatenate([mass, mass, mass], axis=-1)

    mass_tensor = get_weight_tensor(atom_num)
    assert_allclose(mass_tensor.numpy(), mass)


def test_simulation():
    coordinates = np.load("tests/data/c_test.npy")
    velocities = np.load("tests/data//v_test.npy")
    forces = np.load("tests/data/f_test.npy")

    coordinates = torch.tensor(coordinates)
    velocities = torch.tensor(velocities)
    forces = torch.tensor(forces)

    weight = get_weight_tensor(velocities.size(1))

    for idx in range(velocities.size()[0] - 1):
        cal_next_coord = cal_next_coord_using_pred_forces(
                coordinates[idx], velocities[idx], forces[idx], weight)
        assert_allclose(coordinates[idx+1].numpy(), cal_next_coord.numpy(),rtol=1e-6)
