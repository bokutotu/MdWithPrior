import torch

MASS = {'CA': 12.011, 'CB': 12.011, 'C': 12.011, 'O': 15.999, 'N': 14.007}


def cal_coord(coordinates, velocity, dt=0.002): 
    """Calculate coordinates using Leap Frog methods

    Parameters
    ----------
    coordinates: torch.tensor
        1 step before coordinates this tensor dimention is 
        (number of atoms or beads, 3 (x,y,z))
    velocity: torch.tensor
        half step before velocities.this tensor's dimention is 
        (number of atoms or beads, 3(x,y,z))
    dt: float
        step time
    """
    if coordinates.size() != velocity.size():
        raise ValueError("coordinates shape and velocities shape must be same")
    return coordinates + dt * velocity


def cal_v(v_2, m, f, dt=0.002):
    """time step at n + 1/2の速度を計算する
    Parameters
    ==========
    v_2 : numpy.array or torch.tensor
        time step at n - 1/2の速度
    step : int
        step サイズ
    m : float
        計算する原子の重さ
    f : numpy.array or torch.tesor
        原子にかかる力
    """
    # assert v_2.size() == m.size() == f.size()
    return v_2 + f * dt / m


def get_weight_tensor(atom_num):
    weight = torch.zeros((atom_num, 3))
    weight[0::4] = MASS["N"]
    weight[1::4] = MASS["CA"]
    weight[2::4] = MASS["C"]
    weight[3::4] = MASS["O"]
    return weight


def cal_next_coord_using_pred_forces(
        coordinates, velocities, pred_forces, weight, dt=0.002):
    """Use the force obtained from the coordinates to calculate the coordinates 
    of the next step from the velocity and the current coordinates.

    Parameters
    ----------
    coordinates: torch.tensor
        coordinates pytorch tensor dimention is (number of atoms or beads, 3)
    velocity: torch.tensor
        velocity pytorch tensor dimention is (number of atoms or beads, 3)
    pred_forces: torch.tensor
        predicted force pytroch tensor. dimention is (number of atoms or beads, 3)
    weight: torch.tensor
        atom or beads weight tensor using calculating velocity from forces
    dt: float
        duration between steps
    """
    next_velocity = cal_v(velocities, weight, pred_forces, dt=dt)
    return cal_coord(coordinates, next_velocity, dt)
