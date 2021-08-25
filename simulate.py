import argparse

import numpy as np
import torch

from src.simulater.leapfrog import cal_next_coord_using_pred_forces, get_weight_tensor
from src.load import load_from_run_id


def simulate_step_mlp(coordinates, velocities, model, weight, step, dt):
    pred_forces, _ = model(coordinates[step].unsqueeze(dim=0))
    pred_forces = pred_forces[0]
    cal_coordinates, cal_velocities = cal_next_coord_using_pred_forces(
        coordinates[step], velocities[step], pred_forces, weight, dt
    )
    coordinates[step+1] = cal_coordinates
    velocities[step+1] = cal_velocities
    return coordinates, velocities, model


def simulate_step_lstm(coordinates, velocities, model, weight, step, feature_len, dt):
    input_features = coordinates[step:step+feature_len]
    input_features = input_features.unsqueeze(dim=0)
    cal_coordinates, cal_velocities = model(input_features)
    coordinates[step+feature_len] = cal_coordinates[-1]
    velocities[step+feature_len] = cal_velocities[-1]
    return coordinates, velocities, model


def simulate(
        init_coordinates, init_velocities, model, step, mode, save_name,
        feature_len=None, dt=0.002):

    if mode not in ["MLP", "LSTM"]:
        ValueError("mode {} is not impl".format(mode))

    init_coordinates = torch.tensor(init_coordinates, requires_grad=True)
    init_velocities = torch.tensor(init_velocities, requires_grad=True)

    atom_num = init_coordinates.size()[1]

    result_coordinates = torch.zeros((step + feature_len, atom_num, 3))
    result_velocities = torch.zeros((step + feature_len, atom_num, 3))

    result_coordinates[0:feature_len] = init_coordinates[0:feature_len]
    result_velocities[0:feature_len] = init_velocities[0:feature_len]

    weight = get_weight_tensor(atom_num)

    print("  start simulation  ")
    for step in range(step):
        if mode == "MLP":
            result_coordinates, result_velocities, model = \
                simulate_step_mlp(
                    result_coordinates, result_velocities, model,
                    weight, step, args.dt)
        elif mode == "LSTM":
            result_coordinates, result_velocities, model = \
                simulate_step_lstm(
                    result_coordinates, result_velocities,
                    model, weight, step, feature_len)

    if mode == "MLP":
        result_coordinates = result_coordinates[1:-1:]
    elif mode == "LSTM":
        result_coordinates = result_coordinates[feature_len:-1:]

    np.save(save_name, result_coordinates.detach().cpu().numpy())


def main(args):
    coordinates = np.load(args.coordinates_path)
    velocities = np.load(args.velocities_path)

    model = load_from_run_id(args.run_id)

    mode = model.config.models._target_.split(".")[-1]
    feature_len = model.config.dataset.features_lengthS if mode == "LSTM" \
        else 1

    simulate(
        coordinates, velocities, model,
        args.num_steps, mode, args.save_name, feature_len)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LeapFrog or ベルれ Simulation")
    parser.add_argument("--num_steps", type=int,
                        help="How many steps to run")
    parser.add_argument("--run_id", type=str,
                        help="mlflow run id for use model config and so on")
    parser.add_argument("--save_name", type=str,
                        help="simulation result trj npy file name")
    parser.add_argument("--dt", default=0.002,
                        help="buration between simulation steps")
    parser.add_argument("--coordinates_path", type=str,
                        help="path to init protein state npy file for simulation(coordinates)")
    parser.add_argument("--velocities_path", type=str,
                        help="path to init protein state npy file for simulation(velocities)")
    args = parser.parse_args()
    main(args)
