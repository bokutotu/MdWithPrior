import argparse
import random

from cgnet.molecule import CGMolecule
from cgnet.network import (HarmonicLayer, CGnet, ForceLoss,
                           lipschitz_projection, Simulation)
from cgnet.feature import (MoleculeDataset, GeometryStatistics,
                           GeometryFeature, ShiftedSoftplus,
                           CGBeadEmbedding, SchnetFeature,
                           FeatureCombiner, LinearLayer,
                           GaussianRBF)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from torch.optim.lr_scheduler import MultiStepLR

import numpy as np


def get_embeddings(atom_num, sim_len):
    embeddings = [i % 4 + 1 for i in range(atom_num)]
    return np.tile(embeddings, [sim_len, 1])


def get_stat(coords, forces, num_samples):
    idx = random.sample(range(coords.shape[0]), num_samples)
    coords, forces = coords[idx], forces[idx]
    return GeometryStatistics(coords, backbone_inds='all', get_all_distances=True,
                              get_backbone_angles=True, get_backbone_dihedrals=True)


def main(args):
    coords = np.load(args.coords_npy)
    forces = np.load(args.forces_npy)

    atom_num = coords.shape[1]
    sim_len = coords.shape[0]

    device = torch.device("cuda")

    embeddings = get_embeddings(atom_num, sim_len)
    data = MoleculeDataset(coords, forces, embeddings,
                           device=device)

    stats = get_stat(coords, forces, num_samples=10000)

    bond_list, bond_keys = stats.get_prior_statistics(
        features='Bonds', as_list=True)
    bond_indices = stats.return_indices('Bonds')

    angle_list, angle_keys = stats.get_prior_statistics(
        features='Angles', as_list=True)
    angle_indices = stats.return_indices('Angles')

    n_beads = coords.shape[1]
    geometry_feature = GeometryFeature(feature_tuples='all_backbone',
                                       n_beads=n_beads,
                                       device=device)

    activation = ShiftedSoftplus()

    embedding_layer = CGBeadEmbedding(n_embeddings=args.n_embeddings,
                                      embedding_dim=args.n_nodes)

    rbf_layer = GaussianRBF(high_cutoff=args.cutoff,
                            n_gaussians=args.n_gaussians)

    schnet_feature = SchnetFeature(feature_size=args.n_nodes,
                                   embedding_layer=embedding_layer,
                                   rbf_layer=rbf_layer,
                                   n_interaction_blocks=args.n_interaction_blocks,
                                   calculate_geometry=False,
                                   n_beads=n_beads,
                                   neighbor_cutoff=None,
                                   device=device)

    distance_feature_indices = stats.return_indices('Distances')
    layer_list = [geometry_feature, schnet_feature]
    feature_combiner = FeatureCombiner(layer_list,
                                       distance_indices=distance_feature_indices)

    layers = LinearLayer(args.n_nodes,
                         args.n_nodes,
                         activation=activation)

    for _ in range(args.n_layers - 1):
        layers += LinearLayer(args.n_nodes,
                              args.n_nodes,
                              activation=activation)

    # The last layer produces a single value
    layers += LinearLayer(args.n_nodes, 1, activation=None)

    priors = [HarmonicLayer(bond_indices, bond_list)]
    priors += [HarmonicLayer(angle_indices, angle_list)]

    net = CGnet(layers, ForceLoss(),
                feature=feature_combiner,
                priors=priors).to(device)

    trainloader = DataLoader(data, sampler=RandomSampler(data),
                             batch_size=args.batch_size)
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=args.learning_rate)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 30, 40, 50],
                            gamma=args.rate_decay)
    epochal_train_losses = []
    epochal_test_losses = []
    verbose = True

    # printout settings
    batch_freq = 500
    epoch_freq = 1

    for epoch in range(1, args.num_epochs+1):
        train_loss = 0.00
        test_loss = 0.00
        n = 0
        for num, batch in enumerate(trainloader):
            optimizer.zero_grad()
            coord, force, embedding_property = batch

            energy, pred_force = net.forward(coord,
                                             embedding_property=embedding_property)
            batch_loss = net.criterion(pred_force, force)
            batch_loss.backward()
            optimizer.step()

            # perform L2 lipschitz check and projection
            lipschitz_projection(net, strength=args.lipschitz_strength)
            if verbose:
                if (num+1) % batch_freq == 0:
                    print(
                        "Batch: {: <5} Train: {: <20} Test: {: <20}".format(
                            num+1, batch_loss, test_loss)
                    )
            train_loss += batch_loss.detach().cpu()
            n += 1

        train_loss /= n
        if verbose:
            if epoch % epoch_freq == 0:
                print(
                    "Epoch: {: <5} Train: {: <20} Test: {: <20}".format(
                        epoch, train_loss, test_loss))
        epochal_train_losses.append(train_loss)
        scheduler.step()

    if args.save_model:
        torch.save(net, "{}/ala2_cgschnet.pt".format(args.directory))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='このプログラムの説明（なくてもよい）')

    parser.add_argument('--coords_npy', type=str,
                        help="path to coordinates numpy file")
    parser.add_argument('--forces_npy', type=str,
                        help="path to forces numpy file")
    parser.add_argument("--n_layers", type=int, default=5,
                        help="SchNet number of layers")
    parser.add_argument("--n_nodes", type=int, default=128,
                        help="SchNet Number of node")
    parser.add_argument("--batch_size", type=int,
                        default=512, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="lr")
    parser.add_argument("--rate_decay", type=float,
                        default=0.3, help="rate decay")
    parser.add_argument("--lipschitz_strength", type=float, default=4.0,)

    parser.add_argument("--n_embeddings", type=int, default=10)
    parser.add_argument("--n_gaussians", type=int, default=50)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--n_interaction_blocks", type=int, default=5)

    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--save_model", type=bool, default=True)
    parser.add_argument("--directory", type=str, default=".")

    args = parser.parse_args()
    main(args)
