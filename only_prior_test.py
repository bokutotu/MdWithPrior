import argparse

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


from src.features.angles import AngleLayer
from src.features.dihedral import DihedralLayer
from src.features.length import LengthLayer
from src.dataset import MLPDataset
from src.layers.prior import PriorEnergyLayer


class Model(nn.Module):

    def __init__(self, num_atom):
        super().__init__()
        self.angle = AngleLayer()
        self.dihedral = DihedralLayer()
        self.lentgh = LengthLayer()

        self.angle_prior_layer = PriorEnergyLayer(num_atom - 2)
        self.length_prior_layer = PriorEnergyLayer(num_atom - 1)
        self.dihedral_prior_layer = PriorEnergyLayer(2 * (num_atom - 3))

    def forward(self, x):
        energy = self.angle_prior_layer(self.angle(x))
        energy = energy + self.length_prior_layer(self.lentgh(x))
        energy = energy + self.dihedral_prior_layer(self.dihedral(x))

        force = torch.autograd.grad(-torch.sum(energy),
                                    x,
                                    create_graph=True,
                                    retain_graph=True)
        return force[0], energy


def main(args):
    coord = np.load(args.coord_path)
    force = np.load(args.force_path)

    model = Model(coord.shape[1]).to("cuda")

    optim = Adam(params=model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optim, mode="min", factor=0.5)

    dataset = MLPDataset(coordinates=coord, forces=force)
    dataloader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=True, num_workers=6)

    mse_loss = nn.MSELoss()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for idx, batch in enumerate(dataloader):
            x, y = batch
            x = x.requires_grad_(True).to("cuda")
            y = y.requires_grad_(True).to("cuda")

            optim.zero_grad()
            force, energy = model(x)
            loss = mse_loss(force, y)
            loss.backward()
            optim.step()
            epoch_loss += float(loss.detach().cpu())
        epoch_loss /= len(dataloader)
        print("epoch {} loss is {:.5e}".format(epoch+1, epoch_loss))

    torch.save(model.state_dict(), args.model_save_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pred Model only using Prior Energy")

    parser.add_argument("--epochs", type=int, default=100,
                        help="how many epochs to train")
    parser.add_argument("--coord_path", type=str,
                        default="/home/bokutotu/HDD/Lab/data/NPY/c_trj.npy",
                        help="coordinates npy file")
    parser.add_argument("--force_path", type=str,
                        default="/home/bokutotu/HDD/Lab/data/NPY/f_trj.npy",
                        help="force npy file")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--batch_size", type=int,
                        default=128, help="batch_size")
    parser.add_argument("--model_save_name", type=str,
                        default="prior_only.pth", help="model save path")

    args = parser.parse_args()

    main(args)
