import torch
import torch.nn.functional as F
import numpy as np

from metrics.rayleigh import rayleigh_quotients
import matplotlib.pyplot as plt

import os

@torch.no_grad()
def evaluate_heat_flow(model, loader, device):
    model.eval()
    total_mse, total_nodes = 0, 0
    rayleigh_quotients_x = []
    rayleigh_quotients_xprime = []
    rayleigh_quotients_y = []

    for data in loader:
        data = data.to(device)
        out = model(data)
        mse = F.mse_loss(out, data.y, reduction="sum").item()
        total_mse += mse
        total_nodes += data.num_nodes

        x, xprime, y = rayleigh_quotients(model, data)
        rayleigh_quotients_x.append(x.item())
        rayleigh_quotients_xprime.append(xprime.item())
        rayleigh_quotients_y.append(y.item())

    avg_mse = total_mse / total_nodes

    return avg_mse, np.mean(rayleigh_quotients_x), np.mean(rayleigh_quotients_xprime), np.mean(rayleigh_quotients_y)


@torch.no_grad()
def rayleigh_quotient_distribution(model, loader, device, save_dir):
    model.eval()
    rayleigh_quotients_x = []
    rayleigh_quotients_xprime = []
    rayleigh_quotients_y = []

    for data in loader:
        data = data.to(device)
        x, xprime, y = rayleigh_quotients(model, data)
        rayleigh_quotients_x.append(x.item())
        rayleigh_quotients_xprime.append(xprime.item())
        rayleigh_quotients_y.append(y.item())

    rayleigh_quotients_xprime = np.array(rayleigh_quotients_xprime)
    plt.hist(rayleigh_quotients_xprime, bins=30, density=True)
    plt.xlabel(r"R_{\mathcal{G}}(f(X))")
    plt.ylabel("Density")
    plt.title("Distribution of Rayleigh Quotients")
    plt.savefig(os.path.join(save_dir, "rayleigh_quotient_distribution.png"))
    plt.close()
    np.save(os.path.join(save_dir, "rayleigh_quotients_x.npy"), rayleigh_quotients_x)
    np.save(os.path.join(save_dir, "rayleigh_quotients_xprime.npy"), rayleigh_quotients_xprime)
    np.save(os.path.join(save_dir, "rayleigh_quotients_y.npy"), rayleigh_quotients_y)






