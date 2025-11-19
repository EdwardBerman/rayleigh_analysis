import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from metrics.rayleigh import rayleigh_quotients, rayleigh_quotients_graphlevel


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
    np.save(os.path.join(save_dir, "rayleigh_quotients_x.npy"),
            rayleigh_quotients_x)
    np.save(os.path.join(save_dir, "rayleigh_quotients_xprime.npy"),
            rayleigh_quotients_xprime)
    np.save(os.path.join(save_dir, "rayleigh_quotients_y.npy"),
            rayleigh_quotients_y)


@torch.no_grad()
def rayleigh_quotient_distribution_graphlevel(model, loader, device, save_dir):
    model.eval()
    rayleigh_quotients_x = []
    rayleigh_quotients_xprime = []

    for data in loader:
        data = data.to(device)
        x, xprime = rayleigh_quotients_graphlevel(model, data)
        rayleigh_quotients_x.append(x.item())
        rayleigh_quotients_xprime.append(xprime.item())

    rayleigh_quotients_xprime = np.array(rayleigh_quotients_xprime)
    plt.hist(rayleigh_quotients_xprime, bins=30, density=True)
    plt.xlabel(r"R_{\mathcal{G}}(f(X))")
    plt.ylabel("Density")
    plt.title("Distribution of Rayleigh Quotients")
    plt.savefig(os.path.join(save_dir, "rayleigh_quotient_distribution.png"))

    plt.close()
    np.save(os.path.join(save_dir, "rayleigh_quotients_x.npy"),
            rayleigh_quotients_x)
    np.save(os.path.join(save_dir, "rayleigh_quotients_xprime.npy"),
            rayleigh_quotients_xprime)
