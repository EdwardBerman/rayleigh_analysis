import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib import rc

def set_rc_params(fontsize=None):
    '''
    Set figure parameters
    '''

    if fontsize is None:
        fontsize = 16
    else:
        fontsize = int(fontsize)

    rc('font', **{'family': 'serif'})
    rc('text', usetex=False)

    plt.rcParams.update({'axes.linewidth': 1.3})
    plt.rcParams.update({'xtick.labelsize': fontsize})
    plt.rcParams.update({'ytick.labelsize': fontsize})
    plt.rcParams.update({'xtick.major.size': 8})
    plt.rcParams.update({'xtick.major.width': 1.3})
    plt.rcParams.update({'xtick.minor.visible': True})
    plt.rcParams.update({'xtick.minor.width': 1.})
    plt.rcParams.update({'xtick.minor.size': 6})
    plt.rcParams.update({'xtick.direction': 'out'})
    plt.rcParams.update({'ytick.major.width': 1.3})
    plt.rcParams.update({'ytick.major.size': 8})
    plt.rcParams.update({'ytick.minor.visible': True})
    plt.rcParams.update({'ytick.minor.width': 1.})
    plt.rcParams.update({'ytick.minor.size': 6})
    plt.rcParams.update({'ytick.direction': 'out'})
    plt.rcParams.update({'axes.labelsize': fontsize})
    plt.rcParams.update({'axes.titlesize': fontsize})
    plt.rcParams.update({'legend.fontsize': int(fontsize-2)})
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amssymb}'



def main():
    set_rc_params(10)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir_GCN", type=str, required=True)
    parser.add_argument("--data_dir_UNI", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="assets")
    args = parser.parse_args()

    val_mse_GCN = np.load(os.path.join(args.data_dir_GCN, "val_mse.npy"))
    val_rayleigh_x_GCN = np.load(os.path.join(args.data_dir_GCN, "val_rayleigh_x.npy"))
    val_rayleigh_xprime_GCN = np.load(os.path.join(args.data_dir_GCN, "val_rayleigh_xprime.npy"))
    num_epochs = len(val_mse_GCN)

    val_mse_UNI = np.load(os.path.join(args.data_dir_UNI, "val_mse.npy"))
    val_rayleigh_x_UNI = np.load(os.path.join(args.data_dir_UNI, "val_rayleigh_x.npy"))
    val_rayleigh_xprime_UNI = np.load(os.path.join(args.data_dir_UNI, "val_rayleigh_xprime.npy"))

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].plot(
        np.arange(num_epochs),
        val_mse_GCN,
        color="blue",
        label="GCN",
        linewidth=2,
    )
    ax[0].plot(
        np.arange(num_epochs),
        val_mse_UNI,
        color="red",
        label="Uni",
        linewidth=2,
    )

    ax[0].set_xlabel("Epoch", fontsize=24)
    ax[0].set_ylabel("Validation MSE", fontsize=24)
    ax[0].legend()

    ax[1].plot(
        np.arange(num_epochs),
        val_rayleigh_x_GCN - val_rayleigh_xprime_GCN,
        color="blue",
        label=r"$R_{\mathcal{G}}(X) - R_{\mathcal{G}}(f_{\rm GCN}(X))$",
        linewidth=2,
    )

    ax[1].plot(
        np.arange(num_epochs),
        val_rayleigh_x_UNI - val_rayleigh_xprime_UNI,
        color="red",
        label=r"$R_{\mathcal{G}}(X) - R_{\mathcal{G}}(f_{\rm Uni-GCN}(X))$",
        linewidth=2,
    )

    ax[1].set_xlabel("Epoch", fontsize=24)
    ax[1].set_ylabel(r"$\Delta R_{\mathcal{G}}$", fontsize=24)
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "validation_comparison.png"), dpi=300)



if __name__ == "__main__":
    main()
