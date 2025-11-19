import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt

from evaluation.plotting_params import set_rc_params

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
    num_epochs_GCN = len(val_mse_GCN)

    val_mse_UNI = np.load(os.path.join(args.data_dir_UNI, "val_mse.npy"))
    val_rayleigh_x_UNI = np.load(os.path.join(args.data_dir_UNI, "val_rayleigh_x.npy"))
    val_rayleigh_xprime_UNI = np.load(os.path.join(args.data_dir_UNI, "val_rayleigh_xprime.npy"))

    num_epochs_UNI = len(val_mse_UNI)

    val_rayleigh_xprime_UNI_final_epoch = np.load(os.path.join(args.data_dir_UNI, "rayleigh_quotients_xprime.npy"))
    val_rayleigh_xprime_GCN_final_epoch = np.load(os.path.join(args.data_dir_GCN, "rayleigh_quotients_xprime.npy"))
    val_rayleigh_x_UNI_final_epoch = np.load(os.path.join(args.data_dir_UNI, "rayleigh_quotients_x.npy"))
    val_rayleigh_x_GCN_final_epoch = np.load(os.path.join(args.data_dir_GCN, "rayleigh_quotients_x.npy"))
    val_rayleigh_y_UNI_final_epoch = np.load(os.path.join(args.data_dir_UNI, "rayleigh_quotients_y.npy"))
    val_rayleigh_y_GCN_final_epoch = np.load(os.path.join(args.data_dir_GCN, "rayleigh_quotients_y.npy"))


    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].plot(
        np.arange(num_epochs_GCN),
        val_mse_GCN,
        color="blue",
        label="GCN",
        linewidth=2,
    )
    ax[0].plot(
        np.arange(num_epochs_UNI),
        val_mse_UNI,
        color="red",
        label="Uni",
        linewidth=2,
    )

    ax[0].set_xlabel("Epoch", fontsize=24)
    ax[0].set_ylabel("Validation MSE", fontsize=24)
    ax[0].legend()

    ax[1].plot(
        np.arange(num_epochs_GCN),
        val_rayleigh_x_GCN - val_rayleigh_xprime_GCN,
        color="blue",
        label=r"$\overline{R_{\mathcal{G}}(X)} - \overline{R_{\mathcal{G}}(f_{\rm GCN}(X))}$",
        linewidth=2,
    )

    ax[1].plot(
        np.arange(num_epochs_UNI),
        val_rayleigh_x_UNI - val_rayleigh_xprime_UNI,
        color="red",
        label=r"$\overline{R_{\mathcal{G}}(X)} - \overline{R_{\mathcal{G}}(f_{\rm Uni-GCN}(X))}$",
        linewidth=2,
    )

    ax[1].set_xlabel("Epoch", fontsize=24)
    ax[1].set_ylabel(r"$\Delta \overline{R_{\mathcal{G}}}$", fontsize=24)
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "validation_comparison.png"), dpi=300)
    plt.close()

    # histogram of final epoch rayleigh_quotients_xprime
    plt.figure(figsize=(6, 4))
    plt.hist(
        val_rayleigh_xprime_GCN_final_epoch,
        bins=30,
        alpha=0.25,
        label=r"R_{\mathcal{G}}(f_{\rm GCN}(X)}})",
        color="blue",
        density=True,
    )
    plt.hist(
        val_rayleigh_xprime_UNI_final_epoch,
        bins=30,
        alpha=0.25,
        label=r"$R_{\mathcal{G}}(f_{\rm Uni-GCN}(X)}}$",
        color="red",
        density=True,
    )
    plt.hist(
        val_rayleigh_x_GCN_final_epoch,
        bins=30,
        alpha=0.25,
        label=r"$R_{\mathcal{G}}(X)}$",
        color="cyan",
        density=True,
    )
    plt.hist(
        val_rayleigh_x_UNI_final_epoch,
        bins=30,
        alpha=0.25,
        label=r"$R_{\mathcal{G}}(X)}$",
        color="magenta",
        density=True,
    )
    plt.xlabel(r"$R_{\mathcal{G}}$", fontsize=24)
    plt.ylabel("Density", fontsize=24)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "rayleigh_quotients_comparison.png"), dpi=300)
    plt.close()



if __name__ == "__main__":
    main()
