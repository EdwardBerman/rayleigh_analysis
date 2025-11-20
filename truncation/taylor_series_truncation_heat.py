import argparse
import os
import pprint
from datetime import datetime
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import entropy, gaussian_kde
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import GCN
from tqdm import tqdm

from external.unitary_gcn import UnitaryGCNConvLayer
from metrics.rayleigh import rayleigh_quotients
from model.edge_aggregator import NodeModel
from model.model_factory import UniStack, str_to_activation
from model.predictor import NodeLevelRegressor
from toy_heat_diffusion.pyg_toy import load_autoregressive_dataset
from train.train import set_seeds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def kl_divergence_samples(p, q, epsilon=1e-12):

    p = np.asarray(p)
    q = np.asarray(q)

    kde_x = gaussian_kde(p)
    kde_xprime = gaussian_kde(q)

    xmin = min(p.min(), q.min())
    xmax = max(p.max(), q.max())
    grid = np.linspace(xmin, xmax, len(q))

    p = kde_x(grid) + epsilon
    q = kde_xprime(grid) + epsilon

    p /= p.sum()
    q /= q.sum()

    kl = entropy(p, q)
    return kl


def plot_distributions(save_dir, x, xprime, y):

    plt.figure(figsize=(8, 5))
    h1 = plt.hist(x, bins=30, alpha=0.5, label='x', density=True)
    h2 = plt.hist(xprime, bins=30, alpha=0.5, label='xprime', density=True)
    h3 = plt.hist(y, bins=30, alpha=0.5, label='y', density=True)

    median_peak = np.median([h1[0].max(), h2[0].max(), h3[0].max()])

    ymax = 5 * median_peak
    plt.ylim(top=ymax)

    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Distribution of Rayleigh quotients")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "rayleigh_quotient_distribution.png"))
    plt.close()


def run_truncation_experiments(model, loader):

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
    return rayleigh_quotients_x, rayleigh_quotients_xprime, rayleigh_quotients_y


def build_model(args):

    activation_function = str_to_activation(args.ACTIVATION_FUNCTION)

    if args.architecture in ("Uni", "LieUni"):
        module_list = []

        if args.architecture == "LieUni":
            input_dim = output_dim = 1

            if args.HIDDEN_SIZE != 1:
                print(
                    f"Warning: For Lie Unitary GCN, input/hidden dims must match (got hidden {args.HIDDEN_SIZE}). "
                    "Did you mean Separable Unitary Convolution?"
                )

            for _ in range(args.NUM_LAYERS):
                module_list.append(UnitaryGCNConvLayer(
                    input_dim,
                    input_dim,
                    dropout=0.1,
                    residual=args.SKIP_CONNECTIONS,
                    global_bias=True,
                    T=args.truncation,
                    use_hermitian=True,
                    activation=activation_function()
                ))
        else:
            for layer in range(args.NUM_LAYERS):
                input_dim = 1 if layer == 0 else args.HIDDEN_SIZE
                output_dim = 1 if layer == args.NUM_LAYERS - \
                    1 else args.HIDDEN_SIZE

                module_list.append(UnitaryGCNConvLayer(
                    input_dim,
                    output_dim,
                    dropout=0.1,
                    residual=args.SKIP_CONNECTIONS,
                    global_bias=True,
                    T=args.truncation,
                    use_hermitian=False,
                    activation=activation_function()
                ))
        return UniStack(module_list)
    elif args.architecture == "GCN":
        return GCN(
            num_layers=args.NUM_LAYERS,
            in_channels=1,
            hidden_channels=args.HIDDEN_SIZE,
            out_channels=1,
            dropout=0.1,
            act=activation_function()
        )
    else:
        raise Exception("Architecture not recognized.")


def run_experiment(args, save_dir, plot=False):

    set_seeds(args.seed)

    _, eval_graphs = load_autoregressive_dataset(
        args.data_dir, args.start_time, args.train_steps, args.eval_steps
    )

    eval_loader = DataLoader(eval_graphs, batch_size=args.BATCH_SIZE)

    model = NodeLevelRegressor(NodeModel(build_model(
        args)), 1, 1, complex_floats=True).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")

    x, xprime, y = run_truncation_experiments(model, eval_loader)

    if plot:
        plot_distributions(save_dir, x, xprime, y)

    return x, xprime, y


def main(save_dir):

    config = {
        "NUM_LAYERS": 6,
        "SKIP_CONNECTIONS": False,
        "ACTIVATION_FUNCTION": "Identity",
        "BATCH_SIZE": 200,
        "BATCH_NORM": "None",
        "HIDDEN_SIZE": 64
    }

    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--start_time", type=float, default=0.0)
    parser.add_argument("--train_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=5)
    parser.add_argument("--architecture", type=str,
                        help="Uni, LieUni, GCN", required=True)
    parser.add_argument("--truncation", type=int,
                        help="Determines how truncated the taylor series is.", required=True)
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--toy", action="store_true",
                        help="Use a much smaller version of the dataset to test")

    args = parser.parse_args()
    print("Arguments:")
    pprint.pprint(vars(args))

    all_args = {**config, **vars(args)}

    x, xprime, y = run_experiment(all_args, save_dir, plot=True)

    np.save(os.path.join(save_dir, "rayleigh_quotients_x.npy"), x)
    np.save(os.path.join(save_dir, "rayleigh_quotients_xprime.npy"), xprime)
    np.save(os.path.join(save_dir, "rayleigh_quotients_y.npy"), y)


def plot_kl_divergence(all_rq_diffs, all_rq_matches, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    truncations = sorted(all_rq_diffs.keys())

    mean_diffs = [np.mean(all_rq_diffs[t]) for t in truncations]
    mean_matches = [np.mean(all_rq_matches[t]) for t in truncations]

    plt.figure(figsize=(10, 5))
    plt.bar(truncations, mean_diffs, color='skyblue')
    plt.xlabel("Truncation")
    plt.ylabel("Mean KL divergence of RQ")
    plt.title("KL divergence between Y and XPrime")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mean_rq_diffs.png"))
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.bar(truncations, mean_matches, color='salmon')
    plt.xlabel("Truncation")
    plt.ylabel("Mean KL divergence of RQ")
    plt.title("KL divergence between the RQ of X and XPrime")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "mean_rq_matches.png"))
    plt.close()


def run_all_for_architecture(architecture: str, data_dir: str):

    config = {
        "NUM_LAYERS": 6,
        "SKIP_CONNECTIONS": False,
        "ACTIVATION_FUNCTION": "Identity",
        "BATCH_SIZE": 200,
        "BATCH_NORM": "None",
        "HIDDEN_SIZE": 200
    }

    train_steps = 5
    eval_steps = 2
    start_time = 0.0

    all_rq_diffs = {}
    all_rq_matches = {}

    for truncation in tqdm(range(1, 21)):

        rq_diffs = []  # difference in rq x and xprime
        rq_matches = []  # how well xprime matches y

        for seed in range(0, 10):

            args = SimpleNamespace(
                **config,
                seed=seed,
                data_dir=data_dir,
                start_time=start_time,
                train_steps=train_steps,
                eval_steps=eval_steps,
                architecture=architecture,
                truncation=truncation,
                verbose=True,
            )

            x, xprime, y = run_experiment(
                args, save_dir, True if args.seed == 0 else False)
            # p = y, q = xprime
            # p = x, q = xprime
            rq_diffs.append(kl_divergence_samples(y, xprime))
            rq_matches.append(kl_divergence_samples(x, xprime))

        all_rq_diffs[truncation] = rq_diffs
        all_rq_matches[truncation] = rq_matches

    plot_kl_divergence(all_rq_diffs, all_rq_matches, save_dir)

    return all_rq_diffs, all_rq_matches


if __name__ == "__main__":

    save_dir = "output"

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    save_dir = os.path.join(
        save_dir, current_time)
    os.makedirs(save_dir, exist_ok=True)

    rq_diffs, rq_matches = run_all_for_architecture(
        "Uni", "toy_heat_diffusion/data")
