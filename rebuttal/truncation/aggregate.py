import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evaluation.plotting_params import set_rc_params
from toy_heat_diffusion.aggregate import load_metric_from_runs


def find_best_run(runs, metric_name):
    """Find the run with the lowest final value for the given metric."""
    metric_list = load_metric_from_runs(runs, metric_name)
    if not metric_list:
        return None, None

    final_values = [m[-1] for m in metric_list]
    best_idx = np.argmin(final_values)
    return best_idx, metric_list[best_idx]


def collect_runs(aggregate_dir):
    runs = {
        'lie_1': [],
        'lie_3': [],
        'lie_5': [],
        'lie_7': [],
        'lie_9': []
    }

    for subdir in Path(aggregate_dir).iterdir():
        if not subdir.is_dir():
            continue

        dir_name = subdir.name
        if dir_name.startswith('lie_unitary_trunc1_'):
            runs['lie_1'].append(subdir)
        elif dir_name.startswith('lie_unitary_trunc3_'):
            runs['lie_3'].append(subdir)
        elif dir_name.startswith('lie_unitary_trunc5_'):
            runs['lie_5'].append(subdir)
        elif dir_name.startswith('lie_unitary_trunc7_'):
            runs['lie_7'].append(subdir)
        elif dir_name.startswith('lie_unitary_trunc9_'):
            runs['lie_9'].append(subdir)

    for key in runs:
        runs[key] = sorted(runs[key])

    return runs


def save_metrics_table(runs, model_types, model_labels, save_dir):
    import pandas as pd

    rows = []
    for model_type, label in zip(model_types, model_labels):
        val_mse_list = load_metric_from_runs(runs[model_type], "val_mse")
        rayleigh_list = load_metric_from_runs(
            runs[model_type], "val_rayleigh_xprime")

        for i, (mse, rayleigh) in enumerate(zip(val_mse_list, rayleigh_list)):
            rows.append({
                'Model': label,
                'Trial': i,
                'Final Val MSE': mse[-1],
                'Best Val MSE': np.min(mse),
                'Best MSE Epoch': np.argmin(mse),
                'Final Rayleigh': rayleigh[-1],
                'Best Rayleigh': np.min(rayleigh),
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(save_dir, "metrics_table.csv"), index=False)
    print("Saved: metrics_table.csv")
    return df


def main():
    set_rc_params(10)

    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregate_dir", type=str,
                        default="outputs/aggregate")
    parser.add_argument("--save_dir", type=str, default="assets")
    args = parser.parse_args()

    runs = collect_runs(args.aggregate_dir)

    os.makedirs(args.save_dir, exist_ok=True)

    model_types = ['lie_1', 'lie_3', 'lie_5', 'lie_7', 'lie_9']
    model_labels = [r'$K=1$', r'$K=3$', r'$K=5$', r'$K=7$', r'$K=9$']
    model_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    val_rayleigh_x = None
    val_rayleigh_y = None
    for model_type in model_types:
        if len(runs[model_type]) > 0:
            x_path = runs[model_type][0] / "val_rayleigh_x.npy"
            y_path = runs[model_type][0] / "val_rayleigh_y.npy"
            if x_path.exists():
                val_rayleigh_x = np.load(x_path)
                print(
                    f"Loaded X rayleigh quotients from {runs[model_type][0].name}")
            if y_path.exists():
                val_rayleigh_y = np.load(y_path)
                print(
                    f"Loaded Y rayleigh quotients from {runs[model_type][0].name}")
            if val_rayleigh_x is not None and val_rayleigh_y is not None:
                break

    fig, axes = plt.subplots(1, 5, figsize=(15, 4), sharey=True)

    titles = [r'$K=1$', r'$K=3$', r'$K=5$', r'$K=7$', r'$K=9$']
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, 5))

    for ax, model_type, title in zip(axes, model_types, titles):
        val_rayleigh_xprime_list = load_metric_from_runs(
            runs[model_type], "val_rayleigh_xprime")

        for i, rxp in enumerate(val_rayleigh_xprime_list):
            num_epochs = len(rxp)
            ax.plot(
                np.arange(num_epochs),
                rxp,
                color=colors[i % len(colors)],
                linewidth=2,
                alpha=0.7,
                label=r"$\overline{R_{{\mathcal{{G}}}}(f(X))}$" + f" Run {i+1}"
            )

        if val_rayleigh_x is not None:
            num_epochs = len(val_rayleigh_x)
            ax.plot(
                np.arange(num_epochs),
                val_rayleigh_x,
                color='blue',
                linewidth=2.5,
                alpha=0.9,
                linestyle='--',
                label=r"$\overline{R_{\mathcal{G}}(X)}$"
            )

        if val_rayleigh_y is not None:
            num_epochs = len(val_rayleigh_y)
            ax.plot(
                np.arange(num_epochs),
                val_rayleigh_y,
                color='red',
                linewidth=2.5,
                alpha=0.9,
                linestyle='--',
                label=r"$\overline{R_{\mathcal{G}}(Y)}$"
            )

        ax.set_xlabel("Epoch", fontsize=20)
        ax.set_title(title, fontsize=22)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel(r"$\overline{R_{\mathcal{G}}}$", fontsize=20)
    handles, labels = axes[-1].get_legend_handles_labels()
    print(f"Number of handles: {len(handles)}, labels: {labels}")  # debug
    axes[-1].legend(handles, labels, fontsize=14,
                    loc='upper left', bbox_to_anchor=(1.01, 1.0),
                    borderaxespad=0)
    plt.tight_layout()
    plt.subplots_adjust(right=0.80)
    plt.savefig(os.path.join(args.save_dir,
                "rayleigh_comparison_all_runs.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: rayleigh_comparison_all_runs.png")

    fig, axes = plt.subplots(1, 5, figsize=(15, 4), sharey=True)

    for ax, model_type, title in zip(axes, model_types, titles):
        val_mse_list = load_metric_from_runs(runs[model_type], "val_mse")

        for i, mse in enumerate(val_mse_list):
            num_epochs = len(mse)
            ax.plot(
                np.arange(num_epochs),
                mse,
                color=colors[i % len(colors)],
                linewidth=2.5,
                alpha=0.7,
                label=f"Run {i+1}"
            )

        ax.set_xlabel("Epoch", fontsize=20)
        ax.set_title(title, fontsize=22)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Validation MSE", fontsize=20)
    handles, labels = axes[-1].get_legend_handles_labels()
    print(f"Number of handles: {len(handles)}, labels: {labels}")  # debug
    axes[-1].legend(handles, labels, fontsize=14, 
                    loc='upper left', bbox_to_anchor=(1.01, 1.0),
                    borderaxespad=0)
    plt.tight_layout()
    plt.subplots_adjust(right=0.80)
    plt.savefig(os.path.join(args.save_dir,
                "val_mse_comparison_all_runs.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: val_mse_comparison_all_runs.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax_rayleigh = axes[0]
    for model_type, label, color in zip(model_types, model_labels, model_colors):
        best_idx, best_rayleigh = find_best_run(runs[model_type], "val_mse")
        if best_rayleigh is None:
            continue

        rayleigh_list = load_metric_from_runs(
            runs[model_type], "val_rayleigh_xprime")
        if best_idx < len(rayleigh_list):
            best_rayleigh_curve = rayleigh_list[best_idx]
            num_epochs = len(best_rayleigh_curve)
            ax_rayleigh.plot(
                np.arange(num_epochs),
                best_rayleigh_curve,
                color=color,
                linewidth=4.5,
                alpha=0.9,
                label=label
            )
            print(f"{label}: Best run is Run {best_idx+1}")

    if val_rayleigh_x is not None:
        num_epochs = len(val_rayleigh_x)
        ax_rayleigh.plot(
            np.arange(num_epochs),
            val_rayleigh_x,
            color='gray',
            linewidth=4.5,
            alpha=0.7,
            linestyle='--',
            label=r"$\overline{R_{\mathcal{G}}(X)}$"
        )

    if val_rayleigh_y is not None:
        num_epochs = len(val_rayleigh_y)
        ax_rayleigh.plot(
            np.arange(num_epochs),
            val_rayleigh_y,
            color='red',
            linewidth=4.5,
            alpha=0.7,
            linestyle='--',
            label=r"$\overline{R_{\mathcal{G}}(Y)}$"
        )

    ax_rayleigh.set_xlabel("Epoch", fontsize=25)
    ax_rayleigh.set_ylabel(
        r"Validation $\overline{R_{\mathcal{G}}}$", fontsize=25)
    ax_rayleigh.set_title("Rayleigh Quotient", fontsize=25)
    # ax_rayleigh.legend(fontsize=16, loc='upper right')
    ax_rayleigh.legend(fontsize=20, loc='center left',
                       bbox_to_anchor=(-0.95, 0.5))
    ax_rayleigh.grid(True, alpha=0.3)

    ax_mse = axes[1]
    for model_type, label, color in zip(model_types, model_labels, model_colors):
        best_idx, best_mse = find_best_run(runs[model_type], "val_mse")
        if best_mse is None:
            continue

        num_epochs = len(best_mse)
        ax_mse.plot(
            np.arange(num_epochs),
            best_mse,
            color=color,
            linewidth=4.5,
            alpha=0.9,
            label=label
        )

    ax_mse.set_xlabel("Epoch", fontsize=25)
    ax_mse.set_ylabel("Validation MSE", fontsize=25)
    ax_mse.set_title("Validation MSE", fontsize=25)
    ax_mse.set_yscale("log")
    ax_mse.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir,
                "best_runs_comparison.png"), dpi=300)
    plt.close()
    print(f"Saved: best_runs_comparison.png")

    df = save_metrics_table(runs, model_types, model_labels, args.save_dir)

    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
