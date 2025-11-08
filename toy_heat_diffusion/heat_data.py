"""Generates toy examples for graph smoothness based on heat diffusion on graphs. Simulates different levels of smoothness by taking different time steps in the heat diffusion process."""

import argparse
import os
import pickle

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pygsp as pg

from tqdm import tqdm


def generate_num_nodes(num: int, mean: float, variance: float) -> list[int]:
    """Generates the number of nodes of the graphs by drawing from a Gaussian"""
    std = np.sqrt(variance)
    samples = np.random.normal(
        loc=mean, scale=std, size=num).round().astype(int)
    return samples


def generate_heat_graph(n_nodes: int, density: float, n_sources: int, heat_max: float, heat_min: float, times: list[int]):
    """
    Generates a heat graph, diffused to discrete time steps. 

    Parameters
    ----------
    n_nodes : int
        Number of nodes in the graph
    density : float
        Density of the connections in the graph 
    n_sources : int
        Number of heat sources
    heat_max : float
        Maximum initial heat at sources
    heat_min : float
        Minimum initial heat at sources
    times : list[int]
        Discrete time steps at which to take snapshots of the graph. 
    """
    G = pg.graphs.ErdosRenyi(n_nodes, density, connected=True)
    G.compute_fourier_basis()
    sources = np.random.choice(G.N, n_sources, replace=False)
    x0 = np.zeros(G.N)
    x0[sources] = np.random.uniform(heat_min, heat_max, size=n_sources)
    X = np.stack([pg.filters.Heat(G, scale=t).filter(x0)
                  for t in times], axis=1)
    A = G.W.toarray()
    return X, A, G


def generate_heat_grid_graph(n_nodes_side: int, n_sources: int, heat_max: float, heat_min: float, times: list[int]):
    G = pg.graphs.Grid2d(n_nodes_side)
    G.compute_fourier_basis()
    sources = np.random.choice(G.N, n_sources, replace=False)
    x0 = np.zeros(G.N)
    x0[sources] = np.random.uniform(heat_min, heat_max, size=n_sources)
    X = np.stack([pg.filters.Heat(G, scale=t).filter(x0)
                  for t in times], axis=1)
    A = G.W.toarray()
    return X, A, G


def visualize_heat_diffusion(G, X, times, save_dir=None):
    """
    Visualizes graph heat diffusion with a fixed colormap scale across time,
    optionally saving a video of the diffusion.
    """
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    coords = G.coords

    vmin, vmax = X.min(), X.max()
    frames = []

    for i, t in enumerate(times):
        fig, ax = plt.subplots(figsize=(6, 6))

        sc = ax.scatter(coords[:, 0], coords[:, 1], c=X[:, i],
                        cmap="coolwarm", s=200, vmin=vmin, vmax=vmax)

        ax.set_title(f"Heat diffusion at t={t}")
        ax.set_aspect('equal')
        ax.set_axis_off()

        plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, cmap='coolwarm')

        if save_dir:
            frame_path = os.path.join(save_dir, f"heat_t{t:.2f}.png")
            plt.savefig(frame_path, dpi=200)
            frames.append(frame_path)

        plt.close(fig)

    video_path = os.path.join(save_dir, "graph_evolution.mp4")
    with imageio.get_writer(video_path, mode='I', fps=5) as writer:
        for frame in frames:
            image = imageio.imread(frame)
            writer.append_data(image)
    print(f"Video saved to {video_path}")


def main(save_dir: str):

    parser = argparse.ArgumentParser(
        description="Parameters to generate graph heat diffusion data")

    parser.add_argument("--graph_type", type=str, default="grid",
                        help="Type of graph, can be either grid or a standard random graph.")
    parser.add_argument("--n_sources", type=int, default=5,
                        help="Number of heat sources")
    parser.add_argument("--minheat", type=float,
                        default=5, help="Minimum heat value")
    parser.add_argument("--maxheat", type=float,
                        default=10, help="Maximum heat value")
    parser.add_argument("--time_max", type=float, default=10.0,
                        help="Maximum time value (exclusive)")
    parser.add_argument("--time_step", type=float, default=0.5,
                        help="Step size for time values (can be fractional)")
    parser.add_argument("--num_graphs", type=int, default=10,
                        help="Number of graphs to generate")
    parser.add_argument("--size_mean", type=float, default=100,
                        help="Mean of graph size distribution")
    parser.add_argument("--size_std", type=float, default=5,
                        help="Standard deviation of graph size distribution")

    args = parser.parse_args()

    print(args)

    graph_type = args.graph_type
    n_sources = args.n_sources
    heat_min = args.minheat
    heat_max = args.maxheat
    times = np.arange(0, args.time_max, args.time_step).tolist()
    num_graphs = args.num_graphs
    size_mean = args.size_mean
    size_std = args.size_std

    num_nodes_list = generate_num_nodes(num_graphs, size_mean, size_std)
    data_by_time = {t: [] for t in times}

    for i, n_nodes in enumerate(num_nodes_list):

        if args.graph_type == "grid":
            X, A, G = generate_heat_grid_graph(
                n_nodes, n_sources, heat_max, heat_min, times
            )
        else:
            X, A, G = generate_heat_graph(
                n_nodes, 0.3, n_sources, heat_max, heat_min, times)

        # for the first graph, visualize it
        if i == 0:
            visualize_heat_diffusion(G, X, times, save_dir + "/demo")

        for t_idx, t in enumerate(times):
            graph_data = {
                'xt': X[:, t_idx],
                'A': A,
            }
            data_by_time[t].append(graph_data)

    for t in tqdm(times, desc="Saving data by time step"):
        time_step_file = os.path.join(save_dir, f"graphs_t{t}.pkl")
        with open(time_step_file, 'wb') as f:
            pickle.dump(data_by_time[t], f)


if __name__ == "__main__":
    main(save_dir="toy_heat_diffusion/data")
