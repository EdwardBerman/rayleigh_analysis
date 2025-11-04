"""Generates toy examples for graph smoothness."""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pygsp as pg


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
    G = pg.graphs.ErdosRenyi(n_nodes, density)
    G.compute_fourier_basis()
    sources = np.random.choice(G.N, n_sources, replace=False)
    x0 = np.zeros(G.N)
    x0[sources] = np.random.uniform(heat_min, heat_max, size=n_sources)
    X = np.stack([pg.filters.Heat(G, scale=t).filter(x0)
                  for t in times], axis=1)
    A = G.W.toarray()
    return X, A, G, x0


def visualize_heat_diffusion(G, X, times, save_dir=None):
    """
    Visualizes the heat diffusion on the graph at each time step.

    Parameters
    ----------
    G : pygsp.graphs.Graph
        The graph object.
    X : np.ndarray
        Heat values at each node over time (shape: [n_nodes, n_times]).
    x0 : np.ndarray
        Initial heat values at nodes.
    times : list[int]
        Time steps corresponding to columns of X.
    save_dir : str, optional
        Directory to save the plots. If None, plots are shown but not saved.
    """
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    G.set_coordinates()

    for i, t in enumerate(times):
        fig, ax = plt.subplots(figsize=(6, 6))
        G.plot(X[:, i], vertex_size=50, edge_width=1.0, ax=ax)
        ax.set_title(f"Heat diffusion at t={t}")
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"heat_t{t}.png"))
        
        plt.close(fig)


def main(save_dir: str):

    n_nodes = 100
    density = 0.05
    n_sources = 10
    heat_min, heat_max = 8, 10
    times = list(range(10))
    num_graphs = 10
    size_mean = 100
    size_std = 5

    num_nodes_list = generate_num_nodes(num_graphs, size_mean, size_std)
    data_by_time = {t: [] for t in times}

    for i, n_nodes in enumerate(num_nodes_list):

        X, A, G, x0 = generate_heat_graph(
            n_nodes, density, n_sources, heat_max, heat_min, times
        )

        # for the first graph, visualize it
        if i == 0:
            visualize_heat_diffusion(G, X, times, save_dir + "/demo")

        for t_idx, t in enumerate(times):
            graph_data = {
                'xt': X[:, t_idx],
                'A': A,
                'x0': x0
            }
            data_by_time[t].append(graph_data)

    for t in times:
        time_step_file = os.path.join(save_dir, f"graphs_t{t}.pkl")
        with open(time_step_file, 'wb') as f:
            pickle.dump(data_by_time[t], f)


if __name__ == "__main__":
    main(save_dir="./data/heat/toy")
