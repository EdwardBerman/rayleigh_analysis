import os
import pickle
import re

import torch
from torch_geometric.data import Data


def graphs_to_autoregressive_pyg(graphs_by_time, start_time, num_steps):
    """
    Converts consecutive time steps into autoregressive PyG Data objects.

    Input at t: xt-1
    Target at t: xt
    """
    import torch
    from torch_geometric.data import Data

    pyg_graphs = []

    times = sorted(graphs_by_time.keys())
    start_idx = times.index(start_time)

    end_idx = start_idx + num_steps
    if end_idx >= len(times):
        raise ValueError(
            f"Not enough time steps :(")

    for i in range(start_idx, end_idx):
        t_prev = times[i]
        t_next = times[i + 1]

        graphs_prev = graphs_by_time[t_prev]
        graphs_next = graphs_by_time[t_next]

        for g_prev, g_next in zip(graphs_prev, graphs_next):
            x_prev = torch.tensor(g_prev['xt'], dtype=torch.float).unsqueeze(1)
            x_next = torch.tensor(g_next['xt'], dtype=torch.float).unsqueeze(1)
            edge_index = torch.tensor(
                list(zip(*g_prev['A'].nonzero())), dtype=torch.long)

            data = Data(x=x_prev, edge_index=edge_index, y=x_next)
            pyg_graphs.append(data)

    return pyg_graphs


def load_autoregressive_dataset(data_dir, start_time, train_steps, eval_steps, time_step_fmt="graphs_t{t}.pkl"):
    """
    Loads consecutive time steps from disk and returns train/eval autoregressive datasets.

    Parameters
    ----------
    data_dir : str
        Directory with pickled graphs.
    start_time : float
        Time step to start from.
    train_steps : int
        Number of consecutive steps for training.
    eval_steps : int
        Number of consecutive steps for evaluation.
    time_step_fmt : str
        Format string for filenames, e.g., "graphs_t{t}.pkl"

    Returns
    -------
    train_graphs, eval_graphs : lists of PyG Data objects
    """
    regex_pattern = re.escape(time_step_fmt).replace(r"\{t\}", r"([0-9.]+)")

    graphs_by_time = {}
    for fname in os.listdir(data_dir):
        match = re.match(regex_pattern, fname)
        if match:
            t = float(match.group(1))
            with open(os.path.join(data_dir, fname), "rb") as f:
                graphs_by_time[t] = pickle.load(f)

    train_graphs = graphs_to_autoregressive_pyg(
        graphs_by_time, start_time, train_steps)
    eval_start_time = start_time + train_steps
    eval_graphs = graphs_to_autoregressive_pyg(
        graphs_by_time, eval_start_time, eval_steps)

    return train_graphs, eval_graphs


if __name__ == "__main__":
    
    data_dir = "toy_heat_diffusion/data"
    start_time = 0.0
    train_steps = 5
    eval_steps = 2

    train_graphs, eval_graphs = load_autoregressive_dataset(
        data_dir, start_time, train_steps, eval_steps
    )

    print(
        f"Train graphs: {len(train_graphs)}, Eval graphs: {len(eval_graphs)}")
