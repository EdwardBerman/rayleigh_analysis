# autoregressive_pyg.py
import os
import pickle
import re

import torch
from torch_geometric.data import Data


def graphs_to_autoregressive_pyg(graphs_by_time, start_index, end_index):
    """
    Converts consecutive time steps into autoregressive PyG Data objects.
    Input at t: xt-1
    Target at t: xt
    """
    pyg_graphs = []
    times = sorted(graphs_by_time.keys())

    for i in range(start_index, min(end_index, len(times) - 1)):

        t_prev, t_next = times[i], times[i+1]
        graphs_prev, graphs_next = graphs_by_time[t_prev], graphs_by_time[t_next]

        for g_prev, g_next in zip(graphs_prev, graphs_next):
            x_prev = torch.tensor(g_prev['xt'], dtype=torch.float).unsqueeze(1)
            x_next = torch.tensor(g_next['xt'], dtype=torch.float).unsqueeze(1)
            edge_index = torch.tensor(
                list(zip(*g_prev['A'].nonzero())), dtype=torch.long)

            pyg_graphs.append(Data(x=x_prev, edge_index=edge_index, y=x_next))

    return pyg_graphs


def load_autoregressive_dataset(data_dir, start_time, train_steps, eval_steps):
    """
    Loads consecutive time steps from disk and returns train/eval datasets.
    """
    graphs_by_time = {}
    for fname in os.listdir(data_dir):
        if fname.endswith(".pkl"):
            t = float(fname.replace("graphs_t", "").replace(".pkl", ""))
            with open(os.path.join(data_dir, fname), "rb") as f:
                graphs_by_time[t] = pickle.load(f)

    times = sorted(graphs_by_time.keys())

    train_start_idx = times.index(start_time)
    train_end_idx = train_start_idx + train_steps
    test_start_idx = train_end_idx
    test_end_idx = test_start_idx + eval_steps

    train_graphs = graphs_to_autoregressive_pyg(
        graphs_by_time, train_start_idx, train_end_idx)

    eval_graphs = graphs_to_autoregressive_pyg(
        graphs_by_time, test_start_idx, test_end_idx)

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
