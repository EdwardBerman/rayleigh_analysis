import pickle

import torch
from torch_geometric.data import Data


def graphs_to_pyg_data(graphs_list):
    """
    Converts a list of graphs (from one time step) to PyTorch Geometric Data objects.

    Parameters
    ----------
    graphs_list : list[dict]
        Each dict contains:
            'xt': node features / target at this time step (1D array)
            'A': adjacency matrix (numpy array)
            'x0': initial heat (optional)

    Returns
    -------
    pyg_graphs : list[torch_geometric.data.Data]
        PyG Data objects ready for node-level regression
    """
    pyg_graphs = []

    for g in graphs_list:
        x_t = torch.tensor(g['xt'], dtype=torch.float).unsqueeze(1)
        x0 = torch.tensor(g['x0'], dtype=torch.float).unsqueeze(1)
        node_features = torch.cat([x0, x_t], dim=1)

        A = g['A']
        src, dst = A.nonzero()
        edge_index = torch.tensor([src, dst], dtype=torch.long)

        y = torch.tensor(g['xt'], dtype=torch.float).unsqueeze(1)

        data = Data(x=node_features, edge_index=edge_index, y=y)
        pyg_graphs.append(data)

    return pyg_graphs


if __name__ == "__main__":

    t = 1
    with open("./data/heat/toy/graphs_t1.pkl", "rb") as f:
        graphs_t1 = pickle.load(f)

    pyg_graphs_t1 = graphs_to_pyg_data(graphs_t1)
    # TODO: Need to check that this works 
