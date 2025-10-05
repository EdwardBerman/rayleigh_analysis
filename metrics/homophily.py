import torch
from torch_geometric.data import Data

from external.homophily import our_measure


def homiphily_classification(graph: Data) -> float:
    """
    Calculate the homophily ratio for a graph based on node labels.

    Args:
        graph (Data): A PyTorch Geometric Data object containing node features and labels.

    Returns:
        float: The homophily ratio, defined as the fraction of edges connecting nodes with the same label.
    """
    edge_index = graph.edge_index
    labels = graph.y
    return our_measure(edge_index, labels)


def homophily_regression(graph: Data) -> float:
    """
    Calculate the R^2 correlation between nodes connected by an edge 
    """
    edge_index = graph.edge_index
    labels = graph.y

    if labels.dim() == 1:
        labels = labels.view(-1, 1)

    src_labels = labels[edge_index[0]]
    dst_labels = labels[edge_index[1]]

    correlation_matrix = torch.corrcoef(
        torch.cat([src_labels, dst_labels], dim=1).T)
    r_squared = correlation_matrix[0, 1] ** 2

    return r_squared.item()
