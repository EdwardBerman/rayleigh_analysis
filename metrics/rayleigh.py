import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, to_torch_coo_tensor


# TODO: Make these more mem efficient
def rayleigh_error(f: nn.Module, X: Data) -> torch.Tensor:
    """ 
        Computes |R_G(f(X; A)) - R_G(X; A)| 
    X' = f(X; A), so this is equivalent to |Tr(X'^TAX')/||X'||_F - Tr(X^TAX)/||X||_F|

    A^~ = D^(-1/2)AD^(-1/2)
    """
    X_prime = f(X)
    edge_indices = X.edge_index

    X = X.x
    num_nodes = X.size(0)
    A_sparse = to_torch_coo_tensor(edge_indices, size=(num_nodes, num_nodes))
    A = A_sparse.to_dense()

    D = torch.diag(A.sum(dim=1)**(-0.5))
    I = torch.eye(A.size(0), device=A.device)
    A_tilde = D @ (I - A) @ D
    rayleigh_X = torch.trace(X.T @ A_tilde @ X) / torch.norm(X, p='fro')
    rayleigh_X_prime = torch.trace(
        X_prime.T @ A_tilde @ X_prime) / torch.norm(X_prime, p='fro')

    return torch.abs(rayleigh_X - rayleigh_X_prime)


def integrated_rayleigh_error(f: nn.Module, X: Data) -> torch.Tensor:
    """
        Sums the Rayleigh errors across all layers of the model f.
    """
    total_error = 0.0
    for layer in f.children():
        if isinstance(layer, nn.Module):
            total_error += rayleigh_error(layer, X)
            X = layer(X)
    return total_error
