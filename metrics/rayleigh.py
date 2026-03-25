import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.utils import degree, k_hop_subgraph


def rayleigh_error(f: nn.Module, X: Data) -> torch.Tensor:
    """ 
        Computes |R_G(f(X; A)) - R_G(X; A)| 
    X' = f(X; A), so this is equivalent to |Tr(X'^TAX')/||X'||_F - Tr(X^TAX)/||X||_F|

    A^~ = D^(-1/2)AD^(-1/2)
    """
    f.eval()
    X_prime = f(X)

    values = X.x

    edge_index = X.edge_index.to(values.device).long()
    src, dst = edge_index[0], edge_index[1]
    N = values.shape[0]

    dtype = values.real.dtype if torch.is_complex(values) else values.dtype
    deg = degree(dst, num_nodes=N, dtype=dtype)

    deg_in = deg.clamp(min=1.0)
    inv_sqrt_deg = deg_in.rsqrt().view(N, 1)

    def norm_sqrt_deg(x: torch.Tensor) -> torch.Tensor:
        return x * inv_sqrt_deg

    X_norm = norm_sqrt_deg(X.x)
    X_prime_norm = norm_sqrt_deg(X_prime)
    diff_X = X_norm[src, 0] - X_norm[dst, 0]
    diff_X_prime = X_prime_norm[src, 0] - X_prime_norm[dst, 0]

    x_numerator = (diff_X.abs().pow(2).sum(dim=-1)).sum()
    x_prime_numerator = (diff_X_prime.abs().pow(2).sum(dim=-1)).sum()

    X_denom = X.x.abs().pow(2).sum()       # ||X||_F^2
    X_prime_denom = X_prime.abs().pow(2).sum()  # ||X'||_F^2

    rayleigh_X = x_numerator*0.5/(X_denom+1e-16)
    rayleigh_X_prime = x_prime_numerator*0.5/(X_prime_denom+1e-16)

    return torch.abs(rayleigh_X - rayleigh_X_prime)


def rayleigh_quotients(f: nn.Module, batch: Data) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ 
    Computes three Rayleigh quotients, for batch.X, f(X) and batch.Y
    """
    batchX = compute_rayleigh_quotient(batch.x, batch.edge_index)
    fX = compute_rayleigh_quotient(f(batch), batch.edge_index)
    batchY = compute_rayleigh_quotient(batch.y, batch.edge_index)
    return batchX, fX, batchY


def rayleigh_quotients_graphlevel(f: nn.Module, batch: Data) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes Rayleigh quotients for batch.X and f(X)
    """
    batchX = compute_rayleigh_quotient(batch.x, batch.edge_index)
    fX = compute_rayleigh_quotient(f.base_model(
        batch.x, batch.edge_index), batch.edge_index)
    return batchX, fX


def compute_rayleigh_quotient(node_features, edge_index):
    """Computes the Rayleigh quotient for one graph given node features and edge index."""
    edge_index = edge_index.long()
    src, dst = edge_index[0], edge_index[1]
    N = node_features.shape[0]

    dtype = node_features.real.dtype if torch.is_complex(
        node_features) else node_features.dtype
    deg = degree(dst, num_nodes=N, dtype=dtype)
    deg_in = deg.clamp(min=1.0)
    inv_sqrt_deg = deg_in.rsqrt().view(N, 1)

    def norm_sqrt_deg(x: torch.Tensor) -> torch.Tensor:
        return x * inv_sqrt_deg

    x_norm = norm_sqrt_deg(node_features)
    diff_X = x_norm[src, 0] - x_norm[dst, 0]
    x_numerator = (diff_X.abs().pow(2).sum(dim=-1)).sum()
    X_denom = node_features.abs().pow(2).sum()
    rayleigh = x_numerator*0.5/(X_denom+1e-16)

    return rayleigh


def compute_localized_rayleigh_quotient(node_features, edge_index, num_hops=2, samples=100):

    rqs = []

    for _ in range(samples):

        node_idx = torch.randint(0, node_features.size(0), (1,)).item()

        subset, sub_edge_index, _, _ = k_hop_subgraph(
            node_idx=node_idx,
            num_hops=num_hops,
            edge_index=edge_index,
            relabel_nodes=True,
        )

        sub_node_features = node_features[subset]
        rqs.append(compute_rayleigh_quotient(
            sub_node_features, sub_edge_index))

    return rqs
