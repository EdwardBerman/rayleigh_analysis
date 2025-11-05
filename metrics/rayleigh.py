import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import degree

def rayleigh_error(f: nn.Module, X: Data) -> torch.Tensor:
    """ 
        Computes |R_G(f(X; A)) - R_G(X; A)| 
    X' = f(X; A), so this is equivalent to |Tr(X'^TAX')/||X'||_F - Tr(X^TAX)/||X||_F|

    A^~ = D^(-1/2)AD^(-1/2)
    """
    X_prime = f(X)

    values = X.x
            
    edge_index = X.edge_index.to(values.device).long()
    src, dst = edge_index[0], edge_index[1]
    N = values.shape[0]
    deg_in = degree(dst, num_nodes=N, dtype=values.dtype).clamp(min=1.0)
    inv_sqrt_deg = deg_in.rsqrt().view(N, 1)

    def norm_sqrt_deg(x: torch.Tensor) -> torch.Tensor:
        return x * inv_sqrt_deg
                    
    X_norm       = norm_sqrt_deg(X.x)       
    X_prime_norm = norm_sqrt_deg(X_prime)  
    diff_X = X_norm[src, 0] - X_norm[dst, 0]         
    diff_X_prime = X_prime_norm[src, 0] - X_prime_norm[dst, 0]

    x_numerator = (diff_X.pow(2).sum(dim=-1)).mean()
    x_prime_numerator = (diff_X_prime.pow(2).sum(dim=-1)).mean()

    X_denom       = X.x.pow(2).sum()       # ||X||_F^2
    X_prime_denom = X_prime.pow(2).sum() # ||X'||_F^2

    rayleigh_X = x_numerator*0.5/(X_denom+1e-16)
    rayleigh_X_prime = x_prime_numerator*0.5/(X_prime_denom+1e-16)

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



