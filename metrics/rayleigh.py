import torch
import torch.nn as nn

def rayleigh_error(f: nn.module, 
                   X: torch.Tensor, 
                   A: torch.Tensor, 
                   edge_indices: torch.Tensor | None = None,
                   edge_features: torch.Tensor | None = None) -> torch.Tensor:
    """ 
        Computes |R_G(f(X; A)) - R_G(X; A)| 
    X' = f(X; A), so this is equivalent to |Tr(X'^TAX')/||X'||_F - Tr(X^TAX)/||X||_F|

    A^~ = D^(-1/2)AD^(-1/2)
    """

    X_prime = f(X, edge_indices, edge_features)
    D = torch.diag(A.sum(dim=1)**(-0.5))
    I = torch.eye(A.size(0), device=A.device)
    A_tilde = D @ (I - A) @ D
    rayleigh_X = torch.trace(X.T @ A_tilde @ X) / torch.norm(X, p='fro')
    rayleigh_X_prime = torch.trace(X_prime.T @ A_tilde @ X_prime) / torch.norm(X_prime, p='fro')

    return torch.abs(rayleigh_X - rayleigh_X_prime)

def integrated_rayleigh_error(f: nn.module, 
                              X: torch.Tensor, 
                              A: torch.Tensor, 
                              edge_indices: torch.Tensor | None = None,
                              edge_features: torch.Tensor | None = None) -> torch.Tensor:
    """
        Sums the Rayleigh errors across all layers of the model f.
    """
    total_error = 0.0
    for layer in f.children():
        if isinstance(layer, nn.Module):
            total_error += rayleigh_error(layer, X, A, edge_indices, edge_features)
            X = layer(X, edge_indices, edge_features)
    return total_error
