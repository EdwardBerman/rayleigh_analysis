import torch
from torch_geometric.utils import degree
from torch_geometric.data import Data
import torch.nn.functional as F

import robust_laplacian

def evaluate_rayleigh_loss(y_true: torch.Tensor, y_pred: torch.Tensor, edge_index: torch.Tensor) -> float:
    src, dst = edge_index[0], edge_index[1]
    N = y_true.size(0)
    deg_in = degree(dst, num_nodes=N, dtype=y_true.dtype).clamp(min=1.0)
    inv_sqrt_deg = deg_in.rsqrt().view(N, 1)

    def norm_sqrt_deg(x: torch.Tensor) -> torch.Tensor:
        return x * inv_sqrt_deg

    y_norm      = norm_sqrt_deg(y_true)
    y_pred_norm = norm_sqrt_deg(y_pred)  
    diff_true = y_norm[src, 0] - y_norm[dst, 0]         
    diff_pred = y_pred_norm[src, 0] - y_pred_norm[dst, 0]
    edge_mse_true = (diff_true ** 2).sum()
    edge_mse_pred = (diff_pred ** 2).sum()

    sum_nodes_sq_gt = y_true.pow(2).sum()
    sum_nodes_sq_pred = y_pred.pow(2).sum()

    return (edge_mse_true.item()*0.5/(sum_nodes_sq_gt.item()+1e-16) - edge_mse_pred.item()*0.5/(sum_nodes_sq_pred.item()+1e-16))**2 + F.mse_loss(y_true, y_pred) * 0.01

def make_rayleigh_loss():
    """
    Factory that returns a callable loss(y_true, y_pred, edge_index).
    Hydra can instantiate this (calls make_rayleigh_loss(...)).
    """
    def loss(y_true, y_pred, edge_index):
        return evaluate_rayleigh_loss(y_true, y_pred, edge_index)
    return loss
