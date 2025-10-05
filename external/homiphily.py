"""Sourced from https://github.com/CUAI/Non-Homophily-Large-Scale/blob/master/homophily.py"""

import torch
from torch_geometric.utils import remove_self_loops

from external.torch_scatter import scatter_add


def compat_matrix_edge_idx(edge_idx, labels):
    """
     c x c compatibility matrix, where c is number of classes
     H[i,j] is proportion of endpoints that are class j 
     of edges incident to class i nodes 
     "Generalizing GNNs Beyond Homophily"
     treats negative labels as unlabeled
     """
    edge_index = remove_self_loops(edge_idx)[0]
    src_node, targ_node = edge_index[0, :], edge_index[1, :]
    labeled_nodes = (labels[src_node] >= 0) * (labels[targ_node] >= 0)
    label = labels.squeeze()
    c = label.max()+1
    H = torch.zeros((c, c)).to(edge_index.device)
    src_label = label[src_node[labeled_nodes]]
    targ_label = label[targ_node[labeled_nodes]]
    label_idx = torch.cat(
        (src_label.unsqueeze(0), targ_label.unsqueeze(0)), axis=0)
    for k in range(c):
        sum_idx = torch.where(src_label == k)[0]
        add_idx = targ_label[sum_idx]
        scatter_add(torch.ones_like(add_idx).to(
            H.dtype), add_idx, out=H[k, :], dim=-1)
    H = H / torch.sum(H, axis=1, keepdims=True)
    return H

# ADDING THE FOLLOWING HELPER BASED ON OUR NEED TO PROCESS MANY GRAPHS


def padded_counts_and_props(label):
    """
    Returns:
      counts_full: int64 tensor of shape [c], where c = max(label)+1,
                   with zeros for any missing class indices.
      proportions_full: float32 tensor of shape [c], same alignment, sums to 1 over labeled entries.
      labeled_total: int (number of labeled items).
    """
    label = label.view(-1).long()
    mask = label >= 0  # keep labeled only

    if not mask.any():
        # no labeled items
        return (torch.zeros(0, dtype=torch.long, device=label.device),
                torch.zeros(0, dtype=torch.float32, device=label.device),
                0)

    labeled = label[mask]
    present, counts = torch.unique(
        labeled, return_counts=True)  # sorted by value
    c = int(labeled.max().item()) + 1

    counts_full = torch.zeros(c, dtype=counts.dtype, device=label.device)
    counts_full[present] = counts

    labeled_total = int(mask.sum().item())
    proportions_full = counts_full.to(torch.float32) / labeled_total

    return counts_full, proportions_full, labeled_total


def our_measure(edge_index, label):
    # TODO: Fix this for classes that don't appear in the label
    """ 
    our measure \hat{h}
    treats negative labels as unlabeled 
    """
    label = label.squeeze()
    c = label.max()+1
    H = compat_matrix_edge_idx(edge_index, label)
    nonzero_label = label[label >= 0]
    # counts = nonzero_label.unique(return_counts=True)[1]
    # proportions = counts.float() / nonzero_label.shape[0]
    counts, proportions, _ = padded_counts_and_props(label)
    val = 0
    for k in range(c):
        class_add = torch.clamp(H[k, k] - proportions[k], min=0)
        if not torch.isnan(class_add):
            # only add if not nan
            val += class_add
    val /= c-1
    return val
