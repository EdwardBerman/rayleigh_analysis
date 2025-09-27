import torch
from torch_geometric.nn import GINEConv

class EdgeAggregator(torch.nn.Module):
    """
    The EdgeAggregator class uses a GINEConv layer to aggregate edge features into node features.
    This is done so that we can run a GNN on a graph with edge features as a baseline against models that use edge features.
    """
    def __init__(self, edge_dim, node_dim):
        super(EdgeAggregator, self).__init__()
        edge_projector = torch.nn.Linear(edge_dim, node_dim)
        self.conv = GINEConv(nn=edge_projector, edge_dim=edge_dim)

    def forward(self, x, edge_index, edge_attr):
        return self.conv(x, edge_index, edge_attr)
