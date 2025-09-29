import torch
from torch_geometric.data import Data
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

class EdgeModel(torch.nn.Module):
    def __init__(self, edge_dim, node_dim, base_model):
        super(FullModel, self).__init__()
        self.edge_aggregator = EdgeAggregator(edge_dim, node_dim)
        self.base_model = base_model

    def forward(self, data: Data) -> torch.Tensor):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.edge_aggregator(x, edge_index, edge_attr)
        return self.base_model(x, edge_index)
