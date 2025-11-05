import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, ResGatedGraphConv


class EdgeAggregatorGINE(torch.nn.Module):
    """
    The EdgeAggregatorGine class uses a GINEConv layer to aggregate edge features into node features.
    This is done so that we can run a GNN on a graph with edge features as a baseline against models that use edge features.
    """

    def __init__(self, edge_dim, node_dim):
        super(EdgeAggregatorGINE, self).__init__()
        update_mlp = nn.Sequential(
            torch.nn.Linear(node_dim, node_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(node_dim, node_dim)
        )
        self.conv = GINEConv(nn=update_mlp, edge_dim=edge_dim)

    def forward(self, x, edge_index, edge_attr):
        return self.conv(x, edge_index, edge_attr)


class EdgeAggregatorGATED(torch.nn.Module):
    """
    The EdgeAggregatorGated class uses a ResGatedGraphConv layer to aggregate edge features into node features.
    This is done so that we can run a GNN on a graph with edge features as a baseline against models that use edge features.
    """

    def __init__(self, edge_dim, node_dim):
        super(EdgeAggregatorGATED, self).__init__()
        self.conv = ResGatedGraphConv(
            in_channels=node_dim, out_channels=node_dim, edge_dim=edge_dim)

    def forward(self, x, edge_index, edge_attr):
        return self.conv(x, edge_index, edge_attr)


class EdgeModel(torch.nn.Module):
    def __init__(self, edge_dim, node_dim, base_model, aggregator_type):
        super(EdgeModel, self).__init__()
        match aggregator_type:
            case "GINE":
                self.edge_aggregator = EdgeAggregatorGINE(edge_dim, node_dim)
            case "GATED":
                self.edge_aggregator = EdgeAggregatorGATED(edge_dim, node_dim)
            case _:
                raise ValueError(f"Unknown aggregator type: {aggregator_type}")

        self.base_model = base_model

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = x.float()
        edge_attr = edge_attr.float()
        x = self.edge_aggregator(x, edge_index, edge_attr)
        return self.base_model(x, edge_index)


class NodeModel(torch.nn.Module):
    def __init__(self, base_model):
        super(NodeModel, self).__init__()
        self.base_model = base_model

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        return self.base_model(x, edge_index)
