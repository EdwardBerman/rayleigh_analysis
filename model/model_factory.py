import torch
import torch.nn as nn
from torch_geometric.nn import BatchNorm
from torch_geometric.nn.models import GCN, GAT, GraphSAGE
from model.edge_aggregator import EdgeModel, NodeModel

def add_skip_connections(model: nn.Module) -> nn.Module:
    class ResidualModel(nn.Module):
            def __init__(self, model):
                super(ResidualModel, self).__init__()
                self.model = model

            def forward(self, x, edge_index):
                x_res = x
                x = self.model(x, edge_index)
                x += x_res
                return x
    return ResidualModel(model)

def build_model(node_dim: int,
                model_type: str,
                num_layers: int,
                hidden_size: int,
                activation_function: nn.Module = nn.ReLU,
                skip_connections: bool,
                batch_size: int,
                batch_norm: str,
                num_attention_heads: int = 2,
                dropout_rate: float = 0.1,
                edge_aggregator: bool = False,
                edge_dim: int | None = None) -> nn.Module:

    # TODO: Check the models being built use all args in build_model (where applicable)

    if edge_aggregator and edge_dim is None:
        raise ValueError("edge_dim must be provided if edge_aggregator is True.")

    match case model_type:
        case 'GCN':
            model = GCN(num_layers=num_layers, 
                        in_channels=node_dim,
                        hidden_channels=hidden_size, 
                        out_channels=in_channels, 
                        dropout=dropout_rate, 
                        norm=batch_norm, 
                        batch_size=batch_size,
                        act=activation_function())
            model = add_skip_connections(model) if skip_connections else model
            return EdgeModel(edge_dim, node_dim, model) if edge_aggregator else NodeModel(model)
        case 'GAT':
            model = GAT(num_layers=num_layers, 
                        in_channels=node_dim,
                        hidden_channels=hidden_size, 
                        out_channels=in_channels, 
                        heads=num_attention_heads, 
                        dropout=dropout_rate, 
                        norm=batch_norm, 
                        batch_size=batch_size,
                        act=activation_function())
            model = add_skip_connections(model) if skip_connections else model
            return EdgeModel(edge_dim, node_dim, model) if edge_aggregator else NodeModel(model)
        case 'MPNN':
            model = nn.Module()  # Placeholder for actual MPNN implementation
            pass
        case 'Sage':
            model = GraphSAGE(num_layers=num_layers, 
                              in_channels=node_dim,
                              hidden_channels=hidden_size, 
                              out_channels=in_channels, 
                              dropout=dropout_rate, 
                              norm=batch_norm, 
                              batch_size=batch_size,
                              act=activation_function())
            model = add_skip_connections(model) if skip_connections else model
            return EdgeModel(edge_dim, node_dim, model) if edge_aggregator else NodeModel(model)
        case 'Uni':
            pass
        case 'CRAWL':
            pass
        case 'LINKX':
            pass
        case _:
            raise ValueError(f"Unsupported model type: {model_type}. Accepts 'GCN', 'GAT', 'MPNN', 'Sage', 'Uni', 'CRAWL', 'LINKX'.")
    
