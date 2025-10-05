import torch
import torch.nn as nn
from torch_geometric.nn import BatchNorm
from torch_geometric.nn.models import GCN, GAT, GraphSAGE, LINKX
from model.edge_aggregator import EdgeModel, NodeModel
from external.unitary_gcn import UnitaryGCNConvLayer, GroupSort, ComplexActivation

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

def str_to_activation(activation_name: str) -> nn.Module:
    match activation_name:
        case 'ReLU':
            return nn.ReLU
        case 'LeakyReLU':
            return nn.LeakyReLU
        case 'Identity':
            return nn.Identity
        case 'GroupSort':
            return GroupSort
        case 'ComplexReLU':
            return ComplexActivation(nn.ReLU)
        case _:
            raise ValueError(f"Unsupported activation function: {activation_name}. Accepts 'ReLU', 'LeakyReLU', 'Identity', 'GroupSort', 'ComplexReLU'.")

def build_model(node_dim: int,
                model_type: str,
                num_layers: int,
                hidden_size: int,
                activation_function: str,
                skip_connections: bool,
                batch_norm: str,
                num_attention_heads: int = 2,
                window_size: int = 4,
                receptive_field: int = 5,
                dropout_rate: float = 0.1,
                edge_aggregator: str | None = None,
                edge_dim: int | None = None) -> nn.Module:

    # TODO: Check the models being built use all args in build_model (where applicable)

    activation_function = str_to_activation(activation_function)

    if edge_aggregator and edge_dim is None:
        raise ValueError("edge_dim must be provided if edge_aggregator is True.")

    match model_type:
        case 'GCN':
            model = GCN(num_layers=num_layers, 
                        in_channels=node_dim,
                        hidden_channels=hidden_size, 
                        out_channels=node_dim,
                        dropout=dropout_rate, 
                        norm=batch_norm, 
                        act=activation_function())
            model = add_skip_connections(model) if skip_connections else model
            return EdgeModel(edge_dim, node_dim, model, edge_aggregator) if edge_aggregator is not None else NodeModel(model)
        case 'GAT':
            model = GAT(num_layers=num_layers, 
                        in_channels=node_dim,
                        hidden_channels=hidden_size, 
                        out_channels=node_dim,
                        heads=num_attention_heads, 
                        dropout=dropout_rate, 
                        norm=batch_norm, 
                        act=activation_function())
            model = add_skip_connections(model) if skip_connections else model
            return EdgeModel(edge_dim, node_dim, model, edge_aggregator) if edge_aggregator is not None else NodeModel(model)
        case 'MPNN':
            model = nn.Module()  # Placeholder for actual MPNN implementation
            pass
        case 'Sage':
            model = GraphSAGE(num_layers=num_layers, 
                              in_channels=node_dim,
                              hidden_channels=hidden_size, 
                              out_channels=node_dim,
                              dropout=dropout_rate, 
                              norm=batch_norm, 
                              act=activation_function())
            model = add_skip_connections(model) if skip_connections else model
            return EdgeModel(edge_dim, node_dim, model, edge_aggregator) if edge_aggregator is not None else NodeModel(model)
        case 'Uni':
            module_list = []
            for layer in range(num_layers):
                input_dim = node_dim if layer == 0 else hidden_size
                output_dim = node_dim if layer == num_layers - 1 else hidden_size
                module_list.append(UnitaryGCNConvLayer(input_dim,
                                                       output_dim, 
                                                       dropout  =  dropout_rate, 	
                                                       residual  = skip_connections, 	
                                                       global_bias  =  True, 		
                                                       T  =  10, 				
                                                       use_hermitian  =  False, 		
                                                       activation  =  activation_function()))
            model = nn.Sequential(*module_list)
            return EdgeModel(edge_dim, node_dim, model, edge_aggregator) if edge_aggregator is not None else NodeModel(model)
        case 'CRAWL':
            pass
        case _:
            raise ValueError(f"Unsupported model type: {model_type}. Accepts 'GCN', 'GAT', 'MPNN', 'Sage', 'Uni', 'CRAWL'.")
    
