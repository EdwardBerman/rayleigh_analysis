import torch
import torch.nn as nn
from torch_geometric.nn import BatchNorm
from torch_geometric.nn.models import GAT, GCN, LINKX, GraphSAGE

from external.crawl.models import CRaWl
from external.unitary_gcn import GroupSort, UnitaryGCNConvLayer
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


class UniStack(nn.Module):
    def __init__(self, layers: list[nn.Module]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
        return x


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
        case "GeLU":
            return nn.GELU
        case _:
            raise ValueError(
                f"Unsupported activation function: {activation_name}. Accepts 'ReLU', 'LeakyReLU', 'Identity', 'GroupSort', 'GELU'.")


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
                edge_dim: int | None = None,
                truncation_level: int = 10,
                **kwargs) -> nn.Module:

    # TODO: Check the models being built use all args in build_model (where applicable)

    activation_function = str_to_activation(activation_function)

    if edge_aggregator and edge_dim is None:
        raise ValueError(
            "edge_dim must be provided if edge_aggregator is True.")
    if edge_aggregator is not None:
        print("Using Edge Aggregator")

    match model_type:
        case 'GCN':
            model = GCN(num_layers=num_layers,
                        in_channels=node_dim,
                        hidden_channels=hidden_size,
                        out_channels=node_dim,
                        dropout=dropout_rate,
                        norm=batch_norm,
                        act=activation_function()) if batch_norm != "None" else GCN(num_layers=num_layers,
                                                                                    in_channels=node_dim,
                                                                                    hidden_channels=hidden_size,
                                                                                    out_channels=node_dim,
                                                                                    dropout=dropout_rate,
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
                        act=activation_function()) if batch_norm != "None" else GAT(num_layers=num_layers,
                                                                                    in_channels=node_dim,
                                                                                    hidden_channels=hidden_size,
                                                                                    out_channels=node_dim,
                                                                                    heads=num_attention_heads,
                                                                                    dropout=dropout_rate,
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
                              act=activation_function()) if batch_norm != "None" else GraphSAGE(num_layers=num_layers,
                                                                                                in_channels=node_dim,
                                                                                                hidden_channels=hidden_size,
                                                                                                out_channels=node_dim,
                                                                                                dropout=dropout_rate,
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
                                                       dropout=dropout_rate,
                                                       residual=skip_connections,
                                                       global_bias=True,
                                                       T=10,
                                                       use_hermitian=False,
                                                       activation=activation_function()))
            model = UniStack(module_list)
            return EdgeModel(edge_dim, node_dim, model, edge_aggregator) if edge_aggregator is not None else NodeModel(model)
        case 'LieUni':
            module_list = []
            input_dim = node_dim
            output_dim = node_dim
            if input_dim != output_dim:
                print(
                    f"Warning: For Lie Unitary GCN, input and output dimensions must be the same, but a distinct output size was set. \nSetting output dim {output_dim} to be input dim {input_dim}\nDid you mean Separable Unitary Convolution?")
            if input_dim != hidden_size:
                print(
                    f"Warning: For Lie Unitary GCN, input and hidden dimensions must be the same, but a distinct hidden size was set. \nSetting hidden dim {hidden_size} to be input dim {input_dim}\nDid you mean Separable Unitary Convolution?")
            for layer in range(num_layers):
                module_list.append(UnitaryGCNConvLayer(input_dim,
                                                       input_dim,
                                                       dropout=dropout_rate,
                                                       residual=False,
                                                       global_bias=False,
                                                       T=truncation_level,
                                                       use_hermitian=True,
                                                       activation=activation_function()))
            model = UniStack(module_list)
            return EdgeModel(edge_dim, node_dim, model, edge_aggregator) if edge_aggregator is not None else NodeModel(model)
        case 'CRAWL':
            assert not skip_connections, "Skip connections should be False for CRaWl, which already includes skip connections."
            assert edge_aggregator == None, "Edge aggregator should be None for CRaWl, which already includes an edge aggregator."
            model = CRaWl(node_feat_dim=node_dim,
                          edge_feat_dim=edge_dim,
                          layers=num_layers,
                          hidden=hidden_size,
                          kernel_size=receptive_field,
                          dropout=dropout_rate,
                          steps=kwargs.get('steps', 50),
                          win_size=window_size,
                          train_start_ratio=kwargs.get(
                              'train_start_ratio', 1.0),
                          compute_id_feat=kwargs.get('compute_id_feat', True),
                          compute_adj_feat=kwargs.get(
                              'compute_adj_feat', True),
                          walk_delta=kwargs.get('walk_delta', 0.0),
                          node_feat_enc=kwargs.get('node_feat_enc', None),
                          edge_feat_enc=kwargs.get('edge_feat_enc', None))
            return model
        case _:
            raise ValueError(
                f"Unsupported model type: {model_type}. Accepts 'GCN', 'GAT', 'MPNN', 'Sage', 'Uni', 'CRAWL'.")
