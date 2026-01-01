import robust_laplacian
import torch
from torch import nn
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import remove_self_loops

from external.ortho_gcn import GroupSort, OrthogonalGCNConvLayer


class PadToDim(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, *args, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, x, *args, **kwargs):
        """Pads x to the desired dimension. Takes in whatever other arguments but ignores them to be compatible with PyG style models like the GCN."""
        padding_size = self.hidden_dim - self.input_dim
        return torch.nn.functional.pad(x, (0, padding_size), value=0)

class Sin(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.sin(x)

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int=3, activation: str="sin"):
        super().__init__()
        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        if activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "sin":
            act_fn = Sin()
        else:
            raise ValueError(f"Activation {activation} not recognized.")

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(act_fn)
        self.network = nn.Sequential(*layers)

    def forward(self, x, *args, **kwargs):
        return self.network(x)


def determine_layer(layer: str) -> nn.Module:
    """Determines the layer type for non-Uni layers"""
    match layer:
        case "gcn":
            return GCNConv
        case "gat":
            return GATConv
        case "pad":
            return PadToDim
        case "mlp":
            return MLP
        case _:
            raise ValueError(f"Layer type {layer} not recognized.")


class Uni(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        projection: str,
        num_encoder_layers: int,
        decoder: str,
        num_decoder_layers: int,
        null_isolated,
        add_self_loops,
        dropout,
        final_activation,
        T=10,
    ):
        """
        Unitary Convolution on meshes.

        projection : str
            Projection layer into the hidden_dim. 
            One of 'gcn', 'gat', or 'pad'.
        num_encoder_layers : int
            Number of layers in the encoder, includes the projection layer and the unitary convolution layers. 
        decoder : str
            Decoder to go from hidden_dim -> output_dim. 
            One of 'gcn', 'gat'. 
        num_decoder_layers : int
            Number of decoder layers. 
        T : int, optional
            # of terms in the Taylor series truncations, by default 10
        """

        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.projection = determine_layer(projection)
        self.num_encoder_layers = num_encoder_layers
        self.decoder = determine_layer(decoder)
        self.num_decoder_layers = num_decoder_layers

        self.null_isolated = null_isolated
        self.add_self_loops = add_self_loops
        self.dropout = dropout
        self.final_activation = final_activation
        self.T = T

        self.transforms = []
        self.blocks = nn.ModuleList()

        for i in range(self.num_encoder_layers):
            if i == 0:
                self.blocks.append(
                    self.projection(self.input_dim, self.hidden_dim,
                                    add_self_loops=add_self_loops)
                )
            else:
                self.blocks.append(
                    OrthogonalGCNConvLayer(self.hidden_dim,
                                           self.hidden_dim,
                                           dropout=dropout,
                                           residual=False,
                                           global_bias=False,
                                           T=T,
                                           use_hermitian=True,
                                           activation=GroupSort)

                )

        for i in range(self.num_decoder_layers):
            if i == self.num_decoder_layers - 1:
                self.blocks.append(
                    self.decoder(self.hidden_dim, self.output_dim,
                                 add_self_loops=add_self_loops)
                )
            else:
                self.blocks.append(self.decoder(
                    self.hidden_dim, self.hidden_dim, add_self_loops=add_self_loops))

    def forward(self, data):
        for transform in self.transforms:
            data = transform(data)

        x = data.x.squeeze(-1)

        if x.dim() == 1:
            x = x.unsqueeze(-1)

        # check if object has "rewired" attribute
        if not hasattr(data, 'rewired') or not data.rewired:

            with torch.no_grad():
                pos, face = data.pos.cpu(), data.face.cpu()
                L, _ = robust_laplacian.mesh_laplacian(
                    pos.cpu().numpy(), face.T.cpu().numpy())
                L = L.tocoo()

                row = torch.from_numpy(L.row).long().to(data.x.device)
                col = torch.from_numpy(L.col).long().to(data.x.device)
                val = torch.from_numpy(L.data).to(
                    data.x.device).to(data.x.dtype)
                mask = row != col
                row, col, val = row[mask], col[mask], val[mask]
                edge_index = torch.stack([row, col], dim=0).contiguous()
                edge_weight = (-val).contiguous()

                edge_index, edge_weight = remove_self_loops(
                    edge_index, edge_weight)

                data.rewired = True
                data.edge_index = edge_index
                data.edge_weight = edge_weight
        else:
            edge_index = data.edge_index
            edge_weight = data.edge_weight

        input_data_obj = data.clone()

        input_data_obj.x = x
        input_data_obj.edge_index = edge_index
        input_data_obj.edge_weight = edge_weight

        for i, block in enumerate(self.blocks):
            # projection layer or decoder layers
            if i == 0 or i > self.num_encoder_layers - 1:
                if isinstance(block, MLP):
                    x = block(x)
                else:
                    x = block(x, edge_index, edge_weight)
            # OrthogonalGCNConvLayer
            else:
                input_data_obj.x = x
                input_data_obj = block(input_data_obj)
                x = input_data_obj.x

        return x[:, :, None]
