import robust_laplacian
import torch
from torch import nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import remove_self_loops

from external.ortho_gcn import GroupSort, OrthogonalGCNConvLayer


class Uni(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        encoder: nn.Module | None,  # if None, will pad.
        null_isolated,
        add_self_loops,
        dropout,
        final_activation,
        T=10,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.null_isolated = null_isolated
        self.add_self_loops = add_self_loops
        self.dropout = dropout
        self.final_activation = final_activation

        self.transforms = []

        self.blocks = nn.ModuleList()

        for i in range(num_layers):
            if i == num_layers - 1:
                self.blocks.append(
                    GCNConv(self.hidden_dim, self.input_dim,
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

    def forward(self, data):
        for transform in self.transforms:
            data = transform(data)

        x = data.x.squeeze(-1)

        if x.dim() == 1:
            x = x.unsqueeze(-1)

        padding_size = self.hidden_dim - x.shape[-1]
        x = torch.nn.functional.pad(
            x, (0, padding_size), mode='constant', value=0)

        # check if object has "rewired" attribute
        if not hasattr(data, 'rewired') or not data.rewired:

            with torch.no_grad():
                pos, face = data.pos.cpu(), data.face.cpu()
                L, M = robust_laplacian.mesh_laplacian(
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
            if i == len(self.blocks) - 1:
                x = block(x, edge_index, edge_weight)
            else:  # OrthogonalGCNConvLayer
                input_data_obj.x = x
                input_data_obj = block(input_data_obj)
                x = input_data_obj.x

        return x[:, :, None]
