from torch import nn
from torch_geometric.utils import remove_isolated_nodes
from torch_geometric.utils import remove_self_loops

import torch

from external.ortho_gcn import OrthogonalGCNConvLayer
from external.torch_scatter import scatter

from torch_geometric.nn import GCNConv

import robust_laplacian

class Uni(nn.Module):
    def __init__(
        self,
        null_isolated,
        add_self_loops,
        dropout,
        final_activation,
    ):
        super().__init__()


        self.null_isolated = null_isolated
        self.add_self_loops = add_self_loops
        self.dropout = dropout
        self.final_activation = final_activation


        block_kwargs = dict(
            add_self_loops=add_self_loops, dropout=dropout
        )

        self.transforms = []

        self.blocks = nn.ModuleList()

        for i in range(12):
            if i == 0:
                self.blocks.append(
                        GCNConv(5, 64, add_self_loops=add_self_loops)
                )
            elif i == 11:
                self.blocks.append(
                        GCNConv(64, 1, add_self_loops=add_self_loops)
                )
            else:
                self.blocks.append(
                        OrthogonalGCNConvLayer(64,
                                               64,
                                               dropout =  dropout,
                                               residual  =  False,
                                               global_bias  =  False,
                                               T  =  10,
                                               use_hermitian  =  True,
                                               activation  =  torch.nn.Identity)
                
                )



    def forward(self, data):
        for transform in self.transforms:
            data = transform(data)


        x = data.x.squeeze(-1)

        with torch.no_grad():
            pos, face = data.pos.cpu(), data.face.cpu()
            L, M = robust_laplacian.mesh_laplacian(
                pos.cpu().numpy(), face.T.cpu().numpy())
            L = L.tocoo()

            row = torch.from_numpy(L.row).long().to(data.x.device)
            col = torch.from_numpy(L.col).long().to(data.x.device)
            val = torch.from_numpy(L.data).to(data.x.device).to(data.x.dtype)
            mask = row != col
            row, col, val = row[mask], col[mask], val[mask]
            edge_index = torch.stack([row, col], dim=0).contiguous()
            edge_weight = (-val).contiguous()

            edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

            #L_offdiag = L_torch.clone()
            #L_offdiag.fill_diagonal_(0)
            #A_M = L_offdiag

            #weighted_edge_index = A_M.nonzero(
               # as_tuple=False).t().long().to(x.device)
            #edge_weights = A_M[weighted_edge_index[0], weighted_edge_index[1]].to(
                #x.device).to(x.dtype)

            #N_x = data.x.size(0)
            #N_pos = data.pos.size(0)

            #row, col = weighted_edge_index[0], weighted_edge_index[1]
            #deg = scatter(edge_weights, col, dim=0, dim_size=data.num_nodes, reduce='sum')

            #print("deg finite:", torch.isfinite(deg).all().item())
            #print("deg min/max:", deg.min().item(), deg.max().item())
            #print("deg == 0:", (deg == 0).sum().item(), "/", deg.numel())
            #print("deg < 0:", (deg < 0).sum().item())

            #deg_inv_sqrt = deg.pow(-0.5)
            #print("deg_inv_sqrt finite:", torch.isfinite(deg_inv_sqrt).all().item())
            #print("deg_inv_sqrt inf:", torch.isinf(deg_inv_sqrt).sum().item())
            #print("deg_inv_sqrt nan:", torch.isnan(deg_inv_sqrt).sum().item())

            #breakpoint()

        input_data_obj = data.clone()

        input_data_obj.x = x
        input_data_obj.edge_index = edge_index
        input_data_obj.edge_weight = edge_weight
    
        for i, block in enumerate(self.blocks):
            if i == 0:  # Last layer is GCNConv
                x = block(x, edge_index, edge_weight)
            elif i == 11:
                x = block(x, edge_index, edge_weight)
            else:  # OrthogonalGCNConvLayer
                input_data_obj.x = x
                input_data_obj = block(input_data_obj)
                x = input_data_obj.x

        return x[:, :, None]
