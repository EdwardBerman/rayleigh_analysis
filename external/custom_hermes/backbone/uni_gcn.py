from torch import nn
from torch_geometric.utils import remove_isolated_nodes

import torch
from torch_geometric.data import Data

from external.ortho_gcn import OrthogonalGCNConvLayer


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
            self.blocks.append(
                    OrthogonalGCNConvLayer(1,
                                           1,
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

        # Setting the features of isolated nodes to 0
        if self.null_isolated:
            non_isol_mask = remove_isolated_nodes(data.edge_index)[-1]
            x[~non_isol_mask] = 0.0
        
        x = data.x[:, 3:4].clone()

        modified_data = Data(
            x=x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr if hasattr(data, 'edge_attr') else None,
            batch=data.batch if hasattr(data, 'batch') else None
        )

        for block in self.blocks:
            modified_data = block(modified_data)
            x = modified_data.x

        return x[:, :, None]
