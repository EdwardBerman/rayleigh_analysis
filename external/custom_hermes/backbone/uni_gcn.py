from torch import nn
from torch_geometric.utils import remove_isolated_nodes

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

        for i in range(len(self.block_dims)):
            self.blocks.append(
                    uni_layer = OrthogonalGCNConvLayer(1,
                                                       1, 
                                                       dropout =  dropout,
                                                       residual  =  False, 		
                                                       global_bias  =  False,
                                                       T  =  10,
                                                       use_hermitian  =  True,
                                                       activation  =  torch.nn.Identity())
            )


    def forward(self, data):
        for transform in self.transforms:
            data = transform(data)

        x = data.x.squeeze(-1)

        # Setting the features of isolated nodes to 0
        if self.null_isolated:
            non_isol_mask = remove_isolated_nodes(data.edge_index)[-1]
            x[~non_isol_mask] = 0.0

        for block in self.blocks:
            x = block(x, data.edge_index)

        return x[:, :, None]
