from torch import nn
from torch_geometric.utils import remove_isolated_nodes

from external.custom_hermes.nn.gcn_res_block import GCNResBlock

class GCN(nn.Module):
    def __init__(
        self,
        block_dims,
        null_isolated,
        add_self_loops,
        batch_norm,
        dropout,
        final_activation,
    ):
        super().__init__()

        assert len(block_dims) >= 2, "minimum length of block_dims must be >= 2"

        self.block_dims = block_dims
        self.out_dim = self.block_dims[-1]

        self.null_isolated = null_isolated
        self.add_self_loops = add_self_loops
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.final_activation = final_activation

        # Add vertex_normals to input
        self.block_dims[0] += 3

        block_kwargs = dict(
            add_self_loops=add_self_loops, batch_norm=batch_norm, dropout=dropout
        )

        self.transforms = []

        self.blocks = nn.ModuleList()

        for i in range(len(self.block_dims) - 3):
            self.blocks.append(
                GCNResBlock(
                    self.block_dims[i],
                    self.block_dims[i + 1],
                    self.block_dims[i + 2],
                    final_activation=True,
                    **block_kwargs,
                )
            )

        # Add final block
        self.blocks.append(
            GCNResBlock(
                self.block_dims[-3],
                self.block_dims[-2],
                self.block_dims[-1],
                final_activation=final_activation,
                **block_kwargs,
            )
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
