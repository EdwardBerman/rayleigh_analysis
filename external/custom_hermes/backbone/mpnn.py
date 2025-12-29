from torch import nn
from torch_geometric.utils import remove_isolated_nodes

from external.custom_hermes.nn.mpnn_layer import MPNNLayer


class MPNN(nn.Module):
    def __init__(
        self,
        message_dims,
        update_dims,
        edge_dims,
        null_isolated,
        message_norm,
        update_norm,
        message_dropout,
        update_dropout,
        final_activation,
    ):
        super().__init__()

        assert len(message_dims) == len(
            update_dims
        ), "message_dims and update_dims should have same number of blocks"

        self.message_dims = message_dims
        self.update_dims = update_dims
        self.edge_dims = edge_dims

        print("MPNN message dims:", self.message_dims)
        print("MPNN update dims:", self.update_dims)
        print("MPNN edge dims:", self.edge_dims)

        self.out_dim = self.update_dims[-1][-1]

        self.null_isolated = null_isolated

        block_kwargs = dict(
            edge_dims=edge_dims,
            message_norm=message_norm,
            update_norm=update_norm,
            message_dropout=message_dropout,
            update_dropout=update_dropout,
        )

        self.transforms = []

        # Construct blocks
        self.blocks = nn.ModuleList()
        for i in range(len(message_dims) - 1):
            self.blocks.append(
                MPNNLayer(
                    self.message_dims[i],
                    self.update_dims[i],
                    final_activation=True,
                    **block_kwargs,
                )
            )
        # Add final block
        self.blocks.append(
            MPNNLayer(
                self.message_dims[-1],
                self.update_dims[-1],
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
            x = block(x, data.edge_index, data.edge_attr)

        return x[:, :, None]
