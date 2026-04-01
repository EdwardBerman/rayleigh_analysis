import torch
from torch_geometric.utils import remove_isolated_nodes

from external.custom_hermes.nn.eman_res_net_block import EmanAttResNetBlock
from external.custom_hermes.transform.gem_precomp import GemPrecomp


class EMAN(torch.nn.Module):
    def __init__(
        self,
        block_dims,
        block_orders,
        reltan_features,
        null_isolated,
        n_rings,
        band_limit,
        num_samples,
        checkpoint,
        node_batch_size,
        equiv_bias,
        regular_non_lin,
        batch_norm,
        dropout,
        n_heads,
        final_activation,
        graph_level_readout=True,
        pooling='mean',
        readout_layers=2,
        readout_hidden_dim=64,
        readout_activation='sin',
        **kwargs
    ):
        super().__init__()

        if not reltan_features:
            assert kwargs == {}, "kwargs not empty but reltan_features=False"

        assert len(block_dims) >= 3, "minimum length of block_dims must be >= 3"
        assert len(block_orders) >= 3, "minimum length of block_orders must be >= 3"
        assert len(block_dims) == len(
            block_orders
        ), "length of block_dims and block_orders must be equal"
        self.block_dims = block_dims
        self.block_orders = block_orders
        self.out_dim = self.block_dims[-1]

        self.reltan_features = reltan_features
        self.null_isolated = null_isolated
        
        self.graph_level_readout = graph_level_readout
        self.pooling = pooling

        block_kwargs = dict(
            n_rings=n_rings,
            band_limit=band_limit,
            num_samples=num_samples,
            checkpoint=checkpoint,
            node_batch_size=node_batch_size,
            equiv_bias=equiv_bias,
            regular_non_lin=regular_non_lin,
            batch_norm=batch_norm,
            dropout=dropout,
            n_heads=n_heads,
        )

        self.transforms = [GemPrecomp(n_rings, band_limit)]

        self.layers = torch.nn.ModuleList()
        for i in range(len(self.block_dims) - 3):
            self.layers.append(
                EmanAttResNetBlock(
                    self.block_dims[i],
                    self.block_dims[i + 1],
                    self.block_dims[i + 2],
                    self.block_orders[i],
                    self.block_orders[i + 1],
                    self.block_orders[i + 2],
                    final_activation=True,
                    **block_kwargs,
                )
            )
        # Add final block
        self.layers.append(
            EmanAttResNetBlock(
                self.block_dims[-3],
                self.block_dims[-2],
                self.block_dims[-1],
                self.block_orders[-3],
                self.block_orders[-2],
                self.block_orders[-1],
                final_activation=final_activation,
                **block_kwargs,
            )
        )

        if self.graph_level_readout:
            # After message passing, node features have shape [num_nodes, out_dim, 1]
            # After pooling: [batch_size, out_dim]
            self._build_readout_mlp(
                in_dim=self.out_dim,
                hidden_dim=readout_hidden_dim,
                num_layers=readout_layers,
                activation=readout_activation,
            )

    def _build_readout_mlp(self, in_dim, hidden_dim, num_layers, activation='sin'):
        """Build MLP head for graph-level prediction."""
        # Choose activation function
        if activation == 'sin':
            act_fn = torch.sin
        elif activation == 'relu':
            act_fn = nn.ReLU()
        elif activation == 'silu':
            act_fn = nn.SiLU()
        elif activation == 'gelu':
            act_fn = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
    def forward(self, data):
        # transform adds precomp feature (cosines and sines with radial weights) to the data
        # rel_transform adds rel_tang_feat (check Sec. 4 in the draft) feature to data
        for transform in self.transforms:
            data = transform(data)

        edge_index, precomp_neigh_edge, precomp_self_edge, connection = (
            data.edge_index,
            data.precomp_neigh_edge,
            data.precomp_self_edge,
            data.connection,
        )

        # Input node features
        assert data.x.dim() == 3
        x = data.x

        # Setting the features of isolated nodes to 0
        if self.null_isolated:
            non_isol_mask = remove_isolated_nodes(edge_index)[-1]
            x[~non_isol_mask] = 0.0

        for layer in self.layers:
            x = layer(x, edge_index, precomp_neigh_edge, precomp_self_edge, connection)
        
        if self.graph_level_readout:
            # x has shape [num_nodes, out_dim, 1]
            # Take the trivial feature (order-0) for pooling
            x_scalar = x[:, :, 0]  # [num_nodes, out_dim]
            
            if self.pooling == 'mean':
                graph_emb = global_mean_pool(x_scalar, data.batch)  # [batch_size, out_dim]
            else:
                raise ValueError(f"Unsupported pooling method: {self.pooling}")
            out = self.readout_mlp(graph_emb)  # [batch_size, 1]
            return out
        else:
            return x
