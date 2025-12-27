from se3_transformer_pytorch import SE3Transformer as BaseSE3Transformer
from torch_geometric.utils import to_dense_adj


class SE3Transformer(BaseSE3Transformer):
    def forward(self, data):
        # Shape into [b, n, d, 1]
        x = data.x.unsqueeze(0).squeeze(-1)

        # Shape into [b, n, d]
        pos = data.pos.unsqueeze(0)

        # Create mask from edge indices
        adj_mat = to_dense_adj(data.edge_index)

        out = super().forward(x, pos, adj_mat=adj_mat)

        return out.squeeze(0).unsqueeze(-1)
