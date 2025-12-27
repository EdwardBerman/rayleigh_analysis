from se3_transformer_pytorch import SE3Transformer as BaseSE3Transformer
from torch_geometric.utils import to_dense_adj

class SE3Transformer(BaseSE3Transformer):
    def __init__(self, *args, radial_hidden=32, **kwargs):  # Add radial_hidden param
        # Override the default radial_hidden (128) with a smaller value
        super().__init__(*args, radial_hidden=radial_hidden, **kwargs)
    
    def forward(self, data):
        x = data.x.unsqueeze(0).squeeze(-1)
        pos = data.pos.unsqueeze(0)
        adj_mat = to_dense_adj(data.edge_index)
        out = super().forward(x, pos, adj_mat=adj_mat)
        return out.squeeze(0).unsqueeze(-1)
