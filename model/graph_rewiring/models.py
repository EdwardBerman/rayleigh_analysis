import torch
from torch_geometric.data import Data
from torch_geometric.nn.models import GraphSAGE

# Example graph data
# Suppose we have 5 nodes, each with 16 features
x = torch.randn((5, 16))  # node features

# Suppose edges (undirected) between nodes; edge_index shape [2, E]
edge_index = torch.tensor([
    [0, 1, 2, 3, 4, 0],
    [1, 0, 3, 2, 0, 4],
], dtype=torch.long)

# Define the model
model = GraphSAGE(
    in_channels=16,
    hidden_channels=32,
    num_layers=2,
    out_channels=10,     # e.g. for classification into 10 classes
    dropout=0.5
)

# Forward pass
out = model(x, edge_index)
# `out` will have shape [N, 10] because out_channels=10
print(out.shape)  # torch.Size([5, 10])
