import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, MessagePassing


class GCNReg(nn.Module):
    def __init__(self, in_ch, hidden, dropout=0.1):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hidden)
        self.conv2 = GCNConv(hidden, 1)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, self.training)
        x = self.conv2(x, edge_index)
        return x.squeeze(-1)


class GATReg(nn.Module):
    def __init__(self, in_ch, hidden, heads=4, dropout=0.2):
        super().__init__()
        self.gat1 = GATConv(in_ch, hidden, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden * heads, 1, heads=1,
                            concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return x.squeeze(-1)


class SimpleMPNN(MessagePassing):
    def __init__(self, in_ch, out_ch):
        super().__init__(aggr="add")
        self.msg = nn.Linear(in_ch, out_ch)
        self.upd = nn.Linear(in_ch + out_ch, out_ch)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return F.relu(self.msg(x_j))

    def update(self, aggr_out, x):
        return F.relu(self.upd(torch.cat([x, aggr_out], dim=-1)))


class MPNNReg(nn.Module):
    def __init__(self, in_ch, hidden, n_layers=2):
        super().__init__()
        self.input_lin = nn.Linear(in_ch, hidden)
        self.layers = nn.ModuleList(
            [SimpleMPNN(hidden, hidden) for _ in range(n_layers)])
        self.output_lin = nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.input_lin(x))
        for layer in self.layers:
            x = layer(x, edge_index)
        x = self.output_lin(x)
        return x.squeeze(-1)
