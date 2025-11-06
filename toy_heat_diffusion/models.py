import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import BatchNorm, GATConv, GCNConv, MessagePassing


class GCNReg(nn.Module):
    def __init__(self, in_ch, hidden=32, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.layers.append(GCNConv(in_ch, hidden))
        self.norms.append(BatchNorm(hidden))

        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden, hidden))
            self.norms.append(BatchNorm(hidden))

        self.layers.append(GCNConv(hidden, 1))

    def forward(self, x, edge_index):
        for conv, norm in zip(self.layers[:-1], self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
        x = self.layers[-1](x, edge_index)
        return x.squeeze(-1)


class GATReg(nn.Module):
    def __init__(self, in_ch, hidden=64, heads=4, num_layers=3, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.layers.append(
            GATConv(in_ch, hidden, heads=heads, concat=True, dropout=dropout))
        self.norms.append(BatchNorm(hidden * heads))

        for _ in range(num_layers - 2):
            self.layers.append(
                GATConv(hidden * heads, hidden, heads=heads, concat=True, dropout=dropout))
            self.norms.append(BatchNorm(hidden * heads))

        self.layers.append(
            GATConv(hidden * heads, 1, heads=1, concat=False, dropout=dropout))

    def forward(self, x, edge_index):
        for conv, norm in zip(self.layers[:-1], self.norms):
            residual = x
            x = conv(x, edge_index)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if x.shape == residual.shape:
                x = x + residual
        x = self.layers[-1](x, edge_index)
        return x.squeeze(-1)


class SimpleMPNN(MessagePassing):
    def __init__(self, in_ch, out_ch, dropout=0.2):
        super().__init__(aggr="add")
        self.msg = nn.Sequential(
            nn.Linear(in_ch, out_ch),
            nn.ReLU(),
            nn.Linear(out_ch, out_ch)
        )
        self.upd = nn.GRUCell(out_ch, out_ch)
        self.dropout = dropout

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        return F.dropout(self.msg(x_j), p=self.dropout, training=self.training)

    def update(self, aggr_out, x):
        return self.upd(aggr_out, x)


class MPNNReg(nn.Module):
    def __init__(self, in_ch, hidden=64, n_layers=4, dropout=0.2):
        super().__init__()
        self.input_lin = nn.Linear(in_ch, hidden)
        self.layers = nn.ModuleList(
            [SimpleMPNN(hidden, hidden, dropout) for _ in range(n_layers)])
        self.norms = nn.ModuleList([BatchNorm(hidden)
                                   for _ in range(n_layers)])
        self.output = nn.Linear(hidden, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.input_lin(x))
        for layer, norm in zip(self.layers, self.norms):
            residual = x
            x = layer(x, edge_index)
            x = norm(x)
            if x.shape == residual.shape:
                x = x + residual
        x = self.output(x)
        return x.squeeze(-1)
