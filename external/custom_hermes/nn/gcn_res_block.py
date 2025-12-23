from torch import nn
from torch_geometric.nn import GCNConv


class GCNResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        hid_channels,
        out_channels,
        add_self_loops=False,
        batch_norm=False,
        dropout=False,
        final_activation=True,
    ):
        super().__init__()
        self.conv1 = GCNConv(
            in_channels,
            hid_channels,
            add_self_loops=add_self_loops,
        )
        self.conv2 = GCNConv(
            hid_channels,
            out_channels,
            add_self_loops=add_self_loops,
        )

        # Apply batch norm and dropout inside RegularNonLinearity
        act1 = []
        act2 = []
        if batch_norm:
            act1.append(nn.BatchNorm1d(hid_channels))
            act2.append(nn.BatchNorm1d(out_channels))
        if dropout:
            act1.append(nn.Dropout())
            act2.append(nn.Dropout())
        act1.append(nn.ReLU())
        act2.append(nn.ReLU())

        self.nonlin1 = nn.Sequential(*act1)

        if final_activation:
            self.nonlin2 = nn.Sequential(*act2)
        else:
            self.nonlin2 = nn.Identity()

    def forward(self, x, edge_index):
        y = self.conv1(x, edge_index)
        y = self.nonlin1(y)
        y = self.conv2(y, edge_index)
        y = self.nonlin2(y)

        return y
