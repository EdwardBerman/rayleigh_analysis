import torch
import torch.nn as nn
from torch.nn import BatchNorm1d, InstanceNorm1d
from torch_geometric.nn.conv import MessagePassing


class MPNNLayer(MessagePassing):
    def __init__(
        self,
        message_dims,
        update_dims,
        edge_dims,
        message_norm=None,
        update_norm=None,
        message_dropout=False,
        update_dropout=False,
        final_activation=True,
    ):
        super().__init__(aggr="mean", flow="target_to_source", node_dim=0)

        self.edge_dims = edge_dims

        self.message_norm = message_norm
        self.update_norm = update_norm
        self.message_dropout = message_dropout
        self.update_dropout = update_dropout
        self.final_activation = final_activation

        # Construct message layers
        self.message_layers = torch.nn.ModuleList()
        for i in range(len(message_dims) - 1):
            if i == 0:
                # Source + destination + edge features
                in_dim = message_dims[0] + message_dims[0] + self.edge_dims
            else:
                in_dim = message_dims[i]

            self.message_layers.append(
                nn.Linear(
                    in_dim,
                    message_dims[i + 1],
                )
            )

            message_act = []

            if (
                i == len(message_dims) - 2
                and len(update_dims) == 1
                and not final_activation
            ):
                continue
            if self.message_norm == "batch":
                message_act.append(BatchNorm1d(message_dims[i + 1]))
            elif self.message_norm == "instance":
                message_act.append(InstanceNorm1d(message_dims[i + 1]))
            if self.message_dropout:
                message_act.append(nn.Dropout())

            message_act.append(nn.ReLU())

            self.message_layers.append(nn.Sequential(*message_act))

        # Construct update layers
        self.update_layers = torch.nn.ModuleList()
        for i in range(len(update_dims) - 1):
            if i == 0:
                # message dims + residual (source node features)
                in_dim = update_dims[0] + message_dims[0]
            else:
                in_dim = update_dims[i]

            self.update_layers.append(nn.Linear(in_dim, update_dims[i + 1]))

            update_act = []
            if i < len(update_dims) - 2 or (
                i == len(update_dims) - 2 and final_activation
            ):
                if self.update_norm == "batch":
                    update_act.append(BatchNorm1d(update_dims[i + 1]))
                elif self.update_norm == "instance":
                    update_act.append(InstanceNorm1d(update_dims[i + 1]))

                if self.update_dropout:
                    update_act.append(nn.Dropout())

                self.update_layers.append(nn.Sequential(*update_act))

    def forward(self, x, edge_index, edge_attr=None):
        # Propagate messages along edges
        x = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr,
        )

        return x

    def message(self, x_i, x_j, edge_attr=None):
        if edge_attr is not None:
            message = torch.cat([x_i, x_j, edge_attr], dim=1)
        else:
            message = torch.cat([x_i, x_j], dim=1)

        for layer in self.message_layers:
            message = layer(message)

        return message

    def update(self, message, x):
        update = torch.cat([x, message], dim=1)

        for layer in self.update_layers:
            update = layer(update)

        return update

    def __repr__(self):
        return f"\n(message_layers): {repr(self.message_layers)}\n(update_layers):  {repr(self.update_layers)}"
