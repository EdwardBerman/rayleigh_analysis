import os

import torch
from torch.nn import (BatchNorm1d, Conv1d, Dropout, Identity, Linear, Module,
                      ReLU, Sequential)
from torch_scatter import scatter_mean, scatter_sum
from walker import Walker


class VNUpdate(Module):
    def __init__(self, dim, dropout):
        """
        Intermediate update layer for the virtual node
        :param dim: Dimension of the latent node embeddings
        :param dropout: Dropout rate
        """
        super(VNUpdate, self).__init__()

        self.mlp = Sequential(Linear(dim, dim, bias=False),
                              BatchNorm1d(dim),
                              ReLU(),
                              Dropout(dropout),
                              Linear(dim, dim, bias=False))

    def forward(self, data):
        x = scatter_sum(data.h, data.batch, dim=0)
        if 'vn_h' in data:
            x += data.vn_h
        data.vn_h = self.mlp(x)
        data.h += data.vn_h[data.batch]
        return data


class ConvModule(Module):
    def __init__(self, conv_dim, node_dim_in, edge_dim_in, w_feat_dim, dim_out, kernel_size):
        """
        :param conv_dim: Hidden dimension of the convolutions
        :param node_dim_in: Input dimension of the node features
        :param edge_dim_in: Input dimension of the edge features
        :param w_feat_dim: Dimension of the structural encodings of the walk feature tensor (A and I)
        :param dim_out: Dimension of the updated latent node embedding
        :param kernel_size: Kernel size of the convolutions (usually chosen as s+1)
        """
        super(ConvModule, self).__init__()

        self.node_dim_in = node_dim_in
        self.edge_dim_in = edge_dim_in
        self.kernel_size = kernel_size

        # pool into center node
        self.pool_node = kernel_size // 2

        # rescale for residual connection
        self.node_rescale = Linear(
            node_dim_in, dim_out, bias=False) if node_dim_in != dim_out else Identity()

        # lost nodes due to lack of padding:
        self.border = kernel_size - 1

        self.convs = Sequential(
            Conv1d(node_dim_in + edge_dim_in + w_feat_dim,
                   conv_dim, 1, padding=0, bias=False),
            Conv1d(conv_dim, conv_dim, kernel_size,
                   groups=conv_dim, padding=0, bias=False),
            BatchNorm1d(conv_dim),
            ReLU(),
            Conv1d(conv_dim, conv_dim, 1, padding=0, bias=False),
            ReLU()
        )

        self.node_out = Sequential(Linear(conv_dim, 2*dim_out, bias=False),
                                   BatchNorm1d(2*dim_out),
                                   ReLU(),
                                   Linear(2*dim_out, dim_out, bias=False))

    def forward(self, data):
        walk_nodes = data.walk_nodes

        # build walk feature tensor
        walk_node_h = data.h[walk_nodes].transpose(2, 1)
        if 'walk_edge_h' not in data:
            padding = torch.zeros(
                (walk_node_h.shape[0], self.edge_dim_in, 1), dtype=torch.float32, device=walk_node_h.device)
            data.walk_edge_h = torch.cat(
                [padding, data.edge_h[data.walk_edges].transpose(2, 1)], dim=2)
        if 'walk_x' in data:
            x = torch.cat([walk_node_h, data.walk_edge_h, data.walk_x], dim=1)
        else:
            x = torch.cat([walk_node_h, data.walk_edge_h], dim=1)

        # apply the cnn
        y = self.convs(x)

        # pool in walklet embeddings into nodes
        flatt_dim = y.shape[0] * y.shape[2]
        y_flatt = y.transpose(2, 1).reshape(flatt_dim, -1)

        # get center indices
        if 'walk_nodes_flatt' not in data:
            data.walk_nodes_flatt = walk_nodes[:, self.pool_node:-
                                               (self.kernel_size - 1 - self.pool_node)].reshape(-1)

        # pool graphlet embeddings into nodes
        p_node = scatter_mean(y_flatt, data.walk_nodes_flatt,
                              dim=0, dim_size=data.num_nodes)

        # rescale for the residual connection
        data.h = self.node_rescale(data.h)
        data.h += self.node_out(p_node)

        return data


class CRaWl(Module):
    def __init__(self, config, node_feat_dim, edge_feat_dim, layers, hidden, kernel_size, dropout, steps, train_start_ratio, compute_id_feat=True, compute_adj_feat=True, walk_delta=0.0, node_feat_enc=None, edge_feat_enc=None):
        """
        # TODO: Docstrings 
        :param config: Python Dict that specifies the configuration of the model
        :param node_feat_dim: Dimension of the node features
        :param edge_feat_dim: Dimension of the edge features
        :param node_feat_enc: Optional initial embedding of node features
        :param edge_feat_enc: Optional initial embedding of edge features
        """
        super(CRaWl, self).__init__()
        self.config = config
        self.layers = layers
        self.hidden = hidden
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.node_feat_enc = node_feat_enc
        self.edge_feat_enc = edge_feat_enc

        self.walker = Walker(steps, train_start_ratio,
                             compute_id_feat, compute_adj_feat, walk_delta)
        self.walk_dim = self.walker.struc_feat_dim

        modules = []
        for i in range(self.layers):
            modules.append(ConvModule(conv_dim=self.hidden,  # this is always self.hidden for all their configs on github
                                      node_dim_in=node_feat_dim if i == 0 else self.hidden,
                                      edge_dim_in=edge_feat_dim,
                                      w_feat_dim=self.walk_dim,
                                      dim_out=self.hidden,
                                      kernel_size=self.kernel_size))

            if i < self.layers - 1:
                modules.append(VNUpdate(self.hidden, self.dropout))

        self.convs = Sequential(*modules)

        self.node_out = Sequential(BatchNorm1d(self.hidden), ReLU())

        pytorch_total_params = sum(p.numel()
                                   for p in self.parameters() if p.requires_grad)
        print(f'Number of paramters: {pytorch_total_params}')

    def forward(self, data, walk_steps=None, walk_start_p=1.0):
        # apply initial node feature encoding (optional)
        data.h = data.x
        if self.node_feat_enc is not None:
            data.h = self.node_feat_enc(data.h)

        # apply initial edge feature encoding (optional)
        data.edge_h = data.edge_attr
        if self.edge_feat_enc is not None:
            data.edge_h = self.edge_feat_enc(data.edge_h)

        data.walk_edge_h = None
        data.walk_nodes_flatt = None

        # compute walks
        data = self.walker.sample_walks(
            data, steps=walk_steps, start_p=walk_start_p)

        data.vn_h = None

        # apply convolutions
        self.convs(data)

        # pool node embeddings
        data.h = self.node_out(data.h)

        return data
