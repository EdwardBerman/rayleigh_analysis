"""From https://github.com/Weber-GeoML/Unitary_Convolutions/blob/872aebc9500e59ecb61be5abfb2adc30dc1151d1/layers/complex_valued_layers.py"""

class UnitaryGINEConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out, dropout = 0.0, residual = True, global_bias = True, T = 10, 
                 use_hermitian = False, return_real: bool = False,
                 **kwargs):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dropout = dropout
        self.residual = residual
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(2*dim_out, 2*dim_out),
            torch.nn.ReLU(),
            torch.nn.Linear(2*dim_out, 2*dim_out),
        )
        if global_bias:
            self.bias = torch.nn.Parameter(torch.zeros(dim_out, dtype=torch.cfloat))
        else:
            self.register_parameter('bias', None)

        self.return_real = return_real

        if use_hermitian:
            raise NotImplementedError("Hermitian GINEConv not implemented yet")
        else:
            base_conv = ComplexGINEConv

        self.act = nn.Sequential(
            ComplexActivation(torch.nn.ReLU()),
            ComplexDropout(self.dropout),
        )
        self.model = TaylorGCNConv(base_conv(dim_in, dim_out, **kwargs), T = T)

    def forward(self, x, edge_index, batch: OptTensor = None, edge_attr: OptTensor = None):
        x_in = x

        x = self.model(x, edge_index, edge_attr=edge_attr)
        if self.residual:
            x = x_in + x
        # split real and imaginary parts
        x_real = x.real
        x_imag = x.imag
        # concatenate them
        x = torch.cat([x_real, x_imag], dim=-1)
        # pass through neural network
        x = self.nn(x)
        x = torch.view_as_complex(x.view(x.shape[0],-1,2))
        if self.return_real:
            x = x.real
        return x


class ComplexGINEConv(MessagePassing):
    _cached_edge_index: Optional[OptPairTensor]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        add_self_loops: Optional[bool] = False,
        normalize: bool = True,
        bias: bool = True,
        edge_dim: Optional[int] = None,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        if add_self_loops is None:
            add_self_loops = normalize

        if add_self_loops and not normalize:
            raise ValueError(f"'{self.__class__.__name__}' does not support "
                             f"adding self-loops to the graph when no "
                             f"on-the-fly normalization is applied")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        # self.lin = Linear(in_channels, out_channels, bias=False,
        #                   weight_initializer='glorot')
        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.lin.weight.data = init.orthogonal_(self.lin.weight.data)
        self.lin.weight = torch.nn.Parameter(torch.complex(self.lin.weight, torch.zeros_like(self.lin.weight)))

        if edge_dim is not None:
            self.edge_lin = torch.nn.Linear(edge_dim, in_channels, bias=False)
            self.edge_lin.weight.data = init.orthogonal_(self.edge_lin.weight.data)
            self.edge_lin.weight = torch.nn.Parameter(torch.complex(self.edge_lin.weight, torch.zeros_like(self.edge_lin.weight)))
        else:
            self.edge_lin = None

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels, dtype=torch.cfloat))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()


    def reset_parameters(self):
        super().reset_parameters()
        self.lin.reset_parameters()
        self._cached_edge_index = None
        self._cached_adj_t = None
        if self.bias is not None:
            zeros(self.bias)


    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None,
                edge_attr: OptTensor = None,
                apply_feature_lin: bool = True,
                return_feature_only: bool = False) -> Tensor:

        if isinstance(x, (tuple, list)):
            raise ValueError(f"'{self.__class__.__name__}' received a tuple "
                             f"of node features as input while this layer "
                             f"does not support bipartite message passing. "
                             f"Please try other layers such as 'SAGEConv' or "
                             f"'GraphConv' instead")

        if apply_feature_lin:
            if not torch.is_complex(x):
                x = torch.complex(x, torch.zeros_like(x))
            x = self.lin(x)
            if self.bias is not None:
                x = x + self.bias
            if return_feature_only:
                return x

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow, x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        if self.edge_lin is not None:
            edge_attr = self.edge_lin(edge_attr)
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = 1j*self.propagate(edge_index, x=x, edge_weight=edge_weight, edge_attr=edge_attr)

        return out

    # def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        # return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
    
    def message(self, x_j: Tensor, edge_attr: Tensor, edge_weight: OptTensor) -> Tensor:
        if edge_weight is not None:
            message = (x_j + edge_attr) * edge_weight.view(-1, 1)
        else:
            message = x_j + edge_attr
        return torch.complex(F.relu(message.real), F.relu(message.imag))

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

class ComplexDropout(torch.nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        # If input is complex, apply dropout only to the real part
        if torch.is_complex(x):
            mask = F.dropout(torch.ones_like(x.real), p=self.dropout, training=self.training)
            return x * mask
        else:
            # If input is real, apply dropout as usual
            return F.dropout(x, p=self.dropout, training=self.training)
