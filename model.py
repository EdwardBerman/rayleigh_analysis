import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class GroupSort(nn.Module):
    def forward(self, x):
        a, b = x.split(x.size(-1) // 2, 1)
        a, b = torch.max(a, b), torch.min(a, b)
        return torch.cat([a, b], dim=-1)

class ComplexReLU(nn.Module):
    def forward(self, x):
        real = torch.relu(x.real)
        imag = torch.relu(x.imag)
        return torch.complex(real, imag)

class UnitaryMLP(nn.Module):
    def __init__(self, input_dim , layers, activation=ComplexReLU(), taylor_terms):
        super(UnitaryMLP, self).__init__()
        self.layer_list = []
        self.layers = layers
        self.taylor_terms = taylor_terms
        self.lie_algebra = LieAlgebra(max_matrix_power=taylor_terms, inverse_method='taylor')

        for i in range(layers):
            self.layer_list.append(nn.Parameter(torch.randn(input_dim, input_dim, dtype=torch.cfloat)))

    def forward(self, x):
        x = x.to(torch.cfloat)
        for i in range(len(self.layer_list)):
            weight = self.layer_list[i]
            anti_symmeterized_weight = (weight - weight.conj().t()) / 2
            unitary_matrix = self.lie_algebra.forward(anti_symmeterized_weight)
            x = torch.matmul(x, unitary_matrix)
            x = activation(x) if i != len(self.layer_list) - 1 else x
        return x

class UnitaryMessagePassing(MessagePassing):
    def __init__(self, dimension):
        super(UnitaryMessagePassing, self).__init__(aggr='mean')
        self.weight = nn.Parameter(torch.randn(dimension, dimension, dtype=torch.cfloat))
        self.unitary_mlp = UnitaryMLP(input_dim=dimension, layers=2, activation=ComplexReLU(), taylor_terms=10)

        self.phi_x = nn.Sequential(
                nn.Linear(dimension, dimension),
                GroupSort(),
                nn.Linear(dimension, dimension)
                )

    def forward(self, h, x, edge_index, edge_attr=None, add_loops=False):
        h = h.to(torch.cfloat)
        x = x.to(torch.cfloat)

        if add_loops:
            edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value=0., num_nodes=h.size(0))
        
        m_ij = self.propagate(edge_index,
                              h=h,              # name 'x' is arbitrary; suffixes _i/_j disambiguate
                              x=x,
                              edge_attr=edge_attr)


    def message(self, h_i, h_j, x_i, x_j, edge_attr):
        rij = x_j - x_i                        # [E, d]
        dist = torch.norm(rij, dim=-1, keepdim=True).clamp_min(1e-9)  # [E,1]

        if edge_attr is None:
            feat = torch.cat([h_i, h_j, dist], dim=-1)
        else:
            feat = torch.cat([h_i, h_j, dist, edge_attr], dim=-1)

        return self.unitary_mlp(feat)

class LieAlgebra(nn.Module):
    def __init__(self, max_matrix_power: int, inverse_method: str = 'taylor')):
        super(LieAlgebra, self).__init__()
        self.max_matrix_power = max_matrix_power

    def forward(self, x):
        """ Assuming we are working on a lie algebra and mapping to the lie group """
        return self._taylor_truncated_matrix_exponentiation(x)

    def inverse(self, x):
        """ Assuming we are on the lie group and mapping to the lie algebra """
        return self._taylor_truncated_matrix_logarithm(x) if self.inverse_method == 'taylor' else self._gregory_truncated_matrix_exponentiation(x)

    def _taylor_truncated_matrix_exponentiation(self, x):
        batch_size, dim, _ = x.size()
        max_power = self.max_matrix_power
        batch_matrix_exponentials = torch.zeros((batch_size, dim, dim), device=x.device)
        batch_matrix_exponentials += torch.eye(dim, device=x.device).unsqueeze(0).expand(batch_size, -1, -1)

        current_power = x
        factorial = 1.0
        for i in range(1, max_power + 1):
            batch_matrix_exponentials += current_power / factorial
            factorial *= (i + 1)
            current_power = torch.bmm(current_power, current_power)

        return batch_matrix_exponentials

    def _taylor_truncated_matrix_logarithm(self, x):
        """ Eq 83. https://scipp.ucsc.edu/~haber/webpage/MatrixExpLog.pdf """
        batch_size, dim, _ = x.size()
        max_power = self.max_matrix_power
        current_power = x - torch.eye(dim, device=x.device).unsqueeze(0).expand(batch_size, -1, -1)
        batch_matrix_logarithms = torch.zeros((batch_size, dim, dim), device=x.device)

        for i in range(1, max_power + 1):
            batch_matrix_logarithms += (-1)**(i + 1) * current_power / i
            current_power = torch.bmm(current_power, current_power)

        return batch_matrix_logarithms

    def _gregory_truncated_matrix_exponentiation(self, x):
        """ https://scipp.ucsc.edu/~haber/webpage/MatrixExpLog.pdf Eq 85. https://scipp.ucsc.edu/~haber/webpage/MatrixExpLog.pdf """
        batch_size, dim, _ = x.size()
        max_power = self.max_matrix_power

        current_power_factor_one = torch.eye(dim, device=x.device).unsqueeze(0).expand(batch_size, -1, -1) - x
        current_power_factor_two = torch.eye(dim, device=x.device).unsqueeze(0).expand(batch_size, -1, -1) + x
        current_power_factor_two = torch.inverse(current_power_factor_two)
        current_power = torch.bmm(current_power_factor_one, current_power_factor_two)

        batch_matrix_logarithms = torch.zeros((batch_size, dim, dim), device=x.device)

        for i in range(1, max_power + 1):
            batch_matrix_logarithms +=  torch.matrix_power(current_power, 2*i + 1) / (2*i + 1)

        return -2 * batch_matrix_logarithms
