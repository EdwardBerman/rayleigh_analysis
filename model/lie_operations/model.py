import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

from external.torch_scatter import scatter_mean


class ComplexReLU(nn.Module):
    def forward(self, x):
        x = x.to(torch.complex64) if x.dtype in [
            torch.float16, torch.float32, torch.float64] else x
        real = torch.relu(x.real)
        imag = torch.relu(x.imag)
        return torch.complex(real, imag)


class LieAlgebra(nn.Module):
    def __init__(self, max_matrix_power: int, inverse_method: str = 'taylor'):
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
        batch_matrix_exponentials = torch.zeros(
            (batch_size, dim, dim), device=x.device)
        batch_matrix_exponentials += torch.eye(
            dim, device=x.device).unsqueeze(0).expand(batch_size, -1, -1)

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
        current_power = x - \
            torch.eye(dim, device=x.device).unsqueeze(
                0).expand(batch_size, -1, -1)
        batch_matrix_logarithms = torch.zeros(
            (batch_size, dim, dim), device=x.device)

        for i in range(1, max_power + 1):
            batch_matrix_logarithms += (-1)**(i + 1) * current_power / i
            current_power = torch.bmm(current_power, current_power)

        return batch_matrix_logarithms

    def _gregory_truncated_matrix_exponentiation(self, x):
        """ https://scipp.ucsc.edu/~haber/webpage/MatrixExpLog.pdf Eq 85. https://scipp.ucsc.edu/~haber/webpage/MatrixExpLog.pdf """
        batch_size, dim, _ = x.size()
        max_power = self.max_matrix_power

        current_power_factor_one = torch.eye(
            dim, device=x.device).unsqueeze(0).expand(batch_size, -1, -1) - x
        current_power_factor_two = torch.eye(
            dim, device=x.device).unsqueeze(0).expand(batch_size, -1, -1) + x
        current_power_factor_two = torch.inverse(current_power_factor_two)
        current_power = torch.bmm(
            current_power_factor_one, current_power_factor_two)

        batch_matrix_logarithms = torch.zeros(
            (batch_size, dim, dim), device=x.device)

        for i in range(1, max_power + 1):
            batch_matrix_logarithms += torch.matrix_power(
                current_power, 2*i + 1) / (2*i + 1)

        return -2 * batch_matrix_logarithms

# class UnitaryConvolution(
