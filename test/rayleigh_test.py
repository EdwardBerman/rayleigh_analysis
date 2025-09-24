import pytest
import torch
import torch.nn as nn
from metrics.rayleigh import rayleigh_error
from model.model import LieAlgebra, GroupSort

class DummyModule(nn.Module):
    def __init__(self, L=None):
        super(DummyModule, self).__init__()
        self._L = L
        self._generator = LieAlgebra(20)

    @property
    def L(self):
        return self._L

    @L.setter
    def L(self, value):
        self._L = value

    def _skew_symmeterize(matrix: torch.Tensor) -> torch.Tensor:
        complex_conjugate_matrix = torch.conj(matrix)
        return 0.5 * (matrix - complex_conjugate_matrix.T)

    def forward(self, X: torch.Tensor, edge_indices=None, edge_features=None) -> torch.Tensor:
        U = self._generator(self._skew_symmeterize(self.L))
        return GroupSort(U @ X)

Dummy = DummyModule()


@pytest.mark.parametrize("trial", range(10))
def test_rayleigh_error(trial):

    random_L = torch.randn(5, 5, dtype=torch.cfloat) + 1j * torch.randn(5, 5, dtype=torch.cfloat)
    Dummy.L = random_L
    random_X = torch.randn(5, 20)
    error = rayleigh_error(Dummy, random_X, None, None)

    # error tolerance here should be low, lets assert 
    assert error < 1e-5, f"Trial {trial}: Rayleigh error is too high: {error}"

