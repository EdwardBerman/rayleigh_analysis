import pytest
import torch
from model.model import UnitaryMPNN


@pytest.mark.parametrize("trial", range(10))
def test_act_norm_vector(trial):
    # argmax random vec
    one_hot_atomic_num =
    unitary_gnn = UnitaryMPNN(32)

    assert torch.allclose(random_feature_batch, act_norm_inverse, atol=1e-6), \
        f"Trial {trial}: ActNorm forward and inverse do not match within tolerance."
