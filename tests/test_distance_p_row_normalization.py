import pytest
import torch

from model.DiffTreeVQ_density import DMTEVT_model


def test_distance_p_row_normalization_preserves_diagonal_and_normalizes_off_diagonal():
    affinity = torch.tensor(
        [
            [1.0, 0.5, 0.25],
            [0.2, 0.9, 0.4],
            [0.3, 0.1, 0.8],
        ],
        dtype=torch.float64,
    )

    normalized = DMTEVT_model._row_normalize_off_diagonal_affinity(affinity)

    assert normalized.dtype == torch.float32
    assert torch.diagonal(normalized).tolist() == pytest.approx([1.0, 0.9, 0.8])
    off_diagonal = normalized.clone()
    off_diagonal.fill_diagonal_(0.0)
    assert off_diagonal.sum(dim=1).tolist() == pytest.approx([1.0, 1.0, 1.0])


def test_distance_p_row_normalization_handles_zero_off_diagonal_mass():
    affinity = torch.eye(3)
    normalized = DMTEVT_model._row_normalize_off_diagonal_affinity(affinity)
    assert torch.equal(normalized, affinity)
