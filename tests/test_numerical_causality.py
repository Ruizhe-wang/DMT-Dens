import pytest
import torch

from callbacks.numerical_causality import _nonfinite_tensor_summary


def test_nonfinite_tensor_summary_handles_nested_gradients():
    summary = _nonfinite_tensor_summary(
        (None, {"grad": torch.tensor([1.0, float("inf"), float("nan")])})
    )
    assert summary["shape"] == [3]
    assert summary["finite_fraction"] == pytest.approx(1.0 / 3.0)
    assert summary["max_abs_finite"] == 1.0


def test_nonfinite_tensor_summary_returns_none_for_finite_values():
    assert _nonfinite_tensor_summary((torch.ones(2), None)) is None
