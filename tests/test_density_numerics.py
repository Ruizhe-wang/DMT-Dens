"""Regression tests for AMP-sensitive manifold and density calculations."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np

from callbacks.xc_paper_embedding_callback import PaperEmbeddingCallback
from model.DiffTreeVQ_density import DMTEVT_model


def test_student_t_row_normalization_stays_finite_for_fp16_distances():
    distances = torch.tensor(
        [
            [0.0, 1000.0, 2000.0, 3000.0],
            [1000.0, 0.0, 1500.0, 2500.0],
            [2000.0, 1500.0, 0.0, 1200.0],
            [3000.0, 2500.0, 1200.0, 0.0],
        ],
        dtype=torch.float16,
    )
    affinity = DMTEVT_model.t_distribution_similarity(distances, df=0.01)

    assert affinity.dtype == torch.float32
    assert torch.isfinite(affinity).all()
    off_diagonal = affinity.clone()
    off_diagonal.fill_diagonal_(0.0)
    torch.testing.assert_close(
        off_diagonal.sum(dim=1),
        torch.ones(4),
        rtol=1e-5,
        atol=1e-6,
    )


def test_pair_bce_ignores_undefined_diagonal_negative_term():
    target = torch.tensor(
        [[1.0, 0.2, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]],
        dtype=torch.float32,
    )
    # Q's row denominator excludes the diagonal, so Q_ii can exceed one.
    affinity = torch.tensor(
        [[3.0, 0.6, 0.4], [0.7, 2.0, 0.3], [0.2, 0.8, 4.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    loss = DMTEVT_model._bernoulli_pair_loss(target, affinity).mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(affinity.grad).all()


def test_knn_log_density_keeps_fp16_zero_radius_finite():
    points = torch.zeros((16, 2), dtype=torch.float16, requires_grad=True)
    density = DMTEVT_model._compute_knn_log_density(points[:4], points, k=3)
    density.sum().backward()

    assert density.dtype == torch.float32
    assert torch.isfinite(density).all()
    assert points.grad is not None
    assert torch.isfinite(points.grad).all()


def test_knn_log_density_floor_stops_subfloor_distance_gradient():
    points = torch.tensor(
        [[0.0, 0.0], [1.0e-6, 0.0], [1.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    density = DMTEVT_model._compute_knn_log_density(
        points[:1], points, k=1, distance_floor=1.0e-4
    )
    density.sum().backward()

    assert torch.isfinite(density).all()
    torch.testing.assert_close(points.grad, torch.zeros_like(points.grad))


def test_global_standardization_is_differentiable_and_preserves_axis_ratio():
    class ModelProxy:
        hparams = SimpleNamespace(
            embedding_standardization="differentiable_global",
            stable_embedding_standardization=False,
            embedding_std_floor=1.0e-4,
        )
        _embedding_standardization_mode = DMTEVT_model._embedding_standardization_mode

    torch.manual_seed(42)
    embedding = torch.randn(128, 2, requires_grad=True)
    with torch.no_grad():
        embedding[:, 1].mul_(0.01)
    normalized = DMTEVT_model._standardize_embedding(ModelProxy(), embedding)

    raw_ratio = (
        embedding.detach().std(dim=0, unbiased=False).min()
        / embedding.detach().std(dim=0, unbiased=False).max()
    )
    normalized_ratio = (
        normalized.detach().std(dim=0, unbiased=False).min()
        / normalized.detach().std(dim=0, unbiased=False).max()
    )
    torch.testing.assert_close(normalized_ratio, raw_ratio)

    (normalized.square().mean()).backward()
    assert embedding.grad is not None
    assert torch.isfinite(embedding.grad).all()


def test_global_standardization_point_collapse_has_finite_gradient():
    class ModelProxy:
        hparams = SimpleNamespace(
            embedding_standardization="differentiable_global",
            stable_embedding_standardization=False,
            embedding_std_floor=1.0e-4,
        )
        _embedding_standardization_mode = DMTEVT_model._embedding_standardization_mode

    embedding = torch.zeros(128, 2, requires_grad=True)
    normalized = DMTEVT_model._standardize_embedding(ModelProxy(), embedding)
    normalized[0, 0].backward()

    assert torch.isfinite(normalized).all()
    assert embedding.grad is not None
    assert torch.isfinite(embedding.grad).all()


def test_zero_density_weight_skips_density_objective():
    class ModelProxy:
        hparams = SimpleNamespace(
            tau=1.0,
            use_orthogonal=False,
            loss_type="G",
            exaggeration_emb=1.0,
            nu_emb=0.01,
            num_use_mlevel_list=[1],
            density_weight=0.0,
            embedding_standardization="differentiable_global",
            stable_embedding_standardization=False,
            embedding_std_floor=1.0e-4,
        )
        _embedding_standardization_mode = DMTEVT_model._embedding_standardization_mode
        _standardize_embedding = DMTEVT_model._standardize_embedding
        _raw_embedding_diagnostics = staticmethod(
            DMTEVT_model._raw_embedding_diagnostics
        )

        def __call__(self, data_input, tau):
            lat_vis = data_input[:, :2]
            return data_input, data_input, lat_vis, [lat_vis]

        @staticmethod
        def LossManifold_Global(input_data, latent_data, **kwargs):
            return latent_data.square().mean()

        @staticmethod
        def _compute_local_density_loss(*args, **kwargs):
            raise AssertionError("disabled density objective must not be evaluated")

    data_input_item = torch.randn(8, 4)
    data_input_aug = torch.randn(8, 4)
    loss_total, _, _, _, density_loss = DMTEVT_model.forward_train_enc(
        ModelProxy(), data_input_item, data_input_aug
    )

    assert torch.isfinite(loss_total)
    torch.testing.assert_close(density_loss, torch.zeros_like(density_loss))


def test_raw_embedding_diagnostics_disable_amp_for_eigensolver():
    torch.manual_seed(42)
    embedding = torch.randn(128, 2)
    embedding[:, 1].mul_(0.01)

    # CPU autocast reproduces the relevant failure mode with bfloat16: matrix
    # multiplication is downcast, but the symmetric eigensolver requires a
    # supported floating-point dtype.  The helper must keep this path fp32.
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        diagnostics = DMTEVT_model._raw_embedding_diagnostics(embedding)

    assert set(diagnostics) == {
        "embedding/raw_std_axis0",
        "embedding/raw_std_axis1",
        "embedding/raw_global_scale",
        "embedding/raw_coordinate_std_ratio",
        "embedding/raw_axis_ratio",
        "embedding/raw_near_line",
    }
    assert all(value.dtype == torch.float32 for value in diagnostics.values())
    assert all(torch.isfinite(value) for value in diagnostics.values())
    assert 0.0 <= diagnostics["embedding/raw_axis_ratio"] <= 1.0
    assert diagnostics["embedding/raw_near_line"] == 1.0


def test_pearson_denominator_floor_keeps_constant_density_finite():
    x = torch.ones(16, requires_grad=True)
    y = torch.linspace(-1.0, 1.0, 16)
    loss = DMTEVT_model._pearson_correlation_loss(x, y, denominator_floor=1.0e-4)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(x.grad).all()


def test_paper_plot_handles_point_collapsed_embedding(tmp_path):
    callback = PaperEmbeddingCallback(
        output_dir=str(tmp_path),
        formats=("png", "pdf", "svg"),
        dpi=300,
    )
    xy = np.zeros((1000, 2), dtype=np.float64)
    labels = np.asarray([str(i % 5) for i in range(1000)])
    paths = callback._save_clean(
        xy,
        labels,
        categorical=True,
        fname="collapsed",
    )

    assert len(paths) == 3
    assert all(Path(path).stat().st_size > 0 for path in paths)
