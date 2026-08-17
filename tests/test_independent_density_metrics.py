"""Sanity checks for the independent density-fidelity metrics (ASR, FR-DC).

These verify the spec's acceptance criteria locally, without the training
pipeline: value range, directional sanity (a density-faithful embedding beats
a dense-cluster-inflating one), determinism, and NaN handling.

Run: .venv/Scripts/python.exe -m pytest tests/test_independent_density_metrics.py -q
or:  .venv/Scripts/python.exe tests/test_independent_density_metrics.py
"""
import numpy as np

from eval.fidelity_eval import compute_asr, compute_frdc


def _make_dataset(seed=0, n_classes=12, dim=64):
    """HD blobs whose per-class spread varies widely. Returns (hd, labels)."""
    rng = np.random.RandomState(seed)
    spreads = np.linspace(0.2, 6.0, n_classes)  # known, monotone spread per class
    centers = rng.uniform(-50, 50, size=(n_classes, dim))
    hd, labels = [], []
    for c, (ctr, s) in enumerate(zip(centers, spreads)):
        n_c = rng.randint(60, 200)
        hd.append(ctr + s * rng.randn(n_c, dim))
        labels.append(np.full(n_c, c))
    return np.vstack(hd).astype(np.float32), np.concatenate(labels), spreads


def _faithful_embedding(hd, labels, spreads):
    """A density-faithful 2D embedding: class area scales with HD spread."""
    rng = np.random.RandomState(1)
    n_classes = len(spreads)
    centers2d = rng.uniform(-50, 50, size=(n_classes, 2))
    z = np.zeros((len(hd), 2), dtype=np.float32)
    for c in range(n_classes):
        idx = labels == c
        z[idx] = centers2d[c] + spreads[c] * rng.randn(idx.sum(), 2)
    return z


def _inflating_embedding(hd, labels, spreads):
    """A t-SNE-like embedding: every class inflated to roughly equal area,
    so 2D area no longer reflects HD spread."""
    rng = np.random.RandomState(2)
    n_classes = len(spreads)
    centers2d = rng.uniform(-50, 50, size=(n_classes, 2))
    z = np.zeros((len(hd), 2), dtype=np.float32)
    for c in range(n_classes):
        idx = labels == c
        z[idx] = centers2d[c] + 1.0 * rng.randn(idx.sum(), 2)  # constant spread
    return z


def test_range_and_directional_sanity():
    hd, labels, spreads = _make_dataset()
    z_good = _faithful_embedding(hd, labels, spreads)
    z_bad = _inflating_embedding(hd, labels, spreads)

    asr_good, asr_mpd_good = compute_asr(z_good, hd, labels)
    asr_bad, asr_mpd_bad = compute_asr(z_bad, hd, labels)
    frdc_good = compute_frdc(hd, z_good)
    frdc_bad = compute_frdc(hd, z_bad)

    for v in (asr_good, asr_bad, asr_mpd_good, asr_mpd_bad, frdc_good, frdc_bad):
        assert -1.0 <= v <= 1.0, v

    # Density-faithful embedding must score higher than the inflating one.
    assert asr_good > asr_bad, (asr_good, asr_bad)
    assert asr_mpd_good > asr_mpd_bad, (asr_mpd_good, asr_mpd_bad)
    assert frdc_good > frdc_bad, (frdc_good, frdc_bad)
    print(f"ASR good={asr_good:.3f} bad={asr_bad:.3f} | "
          f"ASR_mpd good={asr_mpd_good:.3f} bad={asr_mpd_bad:.3f} | "
          f"FR-DC good={frdc_good:.3f} bad={frdc_bad:.3f}")


def test_determinism():
    hd, labels, spreads = _make_dataset(seed=3)
    z = _faithful_embedding(hd, labels, spreads)
    # Fresh arrays (different id) to defeat the cache and prove determinism.
    a1 = compute_asr(z.copy(), hd.copy(), labels)
    a2 = compute_asr(z.copy(), hd.copy(), labels)
    f1 = compute_frdc(hd.copy(), z.copy())
    f2 = compute_frdc(hd.copy(), z.copy())
    assert a1 == a2, (a1, a2)
    assert f1 == f2, (f1, f2)


def test_asr_nan_when_too_few_classes():
    hd, labels, spreads = _make_dataset(n_classes=3)  # < c_min (8)
    z = _faithful_embedding(hd, labels, spreads)
    asr, asr_mpd = compute_asr(z, hd, labels)
    assert np.isnan(asr) and np.isnan(asr_mpd)


def test_reference_identical_across_methods():
    """S_c / n_HD must be identical for two different embeddings of one dataset."""
    from eval.fidelity_eval import _ASR_SPREAD_CACHE, _FRDC_REF_CACHE
    _ASR_SPREAD_CACHE.clear()
    _FRDC_REF_CACHE.clear()
    hd, labels, spreads = _make_dataset(seed=5)
    z_a = _faithful_embedding(hd, labels, spreads)
    z_b = _inflating_embedding(hd, labels, spreads)
    # Same hd object across both calls -> n_HD cached and reused (byte-identical).
    compute_frdc(hd, z_a)
    key = next(iter(_FRDC_REF_CACHE))
    n_hd_after_a = _FRDC_REF_CACHE[key].copy()
    compute_frdc(hd, z_b)
    assert np.array_equal(n_hd_after_a, _FRDC_REF_CACHE[key])


if __name__ == "__main__":
    test_range_and_directional_sanity()
    test_determinism()
    test_asr_nan_when_too_few_classes()
    test_reference_identical_across_methods()
    print("All sanity checks passed.")
