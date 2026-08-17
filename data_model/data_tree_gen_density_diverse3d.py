"""Diverse-density 3D branching synthetic dataset for the topoBranch Figure 1.

Motivation
----------
This is a 3D analogue of the 2D synthetic illustrations used in the densNE /
densMAP paper (Narayan, Berger & Cho, 2021). Those papers use a deliberately
simple 2D toy where regions have *different* local densities so that the reader
can see at a glance that t-SNE/UMAP equalize density while density-preserving
methods keep it. We push the same idea to a 3D *branching* manifold, which is
the object our paper actually cares about (continuous branching trajectories
with topology + density structure).

Design goals (matching the user's brief)
-----------------------------------------
1. Intuitive & clear:   a single connected tree -- root trunk, a first
   bifurcation, then one arm bifurcates again. 3 tips, 2 branch points. The
   centerline is drawn so the topology reads instantly.
2. Diverse density:     every segment is given a *different* density regime --
   uniformly dense, uniformly sparse, a smooth root->tip gradient, and a
   localized dense "bulge" in the middle of a branch. Tube radius also varies
   per branch, so volumetric density differs too. Coloring by ground-truth
   local density therefore spans the full range, which is exactly what a
   density-preservation claim needs.
3. Baselines + 2D plots: PCA, t-SNE, UMAP, densMAP, densNE and (optionally)
   PHATE are run and rendered as 2D embeddings, colored both by branch id and
   by ground-truth 3D local density. A density-preservation score (Spearman
   rho between original and embedded local density) is reported per method.

This module is intentionally separate from ``data_tree_gen_density_demo.py``
(the older monotone-density version) so both remain runnable.

Outputs (under ``--save-path`` stem):
  - <stem>.npz / <stem>.mat                 raw dataset + ground truth
  - <stem>_3d_truth.png                     3D views (branch id + density)
  - <stem>_baselines_2d.png                 grid of 2D baseline embeddings
  - <stem>_baseline_embeddings.npz          every 2D embedding
  - <stem>_density_preservation_report.txt  per-method Spearman table
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import numpy as np
from scipy.io import savemat
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_DEFAULT_PAPER_SAVE_DIR = Path(
    r"D:\ruizhe\69d8b71abecbd17080a7c4ab\bio_dmt_TB\paper_topoBranch"
    r"\Fig\synthetic_branch3d"
)


# --------------------------------------------------------------------------- #
# Geometry + density specification
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DensityMode:
    """A localized Gaussian bump in the along-branch density profile."""

    center: float       # progress in [0, 1] where the bump sits
    width: float        # std of the bump, in progress units
    amplitude: float    # relative height added on top of the baseline


@dataclass(frozen=True)
class BranchSpec:
    name: str
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    start_time: float
    end_time: float
    samples_scale: float = 1.0
    # density profile along the branch = baseline + sum of Gaussian modes
    density_baseline: float = 1.0
    density_modes: tuple[DensityMode, ...] = field(default_factory=tuple)
    curve: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tube_scale: float = 1.0          # base blob-radius multiplier
    tube_growth: float = 0.0         # blob radius grows with progress (toward tip)
    coverage_fraction: float = 0.18  # fraction of points spread uniformly
    n_components: int = 9            # number of Gaussian blobs strung along branch
    regime: str = ""                 # human-readable density description


@dataclass
class TreeDensityDemo:
    coords: np.ndarray
    branch_id: np.ndarray
    branch_names: np.ndarray
    pseudotime: np.ndarray
    branch_progress: np.ndarray
    local_density: np.ndarray
    local_radius: np.ndarray
    centerlines: list[np.ndarray]
    branch_specs: list[BranchSpec]


def _as_array(x: tuple[float, float, float]) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _smoothstep(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


def _density_profile(spec: BranchSpec, t: np.ndarray) -> np.ndarray:
    """Target relative density along the branch (before normalization)."""
    rho = np.full_like(t, spec.density_baseline, dtype=np.float64)
    for mode in spec.density_modes:
        rho = rho + mode.amplitude * np.exp(
            -0.5 * ((t - mode.center) / max(mode.width, 1e-6)) ** 2
        )
    return np.clip(rho, 1e-6, None)


def _sample_progress(spec: BranchSpec, n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample along-branch progress from the branch's density profile.

    A fraction of points are spread uniformly (``coverage_fraction``) so every
    branch is fully traced even where the profile is near-zero; the rest follow
    the inverse-CDF of the density profile so dense regions get more mass.
    """
    grid = np.linspace(0.0, 1.0, 2001)
    pdf = _density_profile(spec, grid)
    cdf = np.cumsum(pdf)
    cdf = cdf - cdf[0]
    cdf = cdf / cdf[-1]

    n_cover = max(6, int(round(n * spec.coverage_fraction)))
    n_cover = min(n_cover, max(n - 1, 1))
    n_dense = max(n - n_cover, 1)

    u = (np.arange(n_dense, dtype=np.float64) + rng.random(n_dense)) / n_dense
    dense_t = np.interp(u, cdf, grid)

    cover_t = np.linspace(0.0, 1.0, n_cover)
    if n_cover > 2:
        step = 1.0 / (n_cover - 1)
        jitter = rng.normal(0.0, 0.08 * step, size=n_cover)
        jitter[0] = 0.0
        jitter[-1] = 0.0
        cover_t = np.clip(cover_t + jitter, 0.0, 1.0)

    return np.sort(np.clip(np.concatenate([dense_t, cover_t]), 0.0, 1.0))


def _centerline(spec: BranchSpec, n: int = 140) -> np.ndarray:
    start = _as_array(spec.start)
    end = _as_array(spec.end)
    curve = _as_array(spec.curve)
    t = np.linspace(0.0, 1.0, n)
    linear = start[None, :] + (end - start)[None, :] * t[:, None]
    bend = np.sin(np.pi * t)[:, None] * curve[None, :]
    return linear + bend


def _branch_points(
    spec: BranchSpec,
    n: int,
    tube_sigma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = _sample_progress(spec, n, rng)
    start = _as_array(spec.start)
    end = _as_array(spec.end)
    curve = _as_array(spec.curve)
    smooth_t = _smoothstep(t)
    center = start[None, :] + (end - start)[None, :] * smooth_t[:, None]
    center += np.sin(np.pi * t)[:, None] * curve[None, :]

    tangent = end - start
    tangent = tangent / (np.linalg.norm(tangent) + 1e-12)
    raw = rng.normal(size=(n, 3))
    radial = raw - (raw @ tangent)[:, None] * tangent[None, :]
    radial_norm = np.linalg.norm(radial, axis=1, keepdims=True)
    radial = radial / np.clip(radial_norm, 1e-12, None)
    local_tube_scale = spec.tube_scale * (1.0 + spec.tube_growth * smooth_t)
    radius = rng.normal(loc=0.0, scale=tube_sigma * local_tube_scale[:, None], size=(n, 1))
    coords = center + radial * radius

    pseudotime = spec.start_time + (spec.end_time - spec.start_time) * t
    return coords, t, pseudotime


def _branch_points_gmm(
    spec: BranchSpec,
    n_total: int,
    blob_sigma: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render a branch as a mixture of Gaussian blobs strung along its centerline.

    ``n_components`` blob centers are placed along the branch. Each blob's point
    count is proportional to the branch density profile (dense base, sparse tip),
    and each blob's spread grows toward the tip (``tube_growth``) and scales with
    ``tube_scale`` -- so dense arms read as tight, populous clouds and sparse arms
    as few, diffuse clouds.
    """
    k = max(2, spec.n_components)
    start = _as_array(spec.start)
    end = _as_array(spec.end)
    curve = _as_array(spec.curve)

    t_centers = np.linspace(0.04, 0.98, k)
    smooth = _smoothstep(t_centers)
    centers = (start[None, :] + (end - start)[None, :] * smooth[:, None]
               + np.sin(np.pi * t_centers)[:, None] * curve[None, :])
    sigmas = blob_sigma * spec.tube_scale * (1.0 + spec.tube_growth * smooth)

    weights = _density_profile(spec, t_centers)
    weights = weights / weights.sum()
    floor = min(3, max(1, n_total // k))       # keep blobs visible without changing n_total
    n_free = max(n_total - floor * k, 0)
    counts = rng.multinomial(n_free, weights) + floor

    pts, prog = [], []
    for i in range(k):
        n_i = int(counts[i])
        pts.append(centers[i][None, :] + rng.normal(0.0, sigmas[i], size=(n_i, 3)))
        prog.append(np.full(n_i, t_centers[i]) + rng.normal(0.0, 0.012, size=n_i))
    coords = np.vstack(pts)
    progress = np.clip(np.concatenate(prog), 0.0, 1.0)
    pseudotime = spec.start_time + (spec.end_time - spec.start_time) * progress
    return coords, progress, pseudotime


def build_diverse_branch_specs() -> list[BranchSpec]:
    """A single, clean 3D bifurcation: one dense trunk that splits into THREE
    diverging arms, each at a clearly different density.

    This is the minimal object that says everything the motivation figure needs:
      * obvious branching topology (one branch point, three arms);
      * diverse density -- the three arms are dense / medium / sparse, and every
        arm additionally thins from a dense base to a sparse tip;
      * dense root.
    Developmental axis = z (the tree grows UP); the three arms are spread 120 deg
    apart in azimuth and tilted up, so they separate cleanly in 3D without the
    clutter of a multi-level tree.
    """
    branch_point = (0.0, 0.0, 1.1)

    def arm_end(az_deg: float, tilt_deg: float = 48.0, length: float = 3.4):
        az, tilt = np.deg2rad(az_deg), np.deg2rad(tilt_deg)
        direction = np.array([np.sin(tilt) * np.cos(az),
                              np.sin(tilt) * np.sin(az),
                              np.cos(tilt)])
        return tuple(np.asarray(branch_point) + length * direction)

    return [
        BranchSpec(
            name="trunk",
            start=(0.0, 0.0, 0.0), end=branch_point,
            start_time=0.00, end_time=0.30,
            samples_scale=2.2,
            density_baseline=9.0,                 # dense progenitor trunk
            tube_scale=0.20, tube_growth=0.4, n_components=24,
            regime="dense trunk",
        ),
        BranchSpec(
            name="dense arm",
            start=branch_point, end=arm_end(90.0),
            start_time=0.30, end_time=1.00,
            samples_scale=1.5,
            density_baseline=3.5,                 # densest arm
            density_modes=(DensityMode(center=0.0, width=0.45, amplitude=5.0),),
            tube_scale=0.30, tube_growth=1.0, n_components=55,
            regime="dense arm (dense base -> sparse tip)",
        ),
        BranchSpec(
            name="medium arm",
            start=branch_point, end=arm_end(210.0),
            start_time=0.30, end_time=1.00,
            samples_scale=1.0,
            density_baseline=1.1,                 # intermediate arm
            density_modes=(DensityMode(center=0.0, width=0.45, amplitude=4.0),),
            tube_scale=0.46, tube_growth=1.7, n_components=42,
            regime="medium arm (dense base -> sparse tip)",
        ),
        BranchSpec(
            name="sparse arm",
            start=branch_point, end=arm_end(330.0),
            start_time=0.30, end_time=1.00,
            samples_scale=0.6,
            density_baseline=0.28,                # sparsest arm
            density_modes=(DensityMode(center=0.0, width=0.45, amplitude=3.0),),
            tube_scale=0.60, tube_growth=2.2, n_components=32,
            regime="sparse arm (dense base -> very sparse tip)",
        ),
    ]


def build_root_dense_sparse_transition_specs() -> list[BranchSpec]:
    """Dense root connected to three sparse terminal states by sparse transitions."""
    root_end = (0.0, 0.0, 0.95)

    def direction(az_deg: float, tilt_deg: float = 50.0) -> np.ndarray:
        az, tilt = np.deg2rad(az_deg), np.deg2rad(tilt_deg)
        return np.array([np.sin(tilt) * np.cos(az),
                         np.sin(tilt) * np.sin(az),
                         np.cos(tilt)])

    def endpoint(start: tuple[float, float, float], az_deg: float, length: float) -> tuple[float, float, float]:
        return tuple(np.asarray(start) + length * direction(az_deg))

    specs = [
        BranchSpec(
            name="dense root",
            start=(0.0, 0.0, 0.0), end=root_end,
            start_time=0.00, end_time=0.25,
            samples_scale=2.4,
            density_baseline=12.0,
            tube_scale=0.17, tube_growth=0.15, n_components=24,
            regime="dense root state",
        )
    ]

    for label, az, curve_scale in [
        ("A", 90.0, (0.12, 0.00, 0.08)),
        ("B", 210.0, (-0.08, 0.10, 0.04)),
        ("C", 330.0, (-0.08, -0.10, 0.04)),
    ]:
        transition_end = endpoint(root_end, az, 1.55)
        state_end = endpoint(transition_end, az, 1.35)
        curve = tuple(curve_scale)
        specs.extend([
            BranchSpec(
                name=f"transition {label}",
                start=root_end, end=transition_end,
                start_time=0.25, end_time=0.62,
                samples_scale=0.28,
                density_baseline=0.22,
                curve=curve,
                tube_scale=0.78, tube_growth=0.75, n_components=18,
                regime="sparse transition from dense root to terminal state",
            ),
            BranchSpec(
                name=f"sparse state {label}",
                start=transition_end, end=state_end,
                start_time=0.62, end_time=1.00,
                samples_scale=0.42,
                density_baseline=0.18,
                density_modes=(DensityMode(center=0.55, width=0.40, amplitude=0.08),),
                curve=curve,
                tube_scale=0.96, tube_growth=0.55, n_components=16,
                regime="sparse terminal state",
            ),
        ])
    return specs


def build_dense_states_sparse_transition_specs() -> list[BranchSpec]:
    """Dense state clusters connected by sparse transitions through one branch point."""
    root = (0.0, 0.0, 0.0)
    branch_point = (0.0, 0.0, 1.65)

    def direction(az_deg: float, tilt_deg: float = 50.0) -> np.ndarray:
        az, tilt = np.deg2rad(az_deg), np.deg2rad(tilt_deg)
        return np.array([np.sin(tilt) * np.cos(az),
                         np.sin(tilt) * np.sin(az),
                         np.cos(tilt)])

    def endpoint(az_deg: float, length: float = 2.55) -> tuple[float, float, float]:
        return tuple(np.asarray(branch_point) + length * direction(az_deg))

    specs = [
        BranchSpec(
            name="dense root state",
            start=root, end=root,
            start_time=0.00, end_time=0.04,
            samples_scale=1.55,
            density_baseline=10.0,
            tube_scale=0.32, tube_growth=0.0, n_components=8,
            regime="compact dense root state",
        ),
        BranchSpec(
            name="sparse transition trunk",
            start=root, end=branch_point,
            start_time=0.04, end_time=0.42,
            samples_scale=0.09,
            density_baseline=0.18,
            tube_scale=0.40, tube_growth=0.0, n_components=18,
            regime="sparse transition to branch point",
        ),
    ]

    for label, az, curve, state_curve in [
        ("A", 90.0, (0.08, 0.00, 0.02), (0.34, 0.08, 0.12)),
        ("B", 210.0, (-0.05, 0.07, 0.02), (-0.24, 0.26, 0.10)),
        ("C", 330.0, (-0.05, -0.07, 0.02), (-0.24, -0.26, 0.10)),
    ]:
        state_center = endpoint(az)
        specs.extend([
            BranchSpec(
                name=f"sparse transition arm {label}",
                start=branch_point, end=state_center,
                start_time=0.42, end_time=0.90,
                samples_scale=0.08,
                density_baseline=0.16,
                curve=curve,
                tube_scale=0.42, tube_growth=0.0, n_components=16,
                regime="sparse transition from branch point to dense state",
            ),
            BranchSpec(
                name=f"branch state {label}",
                start=state_center, end=state_center,
                start_time=0.90, end_time=1.00,
                samples_scale=0.62,
                density_baseline=1.0,
                curve=state_curve,
                tube_scale=0.86, tube_growth=0.15, n_components=14,
                regime="sparse GMM-like terminal state cloud",
            ),
        ])
    return specs


def build_dense_states_hierarchy_specs() -> list[BranchSpec]:
    """Nested lineage in the same dense-state / sparse-transition style.

    Topology: root -> {A, B}; B -> {B1, B2}. Three dense terminal states where
    B1/B2 are SISTERS (share most of their path through the second branch point)
    and A is an OUTGROUP that diverges at the first branch point. Same density
    regime as the star variant (dense root + dense terminal clusters joined by
    sparse transitions), so the only difference from the star is the lineage
    topology -- which is exactly what we want to compare for Figure 1.
    """
    root = (0.0, 0.0, 0.0)
    p1 = (0.0, 0.0, 1.45)          # first branch point
    p2_az = 200.0

    def direction(az_deg: float, tilt_deg: float = 48.0) -> np.ndarray:
        az, tilt = np.deg2rad(az_deg), np.deg2rad(tilt_deg)
        return np.array([np.sin(tilt) * np.cos(az),
                         np.sin(tilt) * np.sin(az),
                         np.cos(tilt)])

    def endpoint(start, az_deg: float, length: float) -> tuple[float, float, float]:
        return tuple(np.asarray(start) + length * direction(az_deg))

    p2 = endpoint(p1, p2_az, 1.25)             # second branch point (down branch B)

    specs = [
        BranchSpec(
            name="dense root state",
            start=root, end=root,
            start_time=0.00, end_time=0.04,
            samples_scale=1.55,
            density_baseline=10.0,
            tube_scale=0.32, tube_growth=0.0, n_components=8,
            regime="compact dense root state",
        ),
        BranchSpec(
            name="sparse transition trunk",
            start=root, end=p1,
            start_time=0.04, end_time=0.36,
            samples_scale=0.09,
            density_baseline=0.18,
            tube_scale=0.40, tube_growth=0.0, n_components=18,
            regime="sparse transition to first branch point",
        ),
    ]

    # outgroup A: diverges at the first branch point p1
    a_center = endpoint(p1, 20.0, 2.35)
    specs.extend([
        BranchSpec(
            name="sparse transition arm A",
            start=p1, end=a_center,
            start_time=0.36, end_time=0.88,
            samples_scale=0.08,
            density_baseline=0.16,
            curve=(0.10, 0.02, 0.04),
            tube_scale=0.42, tube_growth=0.0, n_components=16,
            regime="sparse transition (outgroup A)",
        ),
        BranchSpec(
            name="branch state A",
            start=a_center, end=a_center,
            start_time=0.88, end_time=1.00,
            samples_scale=0.62,
            density_baseline=1.0,
            curve=(0.30, 0.06, 0.10),
            tube_scale=0.86, tube_growth=0.15, n_components=14,
            regime="dense terminal state (outgroup A)",
        ),
    ])

    # internal bridge to the second branch point p2, then sisters B1 / B2
    specs.append(
        BranchSpec(
            name="sparse transition bridge B",
            start=p1, end=p2,
            start_time=0.36, end_time=0.60,
            samples_scale=0.07,
            density_baseline=0.16,
            tube_scale=0.42, tube_growth=0.0, n_components=14,
            regime="sparse internal bridge (shared B lineage)",
        )
    )
    for label, az, curve, state_curve in [
        ("B1", 235.0, (-0.05, 0.07, 0.02), (-0.24, 0.26, 0.10)),
        ("B2", 165.0, (-0.05, -0.07, 0.02), (-0.24, -0.26, 0.10)),
    ]:
        s_center = endpoint(p2, az, 1.7)
        specs.extend([
            BranchSpec(
                name=f"sparse transition arm {label}",
                start=p2, end=s_center,
                start_time=0.60, end_time=0.90,
                samples_scale=0.07,
                density_baseline=0.16,
                curve=curve,
                tube_scale=0.42, tube_growth=0.0, n_components=14,
                regime=f"sparse transition (sister {label})",
            ),
            BranchSpec(
                name=f"branch state {label}",
                start=s_center, end=s_center,
                start_time=0.90, end_time=1.00,
                samples_scale=0.62,
                density_baseline=1.0,
                curve=state_curve,
                tube_scale=0.86, tube_growth=0.15, n_components=14,
                regime=f"dense terminal state (sister {label})",
            ),
        ])
    return specs


def build_branch_specs(variant: str) -> list[BranchSpec]:
    if variant == "diverse":
        return build_diverse_branch_specs()
    if variant == "root_dense_sparse_transition":
        return build_root_dense_sparse_transition_specs()
    if variant == "dense_states_sparse_transition":      # flat star (3 sibling states)
        return build_dense_states_sparse_transition_specs()
    if variant == "dense_states_hierarchy":              # nested lineage (sisters + outgroup)
        return build_dense_states_hierarchy_specs()
    raise ValueError(f"Unknown branch3d variant: {variant}")


def local_radius(coords: np.ndarray, k: int = 18) -> np.ndarray:
    tree = cKDTree(coords)
    k_eff = min(k + 1, len(coords))
    dists, _ = tree.query(coords, k=k_eff)
    if dists.ndim == 1:
        return np.zeros(len(coords), dtype=np.float64)
    return dists[:, 1:].mean(axis=1)


def generate_tree(
    branch_specs: list[BranchSpec] | None = None,
    variant: str = "diverse",
    samples_per_branch: int = 2000,
    tube_sigma: float = 0.06,
    blob_sigma: float = 0.16,
    seed: int = 42,
    knn_k: int = 18,
) -> TreeDensityDemo:
    rng = np.random.default_rng(seed)
    specs = branch_specs or build_branch_specs(variant)

    coords_list, branch_id_list, branch_name_list = [], [], []
    progress_list, pseudotime_list = [], []

    for branch_id, spec in enumerate(specs):
        n = max(40, int(round(samples_per_branch * spec.samples_scale)))
        coords, progress, pseudotime = _branch_points_gmm(spec, n, blob_sigma, rng)
        coords_list.append(coords)
        branch_id_list.append(np.full(n, branch_id, dtype=np.int64))
        branch_name_list.append(np.full(n, spec.name, dtype=object))
        progress_list.append(progress)
        pseudotime_list.append(pseudotime)

    coords = np.vstack(coords_list)
    branch_id = np.concatenate(branch_id_list)
    branch_names = np.concatenate(branch_name_list)
    branch_progress = np.concatenate(progress_list)
    pseudotime = np.concatenate(pseudotime_list)
    radii = local_radius(coords, k=knn_k)
    density = 1.0 / np.clip(radii, np.percentile(radii, 1), None)
    centerlines = [_centerline(spec) for spec in specs]

    return TreeDensityDemo(
        coords=coords.astype(np.float64),
        branch_id=branch_id,
        branch_names=branch_names,
        pseudotime=pseudotime,
        branch_progress=branch_progress,
        local_density=density,
        local_radius=radii,
        centerlines=centerlines,
        branch_specs=specs,
    )


# --------------------------------------------------------------------------- #
# Plot helpers
# --------------------------------------------------------------------------- #
def _set_axes_equal(ax) -> None:
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    ranges = [abs(x_limits[1] - x_limits[0]),
              abs(y_limits[1] - y_limits[0]),
              abs(z_limits[1] - z_limits[0])]
    centers = [np.mean(x_limits), np.mean(y_limits), np.mean(z_limits)]
    radius = 0.5 * max(ranges)
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])


def _style_3d_axis(ax) -> None:
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("developmental axis")
    ax.grid(False)
    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)
    _set_axes_equal(ax)


def _plot_centerlines(ax, data: TreeDensityDemo, color="black", lw=1.8) -> None:
    for line in data.centerlines:
        ax.plot(line[:, 0], line[:, 1], line[:, 2], color=color, lw=lw, alpha=0.8)


def _density_marker_sizes(density, min_size=2.0, max_size=24.0) -> np.ndarray:
    values = np.log1p(density)
    lo, hi = np.percentile(values, [5, 98])
    scaled = np.clip((values - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    return min_size + (max_size - min_size) * scaled


def _annotate_tree(ax, data: TreeDensityDemo) -> None:
    """Mark the root and the branch points so the topology reads instantly."""
    root = _as_array(data.branch_specs[0].start)
    ax.scatter(*root, s=120, c="#d62728", edgecolor="black", linewidth=0.8, depthshade=False)
    ax.text(root[0], root[1], root[2] - 0.6, "dense root", fontsize=10, color="#b22222",
            ha="center", weight="bold")
    # branch points = where >1 spec shares a start, plus first-trunk end
    starts = [tuple(np.round(s.start, 3)) for s in data.branch_specs]
    for bp in {s for s in starts if starts.count(s) > 1}:
        bp = _as_array(bp)
        ax.scatter(*bp, s=70, c="white", edgecolor="black", linewidth=1.0, depthshade=False)


def _dataset_title(data: TreeDensityDemo) -> str:
    names = {str(name) for name in np.asarray(data.branch_names).reshape(-1)}
    if any(name.startswith("branch state") for name in names):
        return "Synthetic 3D branching tree: dense root with sparse GMM states and sparse transitions"
    if any(name.startswith("transition") for name in names):
        return "Synthetic 3D branching tree: dense root + sparse transitions + sparse terminal states"
    return ("Synthetic 3D branching tree: clear topology + diverse density "
            "(dense root, branches thinning to sparse tips of different density)")


BRANCH_CMAP = "tab10"


def branch_colors(branch_id) -> np.ndarray:
    """Fixed, discrete per-point colors keyed by integer branch id.

    Branch i always maps to tab10 swatch i (mod 10), identically in the 3D-truth
    plot, the 2D-baseline grid, and the PaperEmbeddingCallback -- so a state has
    the SAME colour in the original 3D space and in every 2D embedding.
    """
    ids = np.asarray(branch_id, dtype=int)
    return plt.get_cmap(BRANCH_CMAP)(ids % 10)


def _draw_one_3d(ax, data, *, color_by, view):
    order = np.argsort(data.local_density)
    if color_by == "branch":
        ax.scatter(data.coords[:, 0], data.coords[:, 1], data.coords[:, 2],
                   c=branch_colors(data.branch_id), s=7, alpha=0.65, linewidths=0,
                   depthshade=False)
        sc = None
        ax.set_title("colored by branch (topology)", fontsize=13)
    else:
        sizes = _density_marker_sizes(data.local_density, 2.0, 34.0)
        sc = ax.scatter(data.coords[order, 0], data.coords[order, 1], data.coords[order, 2],
                        c=np.log1p(data.local_density[order]), cmap="magma",
                        s=sizes[order], alpha=0.78, linewidths=0, depthshade=False)
        ax.set_title("colored by local density", fontsize=13)
    _plot_centerlines(ax, data, color="#333333", lw=2.4)
    _annotate_tree(ax, data)
    ax.view_init(elev=view[0], azim=view[1])
    _style_3d_axis(ax)
    return sc


def save_3d_truth(data: TreeDensityDemo, stem: Path) -> Path:
    fig = plt.figure(figsize=(15.5, 7.4))
    _draw_one_3d(fig.add_subplot(121, projection="3d"), data,
                 color_by="branch", view=(14, -62))
    ax = fig.add_subplot(122, projection="3d")
    sc = _draw_one_3d(ax, data, color_by="density", view=(14, -62))
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.10)
    cbar.set_label("log local density")
    fig.suptitle(_dataset_title(data), fontsize=14)
    out = stem.parent / f"{stem.name}_3d_truth.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #
def _standardize_xy(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.median(x, axis=0, keepdims=True)
    scale = np.percentile(np.linalg.norm(x, axis=1), 95)
    return x / max(scale, 1e-12)


def run_baselines(
    data: TreeDensityDemo,
    seed: int = 42,
    tsne_perplexity: float = 45.0,
    umap_neighbors: int = 30,
    dens_lambda: float = 0.1,
    skip: tuple[str, ...] = (),
) -> dict[str, np.ndarray]:
    coords = data.coords
    embeddings: dict[str, np.ndarray] = {}

    def _try(name, fn):
        if name in skip:
            return
        t0 = time.time()
        try:
            emb = fn()
            embeddings[name] = _standardize_xy(emb)
            print(f"  [{name}] ok ({time.time() - t0:.1f}s)")
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  [{name}] FAILED: {exc}")

    def _pca():
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=seed).fit_transform(coords)

    # t-SNE and densNE are run from the SAME den-SNE binary with identical
    # hyperparameters (perplexity, theta, iterations, early-exaggeration, no PCA,
    # and the same random seed => same initialization). They differ ONLY in the
    # density-preservation weight: t-SNE uses dens_lambda=0 (which makes den-SNE
    # reduce to standard Barnes-Hut t-SNE), densNE uses dens_lambda>0. This keeps
    # the comparison fair -- any difference is due purely to the density term.
    def _densne_family(lam: float):
        from tools.densne_win import run_densne_nofork
        emb = run_densne_nofork(
            coords.astype(np.float64), no_dims=2, perplexity=tsne_perplexity,
            theta=0.5, randseed=seed, use_pca=False, max_iter=1000,
            dens_frac=0.3, dens_lambda=lam, final_dens=False,
            early_exaggeration=12.0,
        )
        return np.asarray(emb)

    def _tsne():
        return _densne_family(0.0)        # den-SNE with no density term == t-SNE

    def _densne():
        return _densne_family(dens_lambda)  # same settings, density term ON

    def _umap():
        import umap
        return umap.UMAP(n_components=2, n_neighbors=umap_neighbors,
                         min_dist=0.1, random_state=seed).fit_transform(coords)

    def _densmap():
        import umap
        return umap.UMAP(n_components=2, n_neighbors=umap_neighbors, min_dist=0.1,
                         densmap=True, random_state=seed).fit_transform(coords)

    def _phate():
        import phate
        return phate.PHATE(n_components=2, knn=umap_neighbors, random_state=seed,
                           verbose=False).fit_transform(coords)

    _try("PCA", _pca)
    _try("t-SNE", _tsne)
    _try("densNE", _densne)   # matched pair next to t-SNE for direct comparison
    _try("UMAP", _umap)
    _try("densMAP", _densmap)
    _try("PHATE", _phate)
    return embeddings


def density_preservation_score(data: TreeDensityDemo, emb: np.ndarray, k: int = 18) -> float:
    """Spearman rho between original and embedded local density (higher=better)."""
    emb_density = 1.0 / np.clip(local_radius(emb, k=k),
                                np.percentile(local_radius(emb, k=k), 1), None)
    rho, _ = spearmanr(data.local_density, emb_density)
    return float(rho)


def save_baselines_figure(
    data: TreeDensityDemo,
    embeddings: dict[str, np.ndarray],
    stem: Path,
    knn_k: int = 18,
) -> tuple[Path, dict[str, float]]:
    names = list(embeddings.keys())
    ncols = len(names)
    order = np.argsort(data.local_density)
    density_color = np.log1p(data.local_density)
    sizes = _density_marker_sizes(data.local_density, 2.0, 16.0)

    scores = {n: density_preservation_score(data, embeddings[n], knn_k) for n in names}

    fig, axes = plt.subplots(2, ncols, figsize=(3.3 * ncols, 6.8), squeeze=False)
    for col, name in enumerate(names):
        emb = embeddings[name]
        ax = axes[0, col]
        ax.scatter(emb[:, 0], emb[:, 1], c=branch_colors(data.branch_id),
                   s=4, alpha=0.6, linewidths=0)
        ax.set_title(name, fontsize=13)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        if col == 0:
            ax.set_ylabel("colored by branch", fontsize=11)

        ax = axes[1, col]
        sc = ax.scatter(emb[order, 0], emb[order, 1], c=density_color[order],
                        cmap="magma", s=sizes[order], alpha=0.78, linewidths=0)
        ax.set_title(f"density rho = {scores[name]:+.2f}", fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        if col == 0:
            ax.set_ylabel("colored by 3D density", fontsize=11)

    cbar = fig.colorbar(sc, ax=axes[1, :], shrink=0.8, pad=0.01)
    cbar.set_label("log original local density")
    fig.suptitle(
        "2D baseline embeddings of the diverse-density 3D branching tree\n"
        "(top: branch identity / topology;  bottom: ground-truth density preservation)",
        fontsize=14,
    )
    out = stem.parent / f"{stem.name}_baselines_2d.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)

    np.savez(stem.parent / f"{stem.name}_baseline_embeddings.npz",
             **{n.replace("-", "_"): e for n, e in embeddings.items()})
    return out, scores


# --------------------------------------------------------------------------- #
# IO + reporting
# --------------------------------------------------------------------------- #
def save_dataset(data: TreeDensityDemo, stem: Path) -> tuple[Path, Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    mat_path = stem.with_suffix(".mat")
    npz_path = stem.with_suffix(".npz")
    labels = (data.branch_id + 1).reshape(-1, 1).astype(np.int64)
    coords = data.coords.astype(np.float32)
    savemat(mat_path, {
        "M": coords, "C": labels, "X": coords, "Y": labels,
        "pseudotime": data.pseudotime.reshape(-1, 1).astype(np.float32),
        "local_density": data.local_density.reshape(-1, 1).astype(np.float32),
        "local_radius": data.local_radius.reshape(-1, 1).astype(np.float32),
    })
    np.savez(npz_path,
             data=data.coords, clusters=data.branch_id,
             branch_names=data.branch_names.astype(str),
             pseudotime=data.pseudotime, branch_progress=data.branch_progress,
             density=data.local_density, local_radius=data.local_radius,
             centerlines=np.asarray(data.centerlines, dtype=object))
    return mat_path, npz_path


def write_report(data: TreeDensityDemo, scores: dict[str, float], stem: Path) -> Path:
    lines = []
    lines.append(_dataset_title(data))
    lines.append("=" * 50)
    lines.append(f"points     : {len(data.coords)}")
    lines.append(f"dimensions : {data.coords.shape[1]}")
    lines.append(f"branches   : {len(data.branch_specs)}")
    root = data.local_density[data.pseudotime <= 0.20].mean()
    tip = data.local_density[data.pseudotime >= 0.90].mean()
    lines.append(f"root/tip mean density ratio : {root / max(tip, 1e-9):.1f}x")
    dmin, dmax = np.percentile(data.local_density, [2, 98])
    lines.append(f"density spread (p2..p98)    : {dmin:.2f} .. {dmax:.2f} ({dmax / max(dmin, 1e-9):.1f}x)")
    lines.append("")
    lines.append("Per-branch density regimes:")
    for idx, spec in enumerate(data.branch_specs):
        n = int(np.sum(data.branch_id == idx))
        med = np.median(data.local_density[data.branch_id == idx])
        lines.append(f"  {idx + 1} {spec.name:14s} n={n:4d}  med_density={med:6.2f}  [{spec.regime}]")
    lines.append("")
    lines.append("Density preservation (Spearman rho between 3D and 2D local density;")
    lines.append("higher = better; density-preserving methods should score highest):")
    for name, rho in sorted(scores.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {name:10s} rho = {rho:+.3f}")
    report = "\n".join(lines)
    out = stem.parent / f"{stem.name}_density_preservation_report.txt"
    out.write_text(report, encoding="utf-8")
    print("\n" + report + "\n")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("diverse", "root_dense_sparse_transition",
                 "dense_states_sparse_transition", "dense_states_hierarchy"),
        default="diverse",
        help="Synthetic branch3d geometry/density variant to generate.",
    )
    parser.add_argument(
        "--save-path", type=Path,
        default=None,
        help="Output path stem. Extensions/suffixes are added automatically.",
    )
    parser.add_argument("--samples-per-branch", type=int, default=2000)
    parser.add_argument("--tube-sigma", type=float, default=0.06)
    parser.add_argument("--blob-sigma", type=float, default=0.16,
                        help="Base std of the Gaussian blobs strung along each branch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--knn-k", type=int, default=18)
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Baseline names to skip, e.g. --skip PHATE densNE")
    parser.add_argument("--tsne-perplexity", type=float, default=45.0)
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--dens-lambda", type=float, default=0.1,
                        help="densNE density weight; t-SNE is the same run with dens_lambda=0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.save_path is None:
        stem_name = {
            "diverse": "branch3d_diverse_density",
            "root_dense_sparse_transition": "branch3d_root_dense_sparse_transition",
            "dense_states_sparse_transition": "branch3d_dense_states_sparse_transition",
            "dense_states_hierarchy": "branch3d_dense_states_hierarchy",
        }[args.variant]
        args.save_path = _DEFAULT_PAPER_SAVE_DIR / stem_name
    data = generate_tree(
        variant=args.variant,
        samples_per_branch=args.samples_per_branch,
        tube_sigma=args.tube_sigma, blob_sigma=args.blob_sigma,
        seed=args.seed, knn_k=args.knn_k,
    )
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    mat_path, npz_path = save_dataset(data, args.save_path)
    print(f"saved dataset: {mat_path}")
    print(f"saved dataset: {npz_path}")
    truth = save_3d_truth(data, args.save_path)
    print(f"saved 3D truth figure: {truth}")

    scores: dict[str, float] = {}
    if not args.skip_baselines:
        print("running baselines:")
        embeddings = run_baselines(
            data, seed=args.seed, tsne_perplexity=args.tsne_perplexity,
            umap_neighbors=args.umap_neighbors, dens_lambda=args.dens_lambda,
            skip=tuple(args.skip),
        )
        fig_path, scores = save_baselines_figure(data, embeddings, args.save_path, args.knn_k)
        print(f"saved baselines figure: {fig_path}")
    write_report(data, scores, args.save_path)


if __name__ == "__main__":
    main()
