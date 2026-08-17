"""Decompose line-curvature into manifold-only and density-only parts.

Used to check whether the density term really stabilizes the line (negative
curvature) or the negative curvature comes from kNN-kink noise / the
manifold part at the density-optimized line.
"""

import json
import math

import numpy as np
import torch

import sys

sys.path.insert(0, "/usr/storage/ruizhe/mldr_rz/tmp")

from analyze_line_collapse_stability import (  # noqa: E402
    BATCH, CONDITIONS, GD_LR, GD_STEPS, SEED,
    axis_ratio_std, curvature_probe, density_loss, make_views, manifold_loss,
    pca_directions, set_seed, standardize_embedding, total_loss,
)

COND = dict(affinity="distance", symmetric="unidirectional", hardpair=False)


def run(ds_name, data, out):
    view_a, view_b = make_views(data)
    z1, z2 = pca_directions(view_a)
    set_seed(SEED)
    dens = "multi"
    z_lin = torch.cat([z1, torch.zeros_like(z2)], dim=1).detach().clone()
    z_lin.requires_grad_(True)
    opt = torch.optim.Adam([z_lin], lr=GD_LR)
    for _ in range(GD_STEPS):
        L = total_loss(view_a, view_b, z_lin, COND, dens)
        opt.zero_grad()
        L.backward()
        opt.step()
    z_line = z_lin.detach()
    with torch.no_grad():
        L_man0 = manifold_loss(view_a, view_b, z_line, "distance", "unidirectional", False)
        L_den0 = density_loss(view_a, z_line, dens)
    delta = 0.05
    curv_man, curv_den = {}, {}
    for e in (delta, -delta, 2 * delta, -2 * delta):
        z_t = z_line.clone()
        z_t[:, 1] = (e * z2).reshape(-1)
        z_t.requires_grad_(True)
        with torch.no_grad():
            z_t = z_t.detach().clone()
        z_t.requires_grad_(True)
        curv_man[e] = manifold_loss(view_a, view_b, z_t, "distance", "unidirectional", False).item()
        curv_den[e] = density_loss(view_a, z_t, dens).item()
        del z_t
    man_c = (curv_man[delta] + curv_man[-delta] - 2 * L_man0.item()) / delta**2
    den_c = (curv_den[delta] + curv_den[-delta] - 2 * L_den0.item()) / delta**2
    print(f"[{ds_name}] L_man0={L_man0.item():.4f} L_den0={L_den0.item():.4f} "
          f"curv_manifold={man_c:.4f} curv_density={den_c:.4f} "
          f"curv_total={(man_c + 0.0018*den_c):.4f}", flush=True)
    out[ds_name] = {"L_manifold_at_line": L_man0.item(), "L_density_at_line": L_den0.item(),
                    "curv_manifold": man_c, "curv_density": den_c,
                    "curv_total": man_c + 0.0018 * den_c}
    return out


def main():
    sys.path.insert(0, "/usr/storage/ruizhe/mldr_rz/tmp")
    set_seed(SEED)
    from analyze_line_collapse_stability import load_mnist, synthetic_curve, synthetic_surface
    out = {}
    out = run("MNIST", load_mnist("/usr/storage/ruizhe/mldr_rz/data/MNIST/raw"), out)
    out = run("curve_synthetic", synthetic_curve(), out)
    out = run("surface_synthetic", synthetic_surface(), out)
    with open("/usr/storage/ruizhe/mldr_rz/tmp/density_curvature_decomp.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
