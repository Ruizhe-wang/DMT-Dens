"""Extended embedding-level GD from the 2D PCA start for key conditions.

Directly asks: does the loss itself drive the second output axis to zero
(base) or keep it alive (full / b1 / b3)?  Tracks the standardized axis
ratio and the cosine between the two embedding columns over many steps.
"""

import json
import sys

import torch

sys.path.insert(0, "/usr/storage/ruizhe/mldr_rz/tmp")

from analyze_line_collapse_stability import (  # noqa: E402
    GD_LR, SEED, CONDITIONS, axis_ratio_std, make_views, pca_directions,
    set_seed, standardize_embedding, total_loss,
)

STEPS = 2000
CHECK = 100
DATASETS = {}
LOADERS = {}


def column_cosine(z):
    c = torch.cosine_similarity(z[:, 0], z[:, 1], dim=0)
    return abs(c.item())


def run(ds_name, cname, data, out):
    view_a, view_b = make_views(data)
    z1, z2 = pca_directions(view_a)
    cond = CONDITIONS[cname]
    dens = cond.get("density")
    z = torch.cat([z1, z2], dim=1).detach().clone()
    z.requires_grad_(True)
    opt = torch.optim.Adam([z], lr=GD_LR)
    traj = []
    for step in range(STEPS + 1):
        L = total_loss(view_a, view_b, z, cond, dens)
        if step % CHECK == 0:
            with torch.no_grad():
                zs = standardize_embedding(torch.cat([z.detach(), z.detach()], 0))[:z.shape[0]]
                traj.append({"step": step, "loss": float(L.detach()),
                             "axis_ratio": axis_ratio_std(zs),
                             "std_axis0": float(zs[:, 0].std()),
                             "std_axis1": float(zs[:, 1].std()),
                             "abs_col_cosine": column_cosine(zs)})
        opt.zero_grad()
        L.backward()
        opt.step()
    out[f"{ds_name}_{cname}"] = traj
    print(f"[{ds_name}] {cname} done, final axis_ratio={traj[-1]['axis_ratio']:.4f}",
          flush=True)
    return out


def main():
    sys.path.insert(0, "/usr/storage/ruizhe/mldr_rz/tmp")
    from analyze_line_collapse_stability import load_mnist, synthetic_curve, synthetic_surface
    set_seed(SEED)
    out = {}
    data_sets = {
        "MNIST": load_mnist("/usr/storage/ruizhe/mldr_rz/data/MNIST/raw"),
        "curve": synthetic_curve(),
    }
    for ds_name, data in data_sets.items():
        for cname in ("base", "full", "b1_rank", "b3_hardpair"):
            out = run(ds_name, cname, data, out)
    with open("/usr/storage/ruizhe/mldr_rz/tmp/long_gd_trajectories.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
