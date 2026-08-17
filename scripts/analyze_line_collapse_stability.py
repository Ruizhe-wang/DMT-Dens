"""Standalone loss-level analysis of the 2D->line collapse in the ablation runs.

Replicates the exact manifold / density losses from
``model/DiffTreeVQ_density.py`` (loss_type=G, differentiable_global
standardization, floors 1e-4) without importing Lightning, then measures, at a
low-loss 1D embedding of each mechanism condition:

1. criticality: |dL/dz2| at z2 = 0 (should be ~0 for any distance-only loss,
   proving the line is a critical set of the objective);
2. curvature:  d^2 L / d eps^2 for z2 = eps * v, eps -> 0.  Positive = the
   line is a local minimum (collapse attractive); negative = the line is
   unstable (2D spread reduces the loss);
3. over-weighting: fraction of pairs with Q > P (repulsive gradient weights),
   the mechanism that breaks the line minimum;
4. density-loss curvature at the line (does the density term rescue rank?);
5. embedding-level GD simulation from the 2D PCA start: does the loss itself
   drive the second axis to zero (base) or keep it alive (full)?

Output: JSON + a printed table, written to tmp/line_collapse_stability.json.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch

SEED = 42
BATCH = 2048
NU = 0.01
T_MUL = 1.0  # num_use_mlevel_list[i] == 1; the temperature=0.2 arg is unused by loss_type=G
EXP_B = 0.4
EXP_MIN = 1e-6
HARDPAIR_K = 100
DENSITY_K = 12
DENSITY_ANCHORS = 512
DENSITY_WEIGHT = 0.0018
FLOOR = 1e-4
GD_STEPS = 150
GD_LR = 0.02


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------
# exact replication of the loss pieces (model/DiffTreeVQ_density.py)
# --------------------------------------------------------------------------

def standardize_embedding(embedding):
    embedding_fp32 = embedding.float()
    centered = embedding_fp32 - embedding_fp32.mean(dim=0, keepdim=True)
    mean_square = centered.square().mean()
    scale = mean_square.clamp_min(FLOOR**2).sqrt()
    return centered / scale


def t_distribution_similarity(distance_matrix, df):
    distance_matrix = distance_matrix.float().clamp_min(0.0) + 1e-9
    numerator = (1.0 + distance_matrix.square() / float(df)).pow(
        -(float(df) + 1.0) / 2.0
    )
    off_diagonal = numerator.clone()
    off_diagonal.fill_diagonal_(0.0)
    denominator = off_diagonal.sum(dim=1, keepdim=True)
    denominator = denominator.clamp_min(torch.finfo(numerator.dtype).tiny)
    return numerator / denominator


def cal_dis_to_p(dis, exp_b=EXP_B):
    batch_size = dis.size(0)
    sorted_indices = torch.argsort(dis, dim=1)
    ranks = torch.zeros_like(dis, dtype=torch.long)
    order = torch.arange(batch_size, device=dis.device).expand_as(dis)
    ranks.scatter_(1, sorted_indices, order)
    logb = math.log(exp_b)
    return torch.exp(ranks.float() * logb)


def cal_dis_to_p_distance(dis):
    bandwidth = dis.mean(dim=1, keepdim=True) + 1e-8
    return torch.exp(-dis / bandwidth)


def row_normalize_off_diagonal_affinity(affinity):
    affinity = affinity.float()
    diagonal = torch.diagonal(affinity).clone()
    off_diagonal = affinity.clone()
    off_diagonal.fill_diagonal_(0.0)
    denominator = off_diagonal.sum(dim=1, keepdim=True).clamp_min(
        torch.finfo(off_diagonal.dtype).tiny
    )
    normalized = off_diagonal / denominator
    normalized.diagonal().copy_(diagonal)
    return normalized


def bernoulli_pair_loss(P, Q):
    eps = 1e-8
    q_positive = Q + eps
    q_negative = Q.clone()
    q_negative.fill_diagonal_(0.0)
    q_negative = q_negative.clamp(min=0.0, max=1.0)
    positive = P * torch.log(q_positive)
    negative = (1.0 - P) * torch.log(1.0 - q_negative + eps)
    return -(positive + negative)


def knn_log_density(query, reference, k):
    dist = torch.cdist(query.float(), reference.float())
    k_fetch = min(k + 1, dist.shape[1])
    rk = torch.topk(dist, k=k_fetch, dim=1, largest=False).values[:, -1]
    return -torch.log(rk.clamp_min(FLOOR))


def pearson_correlation_loss(x, y):
    xc = x - x.mean()
    yc = y - y.mean()
    denominator = (xc.norm() * yc.norm()).clamp_min(FLOOR)
    r = (xc * yc).sum() / denominator
    return 1.0 - r


def build_target(view_a, view_b, affinity, symmetric, row_normalize=False):
    """Detached high-dim affinity target P (exact replication)."""
    with torch.no_grad():
        dis_input_ab = torch.cdist(view_a, view_b).square()
        dis_input_ab.fill_diagonal_(0)
        if affinity == "rank":
            P_x = cal_dis_to_p(dis_input_ab)
            P_y = cal_dis_to_p(dis_input_ab.T).T
        elif affinity == "distance":
            P_x = cal_dis_to_p_distance(dis_input_ab)
            P_y = cal_dis_to_p_distance(dis_input_ab.T).T
        else:
            raise ValueError(affinity)
        if symmetric == "bidirectional":
            P = torch.sqrt(P_x * P_y)
        elif symmetric == "unidirectional":
            P = P_x
        else:
            raise ValueError(symmetric)
        P[P < EXP_MIN] = EXP_MIN
        if row_normalize:
            P = row_normalize_off_diagonal_affinity(P)
        return P


def manifold_loss(view_a, view_b, embedding, affinity, symmetric,
                  hardpair, row_normalize=False):
    b = view_a.shape[0]
    # The model standardizes the full 2B-row embedding, then splits views.
    emb_full = torch.cat([embedding, embedding], dim=0)
    z = standardize_embedding(emb_full)
    features_a, features_b = z[:b], z[b:]
    dis_ab = torch.cdist(features_a, features_b) * T_MUL
    Q = t_distribution_similarity(dis_ab, df=NU)
    P = build_target(view_a, view_b, affinity, symmetric, row_normalize)
    loss = bernoulli_pair_loss(P.float(), Q)
    if hardpair:
        with torch.no_grad():
            k_safe = min(HARDPAIR_K, loss.shape[1])
            topk_values, _ = torch.topk(loss, k=k_safe, dim=1)
            threshold = topk_values[:, -1].unsqueeze(1)
            mask = loss >= threshold
        loss = loss[mask]
    return loss.mean()


def density_loss(view_a, embedding, scale_mode):
    b = view_a.shape[0]
    emb_full = torch.cat([embedding, embedding], dim=0)
    z = standardize_embedding(emb_full)[:b]
    flat_batch = view_a.reshape(b, -1)
    k = min(DENSITY_K, b - 1)
    if scale_mode == "single":
        k_scales = [k]
    elif scale_mode == "multi":
        k_small = max(1, k // 2)
        k_scales = [k_small, k] if k_small != k else [k]
    else:
        raise ValueError(scale_mode)
    anchor_idx = torch.randperm(b, device=view_a.device)[:DENSITY_ANCHORS]
    anchor_hd = flat_batch[anchor_idx]
    log_hd = [knn_log_density(anchor_hd, flat_batch, k_s).detach() for k_s in k_scales]
    anchor_ld = z[anchor_idx]
    terms = [
        pearson_correlation_loss(knn_log_density(anchor_ld, z, k_s), hd)
        for k_s, hd in zip(k_scales, log_hd)
    ]
    return torch.stack(terms).mean()


def total_loss(view_a, view_b, embedding, cond, density_scale_mode=None):
    loss = manifold_loss(
        view_a, view_b, embedding,
        affinity=cond["affinity"],
        symmetric=cond["symmetric"],
        hardpair=cond["hardpair"],
        row_normalize=cond.get("row_normalize", False),
    )
    if density_scale_mode is not None:
        loss = loss + DENSITY_WEIGHT * density_loss(view_a, embedding, density_scale_mode)
    return loss


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def load_mnist(path, n=BATCH):
    import gzip
    with gzip.open(os.path.join(path, "train-images-idx3-ubyte.gz"), "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 784)
    rng = np.random.RandomState(SEED)
    data = data[rng.choice(data.shape[0], n, replace=False)].astype(np.float32) / 255.0
    return data


def synthetic_curve(n=BATCH):
    rng = np.random.RandomState(SEED)
    t = np.sort(rng.uniform(0, 2 * np.pi, n))
    x = np.stack([
        np.sin(t), np.cos(1.5 * t), np.sin(2.0 * t) / 2.0,
        np.cos(3.0 * t) / 2.0, np.sin(4.0 * t) / 3.0, np.cos(5.0 * t) / 3.0,
    ], axis=1)
    x = np.pad(x, ((0, 0), (0, 34)))
    return (x + 0.1 * rng.randn(n, 40)).astype(np.float32)


def synthetic_surface(n=BATCH):
    rng = np.random.RandomState(SEED)
    g = int(math.isqrt(n))
    u, v = np.meshgrid(np.linspace(-1, 1, g), np.linspace(-1, 1, g))
    x = np.stack([
        u.ravel() * 0.7, v.ravel() * 0.7,
        (u * v).ravel(), np.cos(2 * u).ravel(), np.sin(2 * v).ravel(),
    ], axis=1)
    x = np.pad(x, ((0, 0), (0, 35)))
    return (x + 0.05 * rng.randn(g * g, 40)).astype(np.float32)


def make_views(data, jitter=0.02):
    x = torch.from_numpy(data)
    j = torch.randn_like(x) * jitter
    return x, torch.clamp(x + j, 0.0, 1.0 if data.max() <= 1.0 else float("inf"))


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------

def pca_directions(data):
    centered = data - data.mean(dim=0, keepdim=True)
    _, _, vt = torch.linalg.svd(centered.float(), full_matrices=False)
    v1 = vt[0] / vt[0].norm()
    v2 = vt[1] / vt[1].norm()
    return (centered.float() @ v1).unsqueeze(1), (centered.float() @ v2).unsqueeze(1)


def axis_ratio_std(z_std):
    cov = z_std.T @ z_std / max(1, z_std.shape[0])
    eig = torch.linalg.eigvalsh(cov).clamp_min(0.0)
    return (eig[0].sqrt() / eig[-1].sqrt().clamp_min(1e-12)).item()


def fit_quadratic(eps_grid, values):
    """Least-squares quadratic a*e^2 + b*e + c; returns (a, b, c)."""
    eps_grid = np.asarray(eps_grid, dtype=np.float64)
    A = np.vstack([eps_grid**2, eps_grid, np.ones_like(eps_grid)]).T
    coef, *_ = np.linalg.lstsq(A, np.asarray(values), rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2])


def curvature_probe(view_a, view_b, z_line, cond, density_scale_mode,
                    v2, delta=0.05, label=""):
    """dL/dz2 at the line, and d^2L/de^2 along z2 = eps * v2."""
    out = {}
    z_lin = torch.zeros_like(z_line)
    z_lin[:, 0] = z_line[:, 0]
    with torch.no_grad():
        z_lin0 = z_lin.clone()
    z_lin0.requires_grad_(True)
    loss0 = total_loss(view_a, view_b, z_lin0, cond, density_scale_mode)
    grad = torch.autograd.grad(loss0, z_lin0)[0]
    out["grad_norm_z2_at_line"] = float(grad[:, 1].norm())
    out["loss_at_line"] = float(loss0.detach())

    eps_vals = []
    for e in [delta, -delta, 2 * delta, -2 * delta]:
        z_t = z_lin.detach().clone()
        z_t[:, 1] = (e * v2).reshape(-1)
        with torch.no_grad():
            z_t = z_t.detach().clone()
        z_t.requires_grad_(True)
        L = total_loss(view_a, view_b, z_t, cond, density_scale_mode)
        eps_vals.append((e, float(L.detach())))
        del z_t
    curv2 = (eps_vals[0][1] + eps_vals[1][1] - 2 * out["loss_at_line"]) / delta**2
    a, b, _ = fit_quadratic([e for e, _ in eps_vals], [v for _, v in eps_vals])
    out["curvature_delta"] = float(curv2)
    out["curvature_quadfit"] = a
    out["curvature_linear"] = b
    return out


def overweight_fraction(view_a, view_b, z_line, cond):
    b = view_a.shape[0]
    z = standardize_embedding(torch.cat([z_line, z_line], dim=0))
    with torch.no_grad():
        dis_ab = torch.cdist(z[:b], z[b:]) * T_MUL
        Q = t_distribution_similarity(dis_ab, df=NU)
        P = build_target(view_a, view_b, cond["affinity"], cond["symmetric"],
                         cond.get("row_normalize", False))
        over = (Q > P).float()
        over.fill_diagonal_(0.0)
        mean_over = over.sum(dim=1).mean() / (P.shape[1] - 1)
        row_max = over.max(dim=1).values.mean()
        return {"pair_fraction_overweight": float(mean_over),
                "row_fraction_with_overweight": float(row_max)}


def run_gd_simulation(view_a, view_b, cond, density_scale_mode, z1, z2, label=""):
    """Minimize the condition's loss from the 2D PCA start; track axis ratio."""
    z = torch.cat([z1, z2], dim=1).detach().clone()
    z.requires_grad_(True)
    opt = torch.optim.Adam([z], lr=GD_LR)
    traj = []
    for step in range(GD_STEPS + 1):
        L = total_loss(view_a, view_b, z, cond, density_scale_mode)
        if step % 25 == 0:
            with torch.no_grad():
                zs = standardize_embedding(
                    torch.cat([z.detach(), z.detach()], dim=0))[:z.shape[0]]
                traj.append({"step": step, "loss": float(L.detach()),
                             "axis_ratio": axis_ratio_std(zs),
                             "std_axis0": float(zs[:, 0].std()),
                             "std_axis1": float(zs[:, 1].std())})
        opt.zero_grad()
        L.backward()
        opt.step()
    return traj


CONDITIONS = {
    "base": dict(affinity="distance", symmetric="unidirectional", hardpair=False),
    "b1_rank": dict(affinity="rank", symmetric="unidirectional", hardpair=False),
    "b2_bidir": dict(affinity="distance", symmetric="bidirectional", hardpair=False),
    "b3_hardpair": dict(affinity="distance", symmetric="unidirectional", hardpair=True),
    "b4_single_density": dict(affinity="distance", symmetric="unidirectional",
                              hardpair=False, density="single"),
    "b5_multi_density": dict(affinity="distance", symmetric="unidirectional",
                             hardpair=False, density="multi"),
    "full": dict(affinity="rank", symmetric="bidirectional", hardpair=True,
                 density="multi"),
    "base_prownorm": dict(affinity="distance", symmetric="unidirectional",
                          hardpair=False, row_normalize=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mnist-root", default="/usr/storage/ruizhe/mldr_rz/data/MNIST/raw")
    ap.add_argument("--no-mnist", action="store_true")
    ap.add_argument("--no-gd", action="store_true")
    ap.add_argument("--out", default="tmp/line_collapse_stability.json")
    args = ap.parse_args()

    set_seed(SEED)
    datasets = {}
    if not args.no_mnist and os.path.isdir(args.mnist_root):
        datasets["MNIST"] = load_mnist(args.mnist_root)
    datasets["curve_synthetic"] = synthetic_curve()
    datasets["surface_synthetic"] = synthetic_surface()

    report = {"meta": {"batch": BATCH, "nu": NU, "t_mul": T_MUL,
                       "exp_b": EXP_B, "exp_min": EXP_MIN, "hardpair_k": HARDPAIR_K,
                       "density_weight": DENSITY_WEIGHT, "gd_steps": GD_STEPS},
              "datasets": {}}

    for dname, data in datasets.items():
        view_a, view_b = make_views(data)
        z1, z2 = pca_directions(view_a)
        ds = {}
        for cname, cond in CONDITIONS.items():
            entry = {}
            dens = cond.get("density")
            set_seed(SEED)  # deterministic density anchor subsets per condition
            # 1D optimization to a low-loss line (z2 column frozen at zero)
            z_lin = torch.cat([z1, torch.zeros_like(z2)], dim=1).detach().clone()
            z_lin.requires_grad_(True)
            opt = torch.optim.Adam([z_lin], lr=GD_LR)
            for _ in range(GD_STEPS):
                L = total_loss(view_a, view_b, z_lin, cond, dens)
                opt.zero_grad()
                L.backward()
                opt.step()
            entry["line_opt_final_loss"] = float(total_loss(
                view_a, view_b, z_lin.detach(), cond, dens).item())

            entry["curvature_pc2"] = curvature_probe(
                view_a, view_b, z_lin.detach(), cond, dens, z2)
            rng = torch.Generator().manual_seed(SEED)
            v_rand = torch.randn(view_a.shape[0], generator=rng).unsqueeze(1)
            v_rand = v_rand / v_rand.norm()
            entry["curvature_random"] = curvature_probe(
                view_a, view_b, z_lin.detach(), cond, dens, v_rand)
            entry["overweight"] = overweight_fraction(
                view_a, view_b, z_lin.detach(), cond)

            if not args.no_gd and cname in ("base", "b3_hardpair", "full",
                                            "base_prownorm"):
                entry["gd_trajectory"] = run_gd_simulation(
                    view_a, view_b, cond, dens, z1, z2)
            ds[cname] = entry
            print(f"[{dname}] {cname} done", flush=True)
        report["datasets"][dname] = ds

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=1, default=float)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
