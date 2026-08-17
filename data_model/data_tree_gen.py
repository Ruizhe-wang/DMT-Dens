import argparse
from pathlib import Path

import numpy as np
import phate
from scipy.io import savemat


def parse_args():
    parser = argparse.ArgumentParser(description="Generate PHATE tree data and save it as a .mat file.")
    parser.add_argument("--n-dim", type=int, default=1000, help="Feature dimension.")
    parser.add_argument("--n-branch", type=int, default=80, help="Number of branches.")
    parser.add_argument("--branch-length", type=int, default=1000, help="Samples per branch.")
    parser.add_argument("--rand-multiplier", type=float, default=2.0, help="PHATE tree rand_multiplier.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--sigma", type=float, default=4.0, help="PHATE tree sigma.")
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("data") / "treedata.mat",
        help="Output .mat path. Can be local or remote-mounted path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    tree_data, tree_clusters = phate.tree.gen_dla(
        n_dim=args.n_dim,
        n_branch=args.n_branch,
        branch_length=args.branch_length,
        rand_multiplier=args.rand_multiplier,
        seed=args.seed,
        sigma=args.sigma,
    )

    tree_data = np.asarray(tree_data, dtype=np.float32)
    tree_clusters = np.asarray(tree_clusters, dtype=np.int64) + 1

    print("raw tree_data shape    :", tree_data.shape)
    print("raw tree_clusters shape:", tree_clusters.shape)

    m_data = tree_data
    c_data = tree_clusters.reshape(-1, 1)

    print("M shape:", m_data.shape, "dtype:", m_data.dtype)
    print("C shape:", c_data.shape, "dtype:", c_data.dtype)

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    savemat(args.save_path, {"M": m_data, "C": c_data, "X": m_data, "Y": c_data})
    print("saved to:", args.save_path.resolve())


if __name__ == "__main__":
    main()

