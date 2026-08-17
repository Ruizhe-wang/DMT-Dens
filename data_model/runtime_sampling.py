import numpy as np


def subsample_adata_arrays(adata, data, label, sample_data_size=None, seed=42):
    """Return a reproducible row subset shared by adata, data, and label."""
    n_samples = data.shape[0]
    if sample_data_size is None or int(sample_data_size) < 0 or n_samples <= int(sample_data_size):
        indices = np.arange(n_samples)
        return adata, data, label, indices

    sample_size = int(sample_data_size)
    rng = np.random.default_rng(int(seed))
    indices = rng.choice(n_samples, sample_size, replace=False)
    return adata[indices].copy(), data[indices], label[indices], indices
