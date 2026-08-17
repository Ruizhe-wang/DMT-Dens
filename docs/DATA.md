# Data inputs

DMT-Dens consumes a processed observations-by-features matrix. The public data
module accepts `.npz` and `.h5ad` directly; dataset-specific modules preserve the
exact preprocessing and filenames used by the manuscript experiments.

## Generic input

For NPZ, save `X` as a finite numeric matrix of shape `(n_observations,
n_features)` and `y` as an optional one-dimensional annotation array:

```python
import numpy as np
np.savez_compressed("dataset.npz", X=X.astype("float32"), y=labels)
```

For H5AD, expression values are read from `adata.X`; pass an observation column
with `--data.init_args.label_key`. Labels are retained for plotting/evaluation
and are not used during representation learning.

## Manuscript data-root layout

The resolved dataset modules expect the following paths beneath the data root:

| Dataset | Expected local path |
| --- | --- |
| ACT | `feature_select/Activity_train.csv`, `feature_select/Activity_test.csv` |
| MNIST | downloaded automatically by `torchvision` into `data/` |
| NG20 | `20NG.npy`, `20NG_labels.npy` |
| HCL | `HCL60kafter-elis-all.h5ad` |
| GAST10K | `gast10kwithcelltype.h5ad` |
| MCA | `mca_data/mca_data_dim_34947.npy`, `mca_data/mca_label_dim_34947.npy` |
| EPI | `difftreedata/data/EpitheliaCell_data_n.npy`, `difftreedata/data/EpitheliaCell_label.npy` |
| ArtificialTree | MAT file specified by the selected tree configuration |
| CELEGAN | files under `celegan/` or `difftreedata/data/` as documented by its module |
| dyngen | H5AD filename supplied by the selected case-study configuration |

Raw datasets are not committed because of size and redistribution constraints.
Before public release, the authors should add stable accession/download links
and checksums for every processed file, after confirming redistribution terms.

## Preprocessing contract

- rows are observations and columns are processed features;
- all values must be finite;
- class or cell-type labels must not be included as features;
- any filtering, normalization, log transform, highly-variable-gene selection,
  and scaling used for a reported result must be recorded with that dataset;
- `num_input_dim` and `num_train_data` in the model config must match the final
  processed matrix.

The generated toy dataset is synthetic test data only and carries no biological
claim.
