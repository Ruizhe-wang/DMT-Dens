# Contributing

Thank you for helping improve DMT-Dens. Before opening a substantial pull
request, please start a GitHub issue describing the proposed change and its
effect on the manuscript workflow or public API.

## Development setup

Create the documented environment from the repository root:

```bash
conda env create -f environment.yml
conda activate dmt-dens
```

Run the lightweight checks before submitting a change:

```bash
python -m pytest -q tests/test_publication_assets.py
python -m compileall -q main.py model data_model callbacks eval scripts tools tests
```

Changes to the model, metrics, or data processing should also run the relevant
numerical tests and include a concise description of the validation data,
configuration, seed, and hardware. Do not commit datasets, checkpoints,
generated figures, credentials, machine-specific paths, or experiment-tracking
metadata.

## Pull requests

- Keep changes focused and document user-visible behavior.
- Add or update tests for bug fixes and new functionality.
- Preserve the resolved manuscript configurations unless the change corrects a
  documented reproducibility issue.
- Confirm that any added data or third-party code may be redistributed under
  its stated license.

By contributing, you agree that your contribution is licensed under the MIT
License in this repository.
