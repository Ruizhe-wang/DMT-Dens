#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-dmt-dens}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required. Install Miniconda or Mambaforge first." >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  conda env update --name "$ENV_NAME" --file environment.yml --prune
else
  conda env create --name "$ENV_NAME" --file environment.yml
fi

echo "Environment ready. Run: conda activate $ENV_NAME"
