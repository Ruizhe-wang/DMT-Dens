#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: bash scripts/run_globalstd_after_lr.sh <predecessor-sweep-id> <new-sweep-path>" >&2
  exit 2
fi

predecessor_sweep_id="$1"
new_sweep_path="$2"
repo_root="/usr/storage/ruizhe/mldr_rz"
log_root="logs/global_standardization_diagnostic"

cd "$repo_root"
mkdir -p "$log_root/smoke" "$log_root/agents"

echo "Waiting for predecessor sweep ${predecessor_sweep_id} to finish..."
while pgrep -af "[w]andb agent.*${predecessor_sweep_id}" >/dev/null; do
  date '+%F %T predecessor still active'
  sleep 60
done

echo "Predecessor complete; running focused unit tests."
python -m pytest \
  tests/test_density_numerics.py \
  tests/test_encoder_benchmark_metrics.py \
  -q 2>&1 | tee "$log_root/unit_tests.log"

configs=(
  configs/encoder_bench/ablation_global_standardization_diagnostic/runs/globalstd_latent-bn_epi_full_seed42.yaml
  configs/encoder_bench/ablation_global_standardization_diagnostic/runs/globalstd_latent-bn_epi_base_seed42.yaml
  configs/encoder_bench/ablation_global_standardization_diagnostic/runs/globalstd_latent-bn_epi_b2_bidirectional_seed42.yaml
  configs/encoder_bench/ablation_global_standardization_diagnostic/runs/globalstd_latent-bn_epi_b5_multi_density_seed42.yaml
  configs/encoder_bench/ablation_global_standardization_diagnostic/runs/globalstd_latent-bn_hcl_full_seed42.yaml
  configs/encoder_bench/ablation_global_standardization_diagnostic/runs/globalstd_latent-bn_hcl_b4_single_density_seed42.yaml
  configs/encoder_bench/ablation_global_standardization_diagnostic/runs/globalstd_latent-bn_mnist_full_seed42.yaml
)

echo "Running seven two-batch GPU smoke tests."
pids=()
for gpu in "${!configs[@]}"; do
  config="${configs[$gpu]}"
  name="$(basename "$config" .yaml)"
  (
    CUDA_VISIBLE_DEVICES="$gpu" python main.py fit \
      --config="$config" \
      --trainer.fast_dev_run=2 \
      2>&1 | tee "$log_root/smoke/${name}.log"
  ) &
  pids+=("$!")
done

smoke_failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    smoke_failed=1
  fi
done
if [[ "$smoke_failed" -ne 0 ]]; then
  echo "At least one smoke test failed; refusing to launch the sweep." >&2
  exit 1
fi

echo "All smoke tests passed; launching ${new_sweep_path}."
bash scripts/run_wandb.sh "$new_sweep_path" 0,1,2,3,4,5,6,7 4 \
  2>&1 | tee "$log_root/agents/agents_$(basename "$new_sweep_path").log"
