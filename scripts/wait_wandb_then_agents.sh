#!/usr/bin/env bash
# Wait for the self-hosted W&B GraphQL endpoint to answer, then start agents.
#
# Why this exists: `wandb agent` registers with the server before it constructs
# the model, so when the backend is slow or down the agent exits with
# "the service process is busy and did not respond in time" and its GPU sits
# idle while the rest of the sweep runs. That happened to 4 of 8 agents in E15.
# Run this inside the `ruizhe` tmux session so the retry survives a dropped
# connection and a closed laptop.
#
# Usage:
#   bash scripts/wait_wandb_then_agents.sh <entity/project/sweep_id> <gpu_list> [cpus]
# Example:
#   bash scripts/wait_wandb_then_agents.sh me/proj/abc123 4,5,6,7 4

set -u

SWEEP_PATH="${1:?sweep path required, e.g. entity/project/sweep_id}"
GPU_LIST="${2:?gpu list required, e.g. 4,5,6,7}"
CPUS="${3:-4}"

BASE_URL="${WANDB_BASE_URL:-http://www.zangzelin.fun:4080}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_CHECKS="${MAX_CHECKS:-240}"   # 240 x 60s = 4 hours

echo "waiting for ${BASE_URL}/graphql to answer before starting agents on GPUs ${GPU_LIST}"

for i in $(seq 1 "${MAX_CHECKS}"); do
    code=$(curl -s -m 20 -o /dev/null -w '%{http_code}' \
        -X POST "${BASE_URL}/graphql" \
        -H 'Content-Type: application/json' \
        -d '{"query":"{__typename}"}' || echo 000)
    if [ "${code}" = "200" ]; then
        echo "[$(date '+%H:%M:%S')] backend answered 200 after ${i} check(s); starting agents"
        exec bash scripts/run_wandb.sh "${SWEEP_PATH}" "${GPU_LIST}" "${CPUS}"
    fi
    echo "[$(date '+%H:%M:%S')] check ${i}/${MAX_CHECKS}: HTTP ${code}, retrying in ${POLL_SECONDS}s"
    sleep "${POLL_SECONDS}"
done

echo "backend never answered within the retry budget; agents NOT started" >&2
exit 1
