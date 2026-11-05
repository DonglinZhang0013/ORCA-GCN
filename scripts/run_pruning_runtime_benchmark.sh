#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash run_pruning_runtime_benchmark.sh FB15k-237 0 41504 30
#   bash run_pruning_runtime_benchmark.sh WN18RR    0 41504 30
#
# Arguments:
#   1 dataset
#   2 GPU id
#   3 random seed
#   4 number of benchmark epochs
#
# PROJECT_ROOT may be set explicitly; otherwise use this script's repository root.

DATASET="${1:-FB15k-237}"
GPU="${2:-0}"
SEED="${3:-41504}"
EPOCHS="${4:-30}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"

cd "$PROJECT_ROOT"
mkdir -p logs/pruning_runtime

echo "Project:  $PROJECT_ROOT"
echo "Dataset:  $DATASET"
echo "GPU:      $GPU"
echo "Seed:     $SEED"
echo "Epochs:   $EPOCHS"
echo "Started:  $(date -Iseconds)"

# Run all pruning levels sequentially on the same GPU and machine.
# These short runs are for runtime measurement only. Do not use their
# short-run link-prediction metrics to replace the full-run Table 3 results.
for K in 0.0 0.2 0.4 0.6
do
    SAFE_K="${K/./p}"
    LOG="logs/pruning_runtime/${DATASET}_seed${SEED}_k${SAFE_K}_${EPOCHS}ep.log"

    echo
    echo "===== ${DATASET} | prune_topk_orc=${K} ====="
    echo "Log: $LOG"

    python main.py \
      --score_func conve \
      --opn ccorr \
      --data "$DATASET" \
      --gpu "$GPU" \
      --seed "$SEED" \
      --uniform_geometry \
      --tau 0 \
      --prune_topk_orc "$K" \
      --epoch "$EPOCHS" \
      --eval_every 5 \
      --patience 100 \
      --profile_runtime \
      2>&1 | tee "$LOG"
done

python scripts/summarize_pruning_runtime.py \
  --project-root "$PROJECT_ROOT" \
  --dataset "$DATASET" \
  --seed "$SEED" \
  --epochs "$EPOCHS"

echo "Finished: $(date -Iseconds)"