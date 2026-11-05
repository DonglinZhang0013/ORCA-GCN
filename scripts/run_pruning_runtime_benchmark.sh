#!/usr/bin/env bash
set -euo pipefail
# Usage: bash scripts/run_pruning_runtime_benchmark.sh DATASET GPU SEED EPOCHS
DATASET="${1:-FB15k-237}"
GPU="${2:-0}"
SEED="${3:-41504}"
EPOCHS="${4:-30}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"
mkdir -p logs/pruning_runtime
echo "Project: $PROJECT_ROOT | Dataset: $DATASET | GPU: $GPU | Seed: $SEED | Epochs: $EPOCHS"
for K in 0.0 0.2 0.4 0.6; do
  SAFE_K="${K/./p}"
  LOG="logs/pruning_runtime/${DATASET}_seed${SEED}_k${SAFE_K}_${EPOCHS}ep.log"
  python main.py --score_func conve --opn ccorr --data "$DATASET" --gpu "$GPU" \
    --seed "$SEED" --uniform_geometry --tau 0 --prune_topk_orc "$K" \
    --epoch "$EPOCHS" --eval_every 5 --patience 100 --profile_runtime 2>&1 | tee "$LOG"
done
python scripts/summarize_pruning_runtime.py --project-root "$PROJECT_ROOT" \
  --dataset "$DATASET" --seed "$SEED" --epochs "$EPOCHS"
