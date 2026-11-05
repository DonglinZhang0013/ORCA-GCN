#!/usr/bin/env bash
set -euo pipefail
DATA="${1:-FB15k-237}"
GPU="${2:-0}"
for K in 0.0 0.2 0.4 0.6; do
  python main.py --data "$DATA" --score_func conve --opn ccorr --seed 41504 --gpu "$GPU" --uniform_geometry --tau 0 --prune_topk_orc "$K"
done
