#!/usr/bin/env bash
set -euo pipefail
GPU="${1:-0}"
python scripts/check_data.py --data FB15k-237
python main.py --data FB15k-237 --score_func conve --opn ccorr --tau 0.6 --seed 41504 --gpu "$GPU"
