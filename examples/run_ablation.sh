#!/usr/bin/env bash
set -euo pipefail
DATA="${1:-FB15k-237}"
GPU="${2:-0}"
BASE=(python main.py --data "$DATA" --score_func conve --opn ccorr --seed 41504 --gpu "$GPU")
"${BASE[@]}" --tau 0.6
"${BASE[@]}" --tau 0.6 --uniform_geometry
"${BASE[@]}" --tau 0
"${BASE[@]}" --tau 0.6 --random_curvature
