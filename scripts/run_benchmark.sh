#!/usr/bin/env bash
# Reproduces the tables in docs/results: accuracy + timing on Middlebury
# trainingQ, and timing-only scaling on resized images.
#
#   scripts/run_benchmark.sh [BUILD_DIR] [OUT_DIR]
#
# Expects the dataset in data/middlebury (scripts/download_middlebury.sh) and a
# configured build (cmake -S . -B build && cmake --build build).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${1:-$ROOT/build}"
OUT="${2:-$ROOT/docs/results}"
DATA="$ROOT/data/middlebury/MiddEval3/trainingQ"
BENCH="$BUILD/bench/pyrsgm_bench"

[ -x "$BENCH" ] || { echo "benchmark binary not found: $BENCH (build with PYRSGM_BUILD_BENCH=ON)"; exit 1; }
[ -d "$DATA" ] || { echo "dataset not found: $DATA (run scripts/download_middlebury.sh)"; exit 1; }

mkdir -p "$OUT/images"
GPU_TAG="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 | tr ' ' '_' | tr -cd '[:alnum:]_-')"
GPU_TAG="${GPU_TAG:-gpu}"

echo "== Middlebury trainingQ: accuracy + timing"
"$BENCH" --dataset "$DATA" \
    --iters 30 --warmup 5 --warmup-ms 500 --rounds 3 \
    --csv "$OUT/middlebury_trainingQ_${GPU_TAG}.csv" \
    --md  "$OUT/middlebury_trainingQ_${GPU_TAG}.md" \
    --save-disp "$OUT/images"

echo "== Resolution scaling, D=128"
"$BENCH" --dataset "$DATA" --scenes Adirondack \
    --sizes 640x480,960x540,1280x720,1920x1080 --ndisp 128 \
    --iters 30 --warmup 5 --warmup-ms 500 --rounds 3 \
    --csv "$OUT/scaling_D128_${GPU_TAG}.csv" \
    --md  "$OUT/scaling_D128_${GPU_TAG}.md"

echo "== Resolution scaling, D=256"
"$BENCH" --dataset "$DATA" --scenes Adirondack \
    --sizes 960x540,1280x720,1920x1080 --ndisp 256 \
    --iters 30 --warmup 5 --warmup-ms 500 --rounds 3 \
    --csv "$OUT/scaling_D256_${GPU_TAG}.csv" \
    --md  "$OUT/scaling_D256_${GPU_TAG}.md"

echo "results written to $OUT"
