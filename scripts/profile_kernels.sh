#!/usr/bin/env bash
# Per-kernel GPU time breakdown of one benchmark algorithm with Nsight Systems.
#
#   scripts/profile_kernels.sh ALGO [WxH] [D] [BUILD_DIR]
#   e.g. scripts/profile_kernels.sh pyrsgm-L2-bt 960x540 128
#
# Prints a table of (time %, total ms, launches, avg us, kernel) for the timed
# iterations; divide "Total ms" by the launch count per frame to get per-frame cost.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ALGO="${1:?algorithm name, see pyrsgm_bench --list}"
SIZE="${2:-960x540}"
NDISP="${3:-128}"
BUILD="${4:-$ROOT/build}"
DATA="$ROOT/data/middlebury/MiddEval3/trainingQ"
OUT="$(mktemp -d)"

nsys profile -o "$OUT/prof" -t cuda --force-overwrite true \
    "$BUILD/bench/pyrsgm_bench" --dataset "$DATA" --scenes Adirondack \
    --sizes "$SIZE" --ndisp "$NDISP" --iters 20 --warmup 2 --warmup-ms 0 \
    --csv "$OUT/tmp.csv" --algos "$ALGO" > /dev/null 2>&1

nsys stats --report cuda_gpu_kern_sum --format csv "$OUT/prof.nsys-rep" 2>/dev/null | python3 -c '
import csv, sys
rows = list(csv.reader(sys.stdin))
start = next((i for i, r in enumerate(rows) if r and r[0] == "Time (%)"), None)
if start is None:
    sys.exit("no kernel data")
print("%-7s %-10s %-7s %-10s %s" % ("Time%", "Total ms", "Count", "Avg us", "Kernel"))
for r in rows[start + 1:]:
    if len(r) < 9:
        continue
    name = r[8].split("(")[0]
    print("%-7s %-10.3f %-7s %-10.1f %s" % (r[0], float(r[1]) / 1e6, r[2], float(r[3]) / 1e3, name))
'
rm -rf "$OUT"
