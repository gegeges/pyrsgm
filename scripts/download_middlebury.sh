#!/usr/bin/env bash
# Downloads the Middlebury 2014 stereo evaluation set (MiddEval3, quarter
# resolution "Q": 15 training scenes with ground truth) used by the benchmark.
#
#   scripts/download_middlebury.sh [DEST]      # default: data/middlebury
#
# Result: DEST/MiddEval3/trainingQ/<scene>/{im0.png,im1.png,calib.txt,disp0GT.pfm,mask0nocc.png}
# Source: https://vision.middlebury.edu/stereo/submit3/  (data ~31 MB, GT ~14 MB)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-$ROOT/data/middlebury}"
BASE_URL="https://vision.middlebury.edu/stereo/submit3/zip"

mkdir -p "$DEST"
cd "$DEST"

for zip in MiddEval3-data-Q.zip MiddEval3-GT0-Q.zip; do
    if [ -f "$zip" ] && unzip -tq "$zip" >/dev/null 2>&1; then
        echo "[download_middlebury] $zip already present"
    else
        echo "[download_middlebury] fetching $zip"
        curl -L --fail --retry 5 --retry-delay 3 -C - -o "$zip" "$BASE_URL/$zip"
    fi
    unzip -q -o "$zip"
done

echo "[download_middlebury] scenes:"
for d in MiddEval3/trainingQ/*/; do
    printf '  %-12s %s\n' "$(basename "$d")" "$(grep '^ndisp=' "$d/calib.txt")"
done
echo "[download_middlebury] done -> $DEST/MiddEval3/trainingQ"
