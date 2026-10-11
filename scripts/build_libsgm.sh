#!/usr/bin/env bash
# Fetches and builds fixstars libSGM (https://github.com/fixstars/libSGM,
# Apache-2.0) into third_party/libSGM/install so that the benchmark can link it.
#
#   scripts/build_libsgm.sh [CUDA_ARCHS]      # e.g. 86 or "86;89;120"
#
# CUDA_ARCHS defaults to the compute capability of the first local GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TP="$ROOT/third_party"
SRC="$TP/libSGM"
ARCHS="${1:-}"

if [ -z "$ARCHS" ]; then
    ARCHS="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '.')"
    ARCHS="${ARCHS:-75;86;89}"
fi

mkdir -p "$TP"
if [ ! -f "$SRC/CMakeLists.txt" ]; then
    echo "[build_libsgm] downloading libSGM master"
    curl -L --fail --retry 5 -o "$TP/libSGM.tar.gz" \
        "https://github.com/fixstars/libSGM/archive/refs/heads/master.tar.gz"
    rm -rf "$TP/libSGM-master" "$SRC"
    tar -xzf "$TP/libSGM.tar.gz" -C "$TP"
    mv "$TP/libSGM-master" "$SRC"
fi

echo "[build_libsgm] building for CUDA architectures: $ARCHS"
cmake -S "$SRC" -B "$SRC/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="$ARCHS" \
    -DCMAKE_INSTALL_PREFIX="$SRC/install" \
    -DENABLE_SAMPLES=OFF -DENABLE_TESTS=OFF -DBUILD_OPENCV_WRAPPER=OFF
cmake --build "$SRC/build" -j"$(nproc)"
cmake --install "$SRC/build"
echo "[build_libsgm] installed to $SRC/install (re-run cmake for pyrsgm to pick it up)"
