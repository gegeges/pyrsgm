#!/usr/bin/env python3
"""Assemble a labelled montage (left image, ground truth, algorithm outputs)
from the disparity PNGs written by `pyrsgm_bench --save-disp`.

    scripts/make_montage.py SCENE [--algos a,b,c] [--images docs/results/images]
                            [--dataset data/middlebury/MiddEval3/trainingQ] [--out FILE]
"""
import argparse
import os
import re
import sys

import cv2
import numpy as np

DEFAULT_ALGOS = [
    "pyrsgm-L0-bt", "pyrsgm-L2-bt", "pyrsgm-L2-census",
    "libsgm-8path", "opencv-cuda-sgm-hh4", "opencv-cpu-sgbm-3way",
]


def read_pfm(path):
    with open(path, "rb") as f:
        if f.readline().strip() != b"Pf":
            raise ValueError("only grayscale PFM supported")
        w, h = map(int, f.readline().split())
        scale = float(f.readline().strip())
        data = np.fromfile(f, "<f4" if scale < 0 else ">f4", w * h)
    return np.flipud(data.reshape(h, w)).astype(np.float32)


def read_ndisp(calib):
    with open(calib) as f:
        for line in f:
            m = re.match(r"ndisp=(\d+)", line)
            if m:
                return int(m.group(1))
    return 128


def round_d(ndisp):
    return 64 if ndisp <= 64 else 128 if ndisp <= 128 else 256


def colorize(disp, max_disp):
    u8 = np.clip(disp * (255.0 / max_disp), 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    color[~np.isfinite(disp) | (disp <= 0)] = 0
    return color


def label(img, text):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--algos", default=",".join(DEFAULT_ALGOS))
    ap.add_argument("--images", default="docs/results/images")
    ap.add_argument("--dataset", default="data/middlebury/MiddEval3/trainingQ")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cols", type=int, default=4)
    args = ap.parse_args()

    scene_dir = os.path.join(args.dataset, args.scene)
    left = cv2.imread(os.path.join(scene_dir, "im0.png"))
    if left is None:
        sys.exit(f"cannot read {scene_dir}/im0.png")
    ndisp = read_ndisp(os.path.join(scene_dir, "calib.txt"))
    d = round_d(ndisp)
    tiles = [label(left, f"{args.scene} left ({left.shape[1]}x{left.shape[0]}, D={d})")]
    gt_path = os.path.join(scene_dir, "disp0GT.pfm")
    if os.path.exists(gt_path):
        tiles.append(label(colorize(read_pfm(gt_path), d), "ground truth"))
    for algo in args.algos.split(","):
        path = os.path.join(args.images, f"{args.scene}_{algo}.png")
        img = cv2.imread(path)
        if img is None:
            print(f"warning: missing {path}", file=sys.stderr)
            continue
        tiles.append(label(img, algo))

    h, w = left.shape[:2]
    tiles = [cv2.resize(t, (w, h)) if t.shape[:2] != (h, w) else t for t in tiles]
    cols = max(1, min(args.cols, len(tiles)))
    rows = (len(tiles) + cols - 1) // cols
    canvas = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = t
    out = args.out or os.path.join(args.images, f"{args.scene}_montage.png")
    cv2.imwrite(out, canvas, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    print(f"wrote {out} ({canvas.shape[1]}x{canvas.shape[0]})")


if __name__ == "__main__":
    main()
