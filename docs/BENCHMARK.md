# Benchmark methodology

The benchmark tool (`bench/pyrsgm_bench.cpp`) compares pyrsgm against the two
most common CUDA SGM implementations and against the CPU algorithm the BT path
was modelled on:

| id | implementation | cost | paths | notes |
|---|---|---|---|---|
| `pyrsgm-L{0,1,2,3}-bt` | this library | Birchfield–Tomasi on gradient + intensity, 5×5 SAD | 3 (L→R, T→B, R→L) | L0 = single scale; L*n* = full search at 1/2ⁿ, then 16-label refinement per finer level |
| `pyrsgm-L{0,2}-census` | this library | 5×5 Census, Hamming | 3 | same pyramid scheme |
| `opencv-cuda-sgm-hh` / `-hh4` | `cv::cuda::StereoSGM` (opencv_contrib `cudastereo`, a port of libSGM) | 9×7 Census | 8 / 4 | P1=10, P2=120, uniqueness 5 %, LR diff 1, 3×3 median, sub-pixel |
| `libsgm-8path` / `-4path` | fixstars libSGM 3.1 | symmetric 9×7 Census | 8 / 4 | P1=10, P2=120, uniqueness 0.95, LR diff 1, 3×3 median, sub-pixel |
| `opencv-cpu-sgbm-3way` | `cv::StereoSGBM` `MODE_SGBM_3WAY` | BT 5×5 | 3 | CPU reference of the BT path with identical P1/P2/uniqueness; no speckle filter |

pyrsgm parameters in the benchmark: `block_size=5`, BT `P1=300 P2=2400`,
Census `P1=8 P2=32`, `uniqueness_ratio=10`, `disp12_max_diff=1`, LR check and
3×3 median on, `pyramid_coarse_disparities=128`, `pyramid_min_confidence=0.05`.
Every library runs with its own recommended penalties because the cost scales
differ (BT SAD sums of 25 pixels vs. 24- or 63-bit Hamming distances).

## Timing

* Inputs are uploaded to device memory once; every GPU algorithm reads device
  memory and writes a device `int16` disparity map, so no PCIe transfer is
  timed.  The CPU reference reports host wall-clock around `compute()` only.
* `gpu ms` is CUDA-event time on the default stream around one call; `wall ms`
  is host time including the final synchronisation.  Reported values are
  medians over `rounds × iters` samples.
* Before timing, each (algorithm, scene) pair is warmed up for at least
  `--warmup` iterations **and** `--warmup-ms` milliseconds.  This matters on
  power-limited GPUs (laptops, Jetson): clocks drop to what the power budget can
  sustain for that kernel mix, and the first few iterations at boost clock
  would otherwise be misleading.  `--rounds 3` interleaves algorithms three
  times so that slow thermal drift affects all of them similarly.
* The disparity range `D` is the smallest of 64/128/256 that covers the scene's
  `ndisp` (libSGM and OpenCV only support those values); pyrsgm uses the same
  `D` as `num_disparities` and covers `[0, D-1]` through the pyramid.

## Accuracy

Middlebury 2014 (MiddEval3) *trainingQ*: 15 scenes at quarter resolution
(~700×480, Teddy/ArtL smaller) with dense ground truth and non-occlusion masks.
Metrics follow `MiddEval3/code/evaldisp.cpp`:

* evaluated pixels: non-occluded (`mask0nocc == 255`), GT finite, `0 < GT <= ndisp`;
* `density`: share of evaluated pixels with a valid estimate (`disp > 0`);
* `bad N (valid)`: share of **valid** evaluated pixels with `|d − GT| > N` px;
* `avg err`: mean absolute error over valid pixels;
* `bad N (all)`: share of **all** evaluated pixels that are invalid or have
  error > N px — the MiddEval3 convention when holes count as errors.  This is
  the fairest single number for algorithms with different densities.

Invalid disparity values are `(min_disparity − 1) × 16 = −16` for pyrsgm and
OpenCV and `0xFFF0` (same bit pattern) for libSGM; the evaluation treats any
value ≤ 0 as invalid.

## Reproducing

```bash
scripts/download_middlebury.sh          # ~45 MB into data/middlebury
scripts/build_libsgm.sh                 # optional, fixstars libSGM into third_party/
cmake -S . -B build -DOpenCV_DIR=/path/to/opencv-cuda/lib/cmake/opencv4
cmake --build build -j
scripts/run_benchmark.sh                # writes docs/results/*.csv|.md and images
scripts/profile_kernels.sh pyrsgm-L2-bt 960x540 128   # per-kernel breakdown (nsys)
```

`pyrsgm_bench --help` lists all options; `--algos`, `--scenes`, `--sizes`,
`--ndisp` select subsets, `--save-disp DIR` writes colour-mapped disparity maps.
