GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU (sm_120)  
CUDA runtime: 12.8, OpenCV 4.12.0  
Scenes: 15, rounds: 3 x 30 timed iterations (median of pooled samples), warm-up: >= 5 iterations and >= 500 ms per algorithm

| algorithm | paths | avg GPU ms | FPS | wall ms | density % | bad1 % (valid) | bad2 % (valid) | avg err px | bad1 % (all) | bad2 % (all) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pyrsgm-L0-bt | 3 | 5.68 | 176.0 | 5.69 | 78.0 | 14.89 | 10.23 | 1.78 | 33.33 | 29.75 |
| pyrsgm-L1-bt | 3 | 4.78 | 209.1 | 4.79 | 88.6 | 16.48 | 11.79 | 1.83 | 25.55 | 21.45 |
| pyrsgm-L2-bt | 3 | 5.74 | 174.3 | 5.74 | 85.7 | 17.83 | 13.20 | 2.26 | 28.98 | 25.05 |
| pyrsgm-L3-bt | 3 | 5.31 | 188.2 | 5.32 | 85.6 | 18.48 | 13.89 | 2.43 | 29.63 | 25.72 |
| pyrsgm-L0-census | 3 | 4.86 | 205.8 | 4.87 | 78.9 | 14.52 | 9.53 | 1.34 | 32.34 | 28.47 |
| pyrsgm-L2-census | 3 | 3.10 | 322.2 | 3.11 | 85.7 | 15.02 | 10.26 | 1.39 | 26.85 | 22.86 |
| opencv-cuda-sgm-hh | 8 | 2.75 | 363.5 | 2.76 | 88.7 | 15.54 | 10.13 | 1.44 | 24.84 | 20.08 |
| opencv-cuda-sgm-hh4 | 4 | 1.71 | 585.2 | 1.71 | 88.4 | 15.37 | 10.02 | 1.55 | 24.91 | 20.24 |
| libsgm-8path | 8 | 2.89 | 345.5 | 2.90 | 88.6 | 15.45 | 10.03 | 1.42 | 24.86 | 20.11 |
| libsgm-4path | 4 | 1.72 | 581.3 | 1.73 | 88.3 | 15.30 | 9.95 | 1.54 | 24.94 | 20.27 |
| opencv-cpu-sgbm-3way | 3 | 12.28 | 81.4 | 12.28 | 77.9 | 14.93 | 10.34 | 1.81 | 33.41 | 29.89 |

`bad N % (valid)` is the share of evaluated non-occluded GT pixels with a valid estimate whose error exceeds N px; `bad N % (all)` additionally counts invalid (hole) pixels as bad, which is the MiddEval3 convention for sparse outputs.  Time columns are the per-scene medians averaged over scenes; the CPU reference reports host time.

### Per-scene GPU time (ms, median)

| scene | size | D | pyrsgm-L0-bt | pyrsgm-L1-bt | pyrsgm-L2-bt | pyrsgm-L3-bt | pyrsgm-L0-census | pyrsgm-L2-census | opencv-cuda-sgm-hh | opencv-cuda-sgm-hh4 | libsgm-8path | libsgm-4path | opencv-cpu-sgbm-3way |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Adirondack | 718x496 | 128 | 5.89 | 5.17 | 6.80 | 6.11 | 4.83 | 3.47 | 2.73 | 1.70 | 3.15 | 1.74 | 13.38 |
| ArtL | 347x277 | 64 | 1.04 | 1.80 | 1.85 | 2.05 | 1.00 | 1.39 | 0.63 | 0.38 | 0.63 | 0.38 | 2.25 |
| Jadeplant | 659x497 | 256 | 10.01 | 5.08 | 4.76 | 4.85 | 8.37 | 2.96 | 5.63 | 2.83 | 5.36 | 2.99 | 17.83 |
| Motorcycle | 741x497 | 128 | 6.12 | 5.44 | 6.83 | 6.41 | 5.04 | 3.53 | 2.92 | 1.77 | 3.43 | 1.81 | 14.75 |
| MotorcycleE | 741x497 | 128 | 6.11 | 5.49 | 6.87 | 6.22 | 5.06 | 3.55 | 2.84 | 1.78 | 3.33 | 1.80 | 14.48 |
| Piano | 707x481 | 128 | 5.68 | 5.11 | 6.54 | 5.76 | 4.75 | 3.36 | 2.63 | 1.72 | 2.91 | 1.71 | 13.34 |
| PianoL | 707x481 | 128 | 5.68 | 4.77 | 5.86 | 5.09 | 4.90 | 3.25 | 2.58 | 1.72 | 2.81 | 1.74 | 13.54 |
| Pipes | 735x485 | 128 | 5.94 | 5.29 | 6.52 | 5.77 | 5.00 | 3.39 | 2.75 | 1.78 | 3.07 | 1.79 | 14.55 |
| Playroom | 699x476 | 128 | 5.59 | 4.91 | 6.06 | 5.35 | 4.73 | 3.19 | 2.51 | 1.72 | 2.74 | 1.72 | 13.20 |
| Playtable | 680x463 | 128 | 5.31 | 4.88 | 6.11 | 5.22 | 4.59 | 3.17 | 2.37 | 1.66 | 2.54 | 1.65 | 11.22 |
| PlaytableP | 681x462 | 128 | 5.34 | 4.92 | 6.08 | 5.26 | 4.67 | 3.18 | 2.39 | 1.65 | 2.54 | 1.65 | 11.68 |
| Recycle | 720x486 | 128 | 5.84 | 5.37 | 6.54 | 6.06 | 4.91 | 3.42 | 2.69 | 1.70 | 3.04 | 1.73 | 12.78 |
| Shelves | 738x497 | 64 | 3.90 | 5.05 | 6.87 | 5.88 | 3.63 | 3.52 | 1.67 | 1.32 | 1.69 | 1.28 | 8.65 |
| Teddy | 450x375 | 64 | 1.95 | 2.88 | 3.09 | 3.33 | 1.97 | 1.96 | 1.00 | 0.62 | 0.97 | 0.63 | 3.78 |
| Vintage | 722x480 | 256 | 10.81 | 5.57 | 5.28 | 6.34 | 9.42 | 3.21 | 5.92 | 3.28 | 5.21 | 3.19 | 18.81 |

### Per-scene bad1 % (all, invalid counted as bad)

| scene | pyrsgm-L0-bt | pyrsgm-L1-bt | pyrsgm-L2-bt | pyrsgm-L3-bt | pyrsgm-L0-census | pyrsgm-L2-census | opencv-cuda-sgm-hh | opencv-cuda-sgm-hh4 | libsgm-8path | libsgm-4path | opencv-cpu-sgbm-3way |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Adirondack | 27.6 | 17.0 | 19.7 | 20.4 | 25.1 | 16.9 | 12.6 | 12.2 | 12.6 | 12.2 | 27.7 |
| ArtL | 29.6 | 30.2 | 36.0 | 36.5 | 21.5 | 25.6 | 27.6 | 27.8 | 27.7 | 27.8 | 29.8 |
| Jadeplant | 52.8 | 43.2 | 58.0 | 61.0 | 47.9 | 40.6 | 34.5 | 34.1 | 34.6 | 34.1 | 53.3 |
| Motorcycle | 21.0 | 11.8 | 14.5 | 15.6 | 21.3 | 12.0 | 13.5 | 13.4 | 13.5 | 13.4 | 21.0 |
| MotorcycleE | 23.6 | 14.2 | 17.7 | 18.1 | 20.9 | 11.7 | 13.0 | 12.9 | 13.0 | 12.9 | 23.9 |
| Piano | 37.5 | 28.5 | 29.3 | 29.3 | 30.5 | 25.0 | 24.9 | 24.0 | 24.9 | 24.0 | 36.4 |
| PianoL | 52.2 | 48.7 | 50.4 | 50.1 | 45.0 | 47.6 | 38.4 | 40.4 | 38.4 | 40.4 | 52.3 |
| Pipes | 24.8 | 20.9 | 24.7 | 25.8 | 23.5 | 22.3 | 22.7 | 22.2 | 22.7 | 22.2 | 24.9 |
| Playroom | 32.4 | 26.0 | 30.3 | 31.2 | 33.7 | 32.4 | 27.9 | 28.0 | 28.0 | 28.1 | 32.0 |
| Playtable | 31.0 | 21.0 | 24.4 | 26.0 | 43.6 | 33.2 | 26.0 | 29.3 | 26.2 | 29.4 | 31.6 |
| PlaytableP | 29.6 | 19.2 | 22.8 | 24.6 | 31.5 | 23.8 | 21.3 | 20.7 | 21.3 | 20.7 | 29.7 |
| Recycle | 29.3 | 14.8 | 16.4 | 17.2 | 29.5 | 17.8 | 15.8 | 15.9 | 15.8 | 15.9 | 29.2 |
| Shelves | 41.9 | 39.5 | 39.8 | 39.5 | 46.9 | 45.1 | 46.3 | 45.7 | 46.3 | 45.7 | 42.1 |
| Teddy | 16.3 | 10.8 | 11.7 | 11.4 | 13.8 | 9.2 | 13.2 | 12.3 | 13.2 | 12.4 | 16.9 |
| Vintage | 50.2 | 37.5 | 39.0 | 37.6 | 50.4 | 39.4 | 34.8 | 34.7 | 34.8 | 34.7 | 50.3 |

### Algorithm settings

- `pyrsgm-L0-bt`: pyrsgm 0.1.0, 3 paths, BT+gradient 5x5 SAD P1=300 P2=2400, uniq 10%, LR diff 1, median, single scale
- `pyrsgm-L1-bt`: pyrsgm 0.1.0, 3 paths, BT+gradient 5x5 SAD P1=300 P2=2400, uniq 10%, LR diff 1, median, pyramid 1 level(s): coarse 1/2 + 16-label refine
- `pyrsgm-L2-bt`: pyrsgm 0.1.0, 3 paths, BT+gradient 5x5 SAD P1=300 P2=2400, uniq 10%, LR diff 1, median, pyramid 2 level(s): coarse 1/4 + 16-label refine
- `pyrsgm-L3-bt`: pyrsgm 0.1.0, 3 paths, BT+gradient 5x5 SAD P1=300 P2=2400, uniq 10%, LR diff 1, median, pyramid 3 level(s): coarse 1/8 + 16-label refine
- `pyrsgm-L0-census`: pyrsgm 0.1.0, 3 paths, 5x5 Census/Hamming P1=8 P2=32, uniq 10%, LR diff 1, median, single scale
- `pyrsgm-L2-census`: pyrsgm 0.1.0, 3 paths, 5x5 Census/Hamming P1=8 P2=32, uniq 10%, LR diff 1, median, pyramid 2 level(s): coarse 1/4 + 16-label refine
- `opencv-cuda-sgm-hh`: cv::cuda::StereoSGM (OpenCV 4.12.0, libSGM port), 8 paths, 9x7 census, P1=10 P2=120, uniq 5%, LR diff 1, median, subpixel
- `opencv-cuda-sgm-hh4`: cv::cuda::StereoSGM (OpenCV 4.12.0, libSGM port), 4 paths, 9x7 census, P1=10 P2=120, uniq 5%, LR diff 1, median, subpixel
- `libsgm-8path`: fixstars libSGM 3.1, 8 paths, symmetric 9x7 census, P1=10 P2=120, uniq 0.95, LR diff 1, median, subpixel
- `libsgm-4path`: fixstars libSGM 3.1, 4 paths, symmetric 9x7 census, P1=10 P2=120, uniq 0.95, LR diff 1, median, subpixel
- `opencv-cpu-sgbm-3way`: cv::StereoSGBM MODE_SGBM_3WAY (OpenCV 4.12.0, CPU reference of the BT path), 3 paths, BT 5x5, P1=300 P2=2400, uniq 10%, LR diff 1, no speckle
