GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU (sm_120)  
CUDA runtime: 12.8, OpenCV 4.12.0  
Scenes: 4, rounds: 3 x 30 timed iterations (median of pooled samples), warm-up: >= 5 iterations and >= 500 ms per algorithm

| algorithm | paths | avg GPU ms | FPS | wall ms |
|---|---|---:|---:|---:|
| pyrsgm-L0-bt | 3 | 17.05 | 58.6 | 17.06 |
| pyrsgm-L1-bt | 3 | 14.64 | 68.3 | 14.65 |
| pyrsgm-L2-bt | 3 | 14.74 | 67.8 | 14.76 |
| pyrsgm-L3-bt | 3 | 15.29 | 65.4 | 15.31 |
| pyrsgm-L0-census | 3 | 15.05 | 66.4 | 15.08 |
| pyrsgm-L2-census | 3 | 7.53 | 132.8 | 7.56 |
| opencv-cuda-sgm-hh | 8 | 8.07 | 123.9 | 8.08 |
| opencv-cuda-sgm-hh4 | 4 | 4.59 | 217.8 | 4.61 |
| libsgm-8path | 8 | 8.38 | 119.3 | 8.43 |
| libsgm-4path | 4 | 4.66 | 214.7 | 4.67 |
| opencv-cpu-sgbm-3way | 3 | 45.38 | 22.0 | 45.38 |

### Per-scene GPU time (ms, median)

| scene | size | D | pyrsgm-L0-bt | pyrsgm-L1-bt | pyrsgm-L2-bt | pyrsgm-L3-bt | pyrsgm-L0-census | pyrsgm-L2-census | opencv-cuda-sgm-hh | opencv-cuda-sgm-hh4 | libsgm-8path | libsgm-4path | opencv-cpu-sgbm-3way |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 640x480 | 640x480 | 128 | 5.18 | 4.58 | 6.10 | 5.42 | 4.37 | 3.17 | 2.29 | 1.57 | 2.53 | 1.56 | 12.34 |
| 960x540 | 960x540 | 128 | 8.38 | 7.19 | 8.01 | 9.53 | 8.52 | 4.28 | 4.44 | 3.00 | 4.11 | 2.71 | 22.28 |
| 1280x720 | 1280x720 | 128 | 19.04 | 16.95 | 16.03 | 16.14 | 16.45 | 8.37 | 7.75 | 4.20 | 8.55 | 4.51 | 40.50 |
| 1920x1080 | 1920x1080 | 128 | 35.60 | 29.84 | 28.82 | 30.06 | 30.88 | 14.30 | 17.81 | 9.59 | 18.36 | 9.85 | 106.41 |

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
