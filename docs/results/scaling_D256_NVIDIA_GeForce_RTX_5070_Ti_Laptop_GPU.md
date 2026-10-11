GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU (sm_120)  
CUDA runtime: 12.8, OpenCV 4.12.0  
Scenes: 3, rounds: 3 x 30 timed iterations (median of pooled samples), warm-up: >= 5 iterations and >= 500 ms per algorithm

| algorithm | paths | avg GPU ms | FPS | wall ms |
|---|---|---:|---:|---:|
| pyrsgm-L0-bt | 3 | 40.46 | 24.7 | 40.53 |
| pyrsgm-L1-bt | 3 | 20.61 | 48.5 | 20.64 |
| pyrsgm-L2-bt | 3 | 17.70 | 56.5 | 17.71 |
| pyrsgm-L3-bt | 3 | 17.40 | 57.5 | 17.41 |
| pyrsgm-L0-census | 3 | 32.37 | 30.9 | 32.39 |
| pyrsgm-L2-census | 3 | 9.55 | 104.7 | 9.56 |
| opencv-cuda-sgm-hh | 8 | 11.60 | 86.2 | 11.61 |
| opencv-cuda-sgm-hh4 | 4 | 10.13 | 98.7 | 10.14 |
| libsgm-8path | 8 | 19.93 | 50.2 | 19.95 |
| libsgm-4path | 4 | 10.40 | 96.2 | 10.40 |
| opencv-cpu-sgbm-3way | 3 | 100.33 | 10.0 | 100.33 |

### Per-scene GPU time (ms, median)

| scene | size | D | pyrsgm-L0-bt | pyrsgm-L1-bt | pyrsgm-L2-bt | pyrsgm-L3-bt | pyrsgm-L0-census | pyrsgm-L2-census | opencv-cuda-sgm-hh | opencv-cuda-sgm-hh4 | libsgm-8path | libsgm-4path | opencv-cpu-sgbm-3way |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 960x540 | 960x540 | 256 | 18.40 | 11.03 | 8.55 | 7.84 | 13.39 | 4.92 | 9.12 | 4.36 | 8.70 | 4.63 | 34.87 |
| 1280x720 | 1280x720 | 256 | 30.31 | 16.17 | 14.18 | 14.35 | 25.59 | 8.09 | 14.07 | 7.59 | 15.74 | 8.44 | 72.88 |
| 1920x1080 | 1920x1080 | 256 | 72.68 | 34.64 | 30.37 | 30.00 | 58.13 | 15.66 | fail | 18.45 | 35.34 | 18.11 | 193.26 |

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
- `opencv-cuda-sgm-hh` failed on 1920x1080: OpenCV(4.12.0) /home/hhm/Documents/OpenCV_CUDA/opencv_contrib/modules/cudastereo/src/cuda/stereosgm.cu:1345: error: (-215:Assertion failed) dest.rows == 1 && buffer_step * num_paths == dest.cols in function 'operator()'

