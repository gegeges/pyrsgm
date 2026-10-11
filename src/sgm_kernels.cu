/**
 * @file sgm_kernels.cu
 * @brief CUDA 3-path Semi-Global Matching with coarse-to-fine pyramid refinement.
 *
 * Extracted from the Tri3SGM tri-camera depth engine (core/src/cuda_3sgm.cu).
 * Two matching costs are available:
 *   - BT  : Birchfield-Tomasi sampling-insensitive cost on a gradient image plus
 *           intensity channel, aggregated with a blockSize x blockSize SAD window.
 *           Strictly follows OpenCV StereoSGBM MODE_SGBM_3WAY (stereosgbm.cpp).
 *   - Census: 5x5 Census transform + Hamming distance (24 informative bits).
 *
 * Pipeline per level:
 *   1. cost volume [y][x][d] (d innermost, 16-bit)
 *   2. 3-path SGM aggregation (left->right, top->bottom, right->left + WTA)
 *   3. WTA, uniqueness check, 1/16-pixel sub-pixel interpolation (OpenCV formula)
 *   4. pseudo left-right consistency check, optional 3x3 median
 *
 * Pyramid refinement levels evaluate exactly 16 contiguous local labels per
 * pixel around the up-scaled parent disparity; path recursion maps real
 * disparity into the previous pixel's local interval, and invalid pixels
 * reset every path.  Comments inside the kernels are kept verbatim from the
 * original implementation (partly Chinese).
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>

#include "pyrsgm/pyrsgm_c.h"
#define CENSUS_THREADS        128

// ─────────────────────────────────────────────────────────────────────────────
// 设备端 helper
// ─────────────────────────────────────────────────────────────────────────────

__device__ inline short dev_min_s(short a, short b) { return a < b ? a : b; }
__device__ inline short dev_max_s(short a, short b) { return a > b ? a : b; }
__device__ inline int   dev_min_i(int a, int b)   { return a < b ? a : b; }
__device__ inline int   dev_max_i(int a, int b)   { return a > b ? a : b; }
__device__ inline unsigned char dev_min_uc(unsigned char a, unsigned char b) { return a < b ? a : b; }
__device__ inline unsigned char dev_max_uc(unsigned char a, unsigned char b) { return a > b ? a : b; }

__device__ inline short dev_saturate_short(int v) {
    return (short)dev_min_i(SHRT_MAX, dev_max_i(SHRT_MIN, v));
}

// ─────────────────────────────────────────────────────────────────────────────
// 自适应 P2：
//   ratio = lower_mult + (1 + alpha - lower_mult) * exp(-|I(p)-I(p-r)|/beta)
//   P2'   = P2 * ratio，clamp 到 [max(1, P2*lower_mult), P2*upper_mult]
// 平坦区（灰度差小）→ ratio→1+alpha，可抬高抑制斑点；边缘（灰度差大）→
// ratio→lower_mult，lower_mult<1 时可低于基础 P2，甚至 < P1，鼓励视差跳变。
// alpha=0 且 lower_mult>=1 时退化为常数 P2。
// 每个 block 的 D 个线程冗余计算同一像素的 p2_eff：两次 byte 读是 L1 广播，
// __expf 为硬件指令，代价远低于为共享该值额外引入一次 block 同步。
// ─────────────────────────────────────────────────────────────────────────────
__device__ inline int dev_adaptive_p2(int P2, int cur, int prev,
                                      float alpha, float beta,
                                      float lowerMult, float upperMult)
{
    const float diff = fabsf((float)(cur - prev));
    const float ratio = lowerMult + (1.0f + alpha - lowerMult) * __expf(-diff / beta);
    float adaptive = (float)P2 * ratio;
    const float lower = fmaxf(1.0f, (float)P2 * lowerMult);
    const float upper = (float)P2 * upperMult;
    adaptive = fminf(upper, fmaxf(lower, adaptive));
    return (int)(adaptive + 0.5f);
}

__global__ void kernel_FillDisp(short* disp, int width, int height, short value)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    disp[y * width + x] = value;
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1: BT 梯度采样强度预计算
//
// 对标 OpenCV stereosgbm.cpp calcPixelCostBT() 中的 precompute 阶段：
//   grad = (row[x+1]-row[x-1])*2 + row[x-width+1]-row[x-width-1] + row[x+width+1]-row[x+width-1]
//   I_grad = clip(grad, -ftzero, ftzero) + ftzero
//   ftzero = max(preFilterCap, 15) | 1 = 63
//   输出范围 [0, 126]
// ─────────────────────────────────────────────────────────────────────────────

__global__ void kernel_BTGradPrecompute(
    const unsigned char* __restrict__ gray,
    unsigned char* __restrict__ grad_img,
    int width, int height, int src_stride, int dst_stride, int preFilterCap)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int x_prev = dev_max_i(x - 1, 0);
    int x_next = dev_min_i(x + 1, width - 1);
    int y_prev = dev_max_i(y - 1, 0);
    int y_next = dev_min_i(y + 1, height - 1);

    int grad = (int)gray[y * src_stride + x_next] - (int)gray[y * src_stride + x_prev];
    grad = grad * 2;
    grad += (int)gray[y_prev * src_stride + x_next] - (int)gray[y_prev * src_stride + x_prev];
    grad += (int)gray[y_next * src_stride + x_next] - (int)gray[y_next * src_stride + x_prev];

    int ftzero = dev_max_i(preFilterCap, 15) | 1;  // 默认 63
    int clip = grad;
    if (clip < -ftzero) clip = -ftzero;
    if (clip >  ftzero) clip =  ftzero;
    grad_img[y * dst_stride + x] = (unsigned char)(clip + ftzero);  // [0, 126]
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2: BT 像素匹配代价计算
//
// 对标 OpenCV calcPixelCostBT() 的代价计算阶段：
//   对每个像素 (x,y) 和视差 d：
//     左像素 u = I_grad_left[x], 半像素插值得 [u0, u1]
//     右像素 v = I_grad_right[x-d], 半像素插值得 [v0, v1]
//     cost_grad = min(max(0, u-v1, v0-u), max(0, v-u1, u0-v))
//
//   同时计算原始强度通道（除以 4）：
//     左像素 u = I_left[x], 半像素插值得 [u0, u1]
//     右像素 v = I_right[x-d], 半像素插值得 [v0, v1]
//     cost_int = min(max(0, u-v1, v0-u), max(0, v-u1, u0-v))
//
//   总代价 = cost_grad + (cost_int >> 2)
// ─────────────────────────────────────────────────────────────────────────────

#define BTCOST_TILE_X 64

__global__ void kernel_BTCost(
    const unsigned char* __restrict__ grad1, const unsigned char* __restrict__ grad2,
    const unsigned char* __restrict__ img1,  const unsigned char* __restrict__ img2,
    short* __restrict__ cost_volume,
    int width, int height, int grad_stride, int img1_stride, int img2_stride,
    int numDisp, int minDisp)
{
    // 一个 block 处理一行里 BTCOST_TILE_X 个连续 x（blockIdx.x 为 tile 序号，
    // blockIdx.y 为 y），block 内 D 个线程各处理一个视差 d。
    //
    // 共享内存 tile 缓存：
    //   - 左图 grad1/img1 需要 [x0-1, x0+TILE] 共 TILE+2 个像素；
    //   - 右图 grad2/img2 需要 x2 = x-(d+minDisp) 的取值范围
    //     [x0-(D-1)-minDisp-1, x0+TILE-1-minDisp+1] 共 TILE+D+2 个像素。
    //   相邻 (x,d) 组合大量复用这些像素：原实现一个 block 只算一个像素、
    //   完全依赖 L1 命中；tile 化后每个输入字节全局内存只读一次，
    //   并且 block 数从 W*H 降为 (W/TILE)*H，调度开销同步下降。
    //
    // 边界语义：加载时把全局下标钳位到 [0, width-1]。原实现的边界复制
    // （x2==0 时 vg_l=vg、x2==width-1 时 vg_r=vg，左图同理）与钳位后读到的
    // 值逐像素严格一致，因此结果与原核完全相同。
    // 写 cost_volume[(y*width + x)*D + d]：同一 block 的 D 线程写连续地址 → 完全合并。
    const int y  = blockIdx.y;
    const int x0 = blockIdx.x * BTCOST_TILE_X;
    const int d  = threadIdx.x;
    if (y >= height || x0 >= width || d >= numDisp) return;

    extern __shared__ unsigned char s_bt[];
    unsigned char* s_grad1 = s_bt;                                       // [TILE+2]
    unsigned char* s_img1  = s_grad1 + (BTCOST_TILE_X + 2);              // [TILE+2]
    unsigned char* s_grad2 = s_img1  + (BTCOST_TILE_X + 2);              // [TILE+D+2]
    unsigned char* s_img2  = s_grad2 + (BTCOST_TILE_X + numDisp + 2);    // [TILE+D+2]

    for (int i = d; i < BTCOST_TILE_X + 2; i += blockDim.x) {
        int gx = dev_min_i(dev_max_i(x0 - 1 + i, 0), width - 1);
        s_grad1[i] = grad1[y * grad_stride + gx];
        s_img1[i]  = img1[y * img1_stride + gx];
    }
    const int base2 = x0 - (numDisp - 1) - minDisp;   // tile 内 x2 的最小取值
    for (int i = d; i < BTCOST_TILE_X + numDisp + 2; i += blockDim.x) {
        int gx = dev_min_i(dev_max_i(base2 - 1 + i, 0), width - 1);
        s_grad2[i] = grad2[y * grad_stride + gx];
        s_img2[i]  = img2[y * img2_stride + gx];
    }
    __syncthreads();

    const int x_end = dev_min_i(x0 + BTCOST_TILE_X, width);
    for (int x = x0; x < x_end; ++x) {
        const int lx = x - x0 + 1;   // 左图 smem 下标（+1 为 halo 偏移）

        // ── 左像素半像素插值（梯度 + 原始强度通道），全 block 广播读 smem ──
        unsigned char ug = s_grad1[lx];
        int ul_g = ((int)ug + s_grad1[lx - 1]) / 2;
        int ur_g = ((int)ug + s_grad1[lx + 1]) / 2;
        int u0_g = dev_min_i(dev_min_i(ul_g, ur_g), (int)ug);
        int u1_g = dev_max_i(dev_max_i(ul_g, ur_g), (int)ug);

        unsigned char ui = s_img1[lx];
        int ul_i = ((int)ui + s_img1[lx - 1]) / 2;
        int ur_i = ((int)ui + s_img1[lx + 1]) / 2;
        int u0_i = dev_min_i(dev_min_i(ul_i, ur_i), (int)ui);
        int u1_i = dev_max_i(dev_max_i(ul_i, ur_i), (int)ui);

        int x2 = x - (d + minDisp);
        short total_cost;

        if (x2 >= 0 && x2 < width) {
            const int rx = x2 - base2 + 1;   // 右图 smem 下标

            unsigned char vg = s_grad2[rx];
            int vl_g = ((int)vg + s_grad2[rx - 1]) / 2;
            int vr_g = ((int)vg + s_grad2[rx + 1]) / 2;
            int v0_g = dev_min_i(dev_min_i(vl_g, vr_g), (int)vg);
            int v1_g = dev_max_i(dev_max_i(vl_g, vr_g), (int)vg);

            int c0_g = dev_max_i(0, (int)ug - v1_g);
            c0_g     = dev_max_i(c0_g, v0_g - (int)ug);
            int c1_g = dev_max_i(0, (int)vg - u1_g);
            c1_g     = dev_max_i(c1_g, (int)u0_g - (int)vg);
            int cost_grad = dev_min_i(c0_g, c1_g);

            unsigned char vi = s_img2[rx];
            int vl_i = ((int)vi + s_img2[rx - 1]) / 2;
            int vr_i = ((int)vi + s_img2[rx + 1]) / 2;
            int v0_i = dev_min_i(dev_min_i(vl_i, vr_i), (int)vi);
            int v1_i = dev_max_i(dev_max_i(vl_i, vr_i), (int)vi);

            int c0_i = dev_max_i(0, (int)ui - v1_i);
            c0_i     = dev_max_i(c0_i, v0_i - (int)ui);
            int c1_i = dev_max_i(0, (int)vi - u1_i);
            c1_i     = dev_max_i(c1_i, (int)u0_i - (int)vi);
            int cost_int = dev_min_i(c0_i, c1_i);

            total_cost = (short)dev_min_i(255, cost_grad + (cost_int >> 2));
        } else {
            total_cost = 255;
        }

        cost_volume[(y * width + x) * numDisp + d] = total_cost;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Census Transform + Hamming 代价（对标 SgmStereoCuda/cusgm_cc.cu 5x5 Census）
// ─────────────────────────────────────────────────────────────────────────────

__device__ inline unsigned char dev_hamming32(uint32_t a, uint32_t b)
{
    return (unsigned char)__popc(a ^ b);
}

// 5x5 Census。图像边界采用复制扩展，保证所有像素都有有效描述子。
__global__ void kernel_CensusCompute(
    const unsigned char* __restrict__ gray,
    uint32_t* __restrict__ census,
    int width, int height,
    int src_stride, int dst_stride)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y;

    constexpr int RX = 2, RY = 2;
    constexpr int WIN_W = 2 * RX + 1;  // 5
    constexpr int WIN_H = 2 * RY + 1;  // 5

    extern __shared__ unsigned char s[];
    const int row_stride = blockDim.x + 2 * RX;
    const int block_x0 = blockIdx.x * blockDim.x;

#pragma unroll
    for (int dy = 0; dy < WIN_H; ++dy) {
        const int gy = dev_min_i(dev_max_i(y + dy - RY, 0), height - 1);
        unsigned char* srow = s + dy * row_stride;

        const int gx = dev_min_i(block_x0 + (int)threadIdx.x, width - 1);
        srow[threadIdx.x + RX] = gray[gy * src_stride + gx];
        if (threadIdx.x < RX) {
            const int halo_x = dev_max_i(block_x0 + (int)threadIdx.x - RX, 0);
            srow[threadIdx.x] = gray[gy * src_stride + halo_x];
        } else if (threadIdx.x >= blockDim.x - RX) {
            const int halo_x = dev_min_i(block_x0 + (int)threadIdx.x + RX, width - 1);
            srow[threadIdx.x + 2 * RX] = gray[gy * src_stride + halo_x];
        }
    }
    __syncthreads();
    if (x >= width) return;

    const unsigned char center = s[RY * row_stride + (threadIdx.x + RX)];
    uint32_t cv = 0;
#pragma unroll
    for (int dy = 0; dy < WIN_H; ++dy) {
        const unsigned char* srow = s + dy * row_stride;
#pragma unroll
        for (int dx = 0; dx < WIN_W; ++dx) {
            cv <<= 1;
            cv |= (srow[threadIdx.x + dx] < center);
        }
    }
    census[y * dst_stride + x] = cv;
}

// 纯 Census Hamming 代价（不含 AD / CombinedCost）
#define CENSUS_COST_TILE_X 64

__global__ void kernel_CensusCost(
    const uint32_t* __restrict__ census_left,
    const uint32_t* __restrict__ census_right,
    short* __restrict__ cost_volume,
    int width, int height,
    int census_stride,
    int numDisp, int minDisp)
{
    const int x0 = blockIdx.x * CENSUS_COST_TILE_X;
    const int y = blockIdx.y;
    const int d = threadIdx.x;
    if (x0 >= width || y >= height || d >= numDisp) return;

    const int x_end = dev_min_i(x0 + CENSUS_COST_TILE_X, width);
    for (int x = x0; x < x_end; ++x) {
        int x2 = x - (d + minDisp);
        short total_cost = 24;  // 5x5 描述子的中心位恒为 0，共 24 个有效比较位
        if (x2 >= 0 && x2 < width) {
            const uint32_t left_cv = census_left[y * census_stride + x];
            const uint32_t right_cv = census_right[y * census_stride + x2];
            total_cost = (short)dev_hamming32(left_cv, right_cv);
        }
        cost_volume[(y * width + x) * numDisp + d] = total_cost;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 3: 可分离 SAD box filter 水平 pass（求和，不归一化）
//
// 对标 OpenCV getRawMatchingCost() 的 box filter 求和
// 输出 = 水平方向 blockSize 个像素的代价之和
//
// 内存布局说明：代价体是 [y][x][d]，d 为最内层连续维度。
// 线程映射为 d = threadIdx.x（一个 block 处理一整行 y），这样同一 warp 内
// 线程的 d 连续、地址相邻，读写都能合并成一次内存事务；
// 原实现用 x = threadIdx.x（d 固定在 blockIdx.z），warp 内相邻线程地址相差
// numDisp 个 short（对齐后 256 字节），导致内存访问完全无法合并，是原先
// box filter 阶段耗时占比过高（>75% GPU 时间）的根因。
//
// 滑动窗口增量累加：每个线程独立处理一整行（H）/一整列（V）里固定的 d，
// 各线程之间读写的地址完全不重叠（thread d 只读写自己那个 d 对应的元素），
// 没有跨线程可复用的数据，因此共享内存缓存整行/整列在这里换不来额外收益，
// 反而会占用宝贵的共享内存、可能降低占用率；而滑动窗口只需要线程自己的
// 寄存器（一个累加和），把每像素 O(blockSize) 次全量重求和降到 O(1) 次
// 增量更新（减去滑出窗口的一个元素、加上滑入窗口的一个元素），blockSize
// 越大收益越明显。递推公式：
//   sum(x) = sum(x-1) - src[clamp(x-1-half, 0, W-1)] + src[clamp(x+half, 0, W-1)]
// 该递推对边界钳位（重复计边界像素）依然精确成立，因为它只是把逐项 clamp
// 求和按“未钳位前的下标差”做了差分，结果与原始全量重求和逐像素严格一致。
// ─────────────────────────────────────────────────────────────────────────────

__global__ void kernel_BoxFilterH(
    const short* __restrict__ src, short* __restrict__ dst,
    int width, int height, int numDisp, int blockSize)
{
    int y = blockIdx.x;
    int d = threadIdx.x;
    if (y >= height || d >= numDisp) return;

    int half = blockSize / 2;
    int row_base = y * width * numDisp;

    // x = 0 的初始窗口和，与原始逐项 clamp 求和公式完全一致
    int sum = 0;
    for (int k = -half; k <= half; ++k) {
        int xx = dev_min_i(dev_max_i(k, 0), width - 1);
        sum += src[row_base + xx * numDisp + d];
    }
    dst[row_base + d] = (short)dev_min_i(SHRT_MAX, sum);

    for (int x = 1; x < width; ++x) {
        int leave_x = dev_min_i(dev_max_i(x - 1 - half, 0), width - 1);
        int enter_x = dev_min_i(dev_max_i(x + half, 0), width - 1);
        sum += (int)src[row_base + enter_x * numDisp + d] - (int)src[row_base + leave_x * numDisp + d];
        dst[row_base + x * numDisp + d] = (short)dev_min_i(SHRT_MAX, sum);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 4: 可分离 SAD box filter 垂直 pass（求和，不归一化）
//
// 最终 SAD 代价 = 垂直方向 blockSize 个水平 sum 之和
// = 整个 blockSize x blockSize 窗口内 BT 代价的总和
//
// 同上：block 处理一整列 x，线程对应 d，保证 warp 内地址连续。
// ─────────────────────────────────────────────────────────────────────────────

__global__ void kernel_BoxFilterV(
    const short* __restrict__ src, short* __restrict__ dst,
    int width, int height, int numDisp, int blockSize)
{
    int x = blockIdx.x;
    int d = threadIdx.x;
    if (x >= width || d >= numDisp) return;

    int half = blockSize / 2;
    int stride = width * numDisp;
    int col_base = x * numDisp + d;

    // y = 0 的初始窗口和
    int sum = 0;
    for (int k = -half; k <= half; ++k) {
        int yy = dev_min_i(dev_max_i(k, 0), height - 1);
        sum += src[yy * stride + col_base];
    }
    dst[col_base] = (short)dev_min_i(SHRT_MAX, sum);

    for (int y = 1; y < height; ++y) {
        int leave_y = dev_min_i(dev_max_i(y - 1 - half, 0), height - 1);
        int enter_y = dev_min_i(dev_max_i(y + half, 0), height - 1);
        sum += (int)src[enter_y * stride + col_base] - (int)src[leave_y * stride + col_base];
        dst[y * stride + col_base] = (short)dev_min_i(SHRT_MAX, sum);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 共享内存布局辅助函数
// ─────────────────────────────────────────────────────────────────────────────

// 2-buffer 布局：L_prev[0..D-1], L_cur[0..D-1]
__device__ inline short* smem_L_prev(short* smem) { return smem; }
__device__ inline short* smem_L_cur(short* smem, int D) { return smem + D; }

// 4-buffer 布局（用于 RL+WTA 核）：
//   L_prev[0..D-1], L_cur[0..D-1], disp2Buf[0..W-1], disp2CostBuf[0..W-1]
__device__ inline short* smem_disp2Buf(short* smem, int D)     { return smem + 2 * D; }
__device__ inline short* smem_disp2CostBuf(short* smem, int D, int W) { return smem + 2 * D + W; }

// ─────────────────────────────────────────────────────────────────────────────
// block 级 min 归约，warp shuffle 版。
//
// 调用约定：blockDim.x == numDisp（调用方核函数均满足），线程 d 传入自己的
// 值 my，red 为至少 ceil(numDisp/32) 个元素的共享内存缓冲。返回全 block 最小值。
//
// 原树形归约每步都要 __syncthreads（log2(D)≈7 次 barrier），且位于 SGM 路径核
// 对 x/y 的串行主循环内，barrier 开销是热点。改为：warp 内 __shfl_down_sync
// 归约（寄存器交换、零 barrier）→ 各 warp 部分最小值写共享内存 → 1 次
// __syncthreads → 线程 0 串行合并 ≤ numDisp/32 个部分值 → 再 1 次
// __syncthreads 广播。共 2 次 barrier。
//
// 函数入口的 __syncthreads 承接调用方对共享内存（L_cur 等）的写入可见性，
// 语义与原实现中“写入后先同步再归约”一致。
//
// 部分 warp 处理：numDisp 为 16 的倍数，因此每个 warp 的活跃线程数只能是
// 32 或 16（最后一个 warp）。按活跃线程数收窄 shuffle 掩码和起始偏移，
// 保证不会从不活跃 lane 读值（那是未定义行为）。
// min 满足结合律与交换律，任何归约顺序结果都与串行扫描严格一致。
// ─────────────────────────────────────────────────────────────────────────────
__device__ inline int dev_warp_min_int(int v, int lane_count)
{
    const unsigned mask = (lane_count >= 32) ? 0xFFFFFFFFu
                                             : ((1u << lane_count) - 1u);
    for (int off = lane_count >> 1; off > 0; off >>= 1) {
        int other = __shfl_down_sync(mask, v, off);
        v = dev_min_i(v, other);
    }
    return v;
}

__device__ inline int dev_block_min_i(int my, int* red, int d, int numDisp)
{
    __syncthreads();
    const int lane = d & 31;
    const int wid  = d >> 5;
    const int lane_count = dev_min_i(32, numDisp - (wid << 5));
    int v = dev_warp_min_int(my, lane_count);
    if (lane == 0) red[wid] = v;
    __syncthreads();
    if (d == 0) {
        const int num_warps = (numDisp + 31) >> 5;
        int m = red[0];
        for (int i = 1; i < num_warps; ++i) m = dev_min_i(m, red[i]);
        red[0] = m;
    }
    __syncthreads();
    return red[0];
}

__device__ inline int dev_block_sum_i(int my, int* red, int d, int numDisp)
{
    __syncthreads();
    const int lane = d & 31;
    const int wid = d >> 5;
    const int lane_count = dev_min_i(32, numDisp - (wid << 5));
    const unsigned mask = lane_count >= 32 ? 0xFFFFFFFFu
                                            : ((1u << lane_count) - 1u);
    int value = my;
    for (int off = lane_count >> 1; off > 0; off >>= 1) {
        value += __shfl_down_sync(mask, value, off);
    }
    if (lane == 0) red[wid] = value;
    __syncthreads();
    if (d == 0) {
        const int num_warps = (numDisp + 31) >> 5;
        int sum = 0;
        for (int i = 0; i < num_warps; ++i) sum += red[i];
        red[0] = sum;
    }
    __syncthreads();
    return red[0];
}

// unsigned 版本（用于 (value<<16)|index 打包 key 的 argmin 归约）。
// key 非负且 < 2^31，直接复用 int min 归约即可保序。
__device__ inline unsigned int dev_block_min_u32(unsigned int my, int* red, int d, int numDisp)
{
    return (unsigned int)dev_block_min_i((int)my, red, d, numDisp);
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 5: SGM 路径 1 —— 左→右 水平聚合
//
// 对标 OpenCV accumulateCostsLeftTop() 的 leftBuf 计算
//   读取原始 SAD 代价 C，写入累计缓冲区 cost_acc
//   L_r(p, d) = C(p, d) + min(L_r(p-r, d), L_r(p-r, d-1)+P1, L_r(p-r, d+1)+P1, min_k(L_r(p-r, k))+P2) - min_k(L_r(p-r, k))
// ─────────────────────────────────────────────────────────────────────────────

__global__ void kernel_SGM_PathLR(
    short* cost_acc, const short* C_sad,
    const unsigned char* __restrict__ img, int img_stride,
    int width, int height, int numDisp, int startX, int endX, int P1, int P2,
    int adaptiveP2, float p2Alpha, float p2Beta, float p2LowerMult, float p2UpperMult)
{
    // int 声明保证基址 4 字节对齐（其中一段用作 int 归约缓冲）
    extern __shared__ int smem_i[];
    short* smem = reinterpret_cast<short*>(smem_i);
    int y = blockIdx.x;
    int d = threadIdx.x;
    if (y >= height || d >= numDisp) return;

    short* L_prev = smem_L_prev(smem);
    short* L_cur  = smem_L_cur(smem, numDisp);
    int*   red_i  = smem_i + numDisp;   // 2*numDisp short = numDisp int，偏移恰好对齐

    int stride = width * numDisp;
    int row_base = y * stride;
    if (startX < 0) startX = 0;
    if (endX > width) endX = width;
    if (startX >= endX) return;

    // OpenCV 3WAY 只在 [minX1, maxX1) 有效范围内聚合，避免左侧不可匹配区域污染路径代价。
    // 原实现由 d==0 单线程串行扫描 numDisp 个元素求 min，期间其余线程全部
    // 空转；该扫描位于对 x 的串行主循环内，是整个核的热点。
    // 现改为 warp shuffle 块级归约（dev_block_min_i，仅 2 次 barrier）。
    L_prev[d] = C_sad[row_base + startX * numDisp + d];
    short min_prev = (short)dev_block_min_i((int)L_prev[d], red_i, d, numDisp);

    cost_acc[row_base + startX * numDisp + d] = L_prev[d];

    const unsigned char* img_row = adaptiveP2 ? (img + y * img_stride) : nullptr;
    for (int x = startX + 1; x < endX; ++x) {
        short c_val = C_sad[row_base + x * numDisp + d];

        int p2_eff = adaptiveP2
            ? dev_adaptive_p2(P2, (int)img_row[x], (int)img_row[x - 1],
                              p2Alpha, p2Beta, p2LowerMult, p2UpperMult)
            : P2;
        int L_minus = (d > 0) ? (int)L_prev[d - 1] + P1 : SHRT_MAX;
        int L_self  = (int)L_prev[d];
        int L_plus  = (d < numDisp - 1) ? (int)L_prev[d + 1] + P1 : SHRT_MAX;
        int L_jump  = dev_min_i(SHRT_MAX, (int)min_prev + p2_eff);
        int best = dev_min_i(dev_min_i(L_minus, L_plus), dev_min_i(L_self, L_jump));
        L_cur[d] = dev_saturate_short((int)c_val + best - (int)min_prev);

        // 块级 min 归约（min 满足结合律，与串行扫描结果严格一致）；
        // 归约入口的 __syncthreads 同时保证 L_cur 全部写完。
        min_prev = (short)dev_block_min_i((int)L_cur[d], red_i, d, numDisp);

        cost_acc[row_base + x * numDisp + d] = L_cur[d];

        // 下一轮迭代读 L_prev[d±1]（本轮的 L_cur）之前，必须保证所有线程都
        // 已经过一次 barrier；归约函数尾部的 __syncthreads 已覆盖，
        // 此处的指针交换是线程局部操作，无需额外同步。
        short* tmp = L_prev; L_prev = L_cur; L_cur = tmp;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 6: SGM 路径 2 —— 上→下 垂直聚合
//
// 对标 OpenCV accumulateCostsLeftTop() 的 topBuf 计算
//   读取原始 SAD 代价 C，累加到 cost_acc
// ─────────────────────────────────────────────────────────────────────────────

__global__ void kernel_SGM_PathTB(
    short* cost_acc, const short* C_sad,
    const unsigned char* __restrict__ img, int img_stride,
    int width, int height, int numDisp, int P1, int P2,
    int adaptiveP2, float p2Alpha, float p2Beta, float p2LowerMult, float p2UpperMult)
{
    // int 声明保证基址 4 字节对齐（其中一段用作 int 归约缓冲）
    extern __shared__ int smem_i[];
    short* smem = reinterpret_cast<short*>(smem_i);
    int x = blockIdx.x;
    int d = threadIdx.x;
    if (x >= width || d >= numDisp) return;

    short* L_prev = smem_L_prev(smem);
    short* L_cur  = smem_L_cur(smem, numDisp);
    int*   red_i  = smem_i + numDisp;   // 2*numDisp short = numDisp int，偏移恰好对齐

    int stride = width * numDisp;

    // 初始化：y=0
    L_prev[d] = C_sad[x * numDisp + d];

    // warp shuffle 块级 min 归约（入口/出口自带 __syncthreads）
    short min_prev = (short)dev_block_min_i((int)L_prev[d], red_i, d, numDisp);

    // y=0：累加 Path2 到 cost_acc
    int out_idx = x * numDisp + d;
    cost_acc[out_idx] = (short)dev_min_i(SHRT_MAX, (int)cost_acc[out_idx] + L_prev[d]);

    for (int y = 1; y < height; ++y) {
        short c_val = C_sad[y * stride + x * numDisp + d];

        int p2_eff = adaptiveP2
            ? dev_adaptive_p2(P2, (int)img[y * img_stride + x], (int)img[(y - 1) * img_stride + x],
                              p2Alpha, p2Beta, p2LowerMult, p2UpperMult)
            : P2;
        int L_minus = (d > 0) ? (int)L_prev[d - 1] + P1 : SHRT_MAX;
        int L_self  = (int)L_prev[d];
        int L_plus  = (d < numDisp - 1) ? (int)L_prev[d + 1] + P1 : SHRT_MAX;
        int L_jump  = dev_min_i(SHRT_MAX, (int)min_prev + p2_eff);
        int best = dev_min_i(dev_min_i(L_minus, L_plus), dev_min_i(L_self, L_jump));
        L_cur[d] = dev_saturate_short((int)c_val + best - (int)min_prev);

        // 块级 min 归约求新的 min_prev
        min_prev = (short)dev_block_min_i((int)L_cur[d], red_i, d, numDisp);

        out_idx = y * stride + x * numDisp + d;
        cost_acc[out_idx] = (short)dev_min_i(SHRT_MAX, (int)cost_acc[out_idx] + L_cur[d]);

        short* tmp = L_prev; L_prev = L_cur; L_cur = tmp;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 7: SGM 路径 3 —— 右→左 水平逆聚合 + WTA 视差选择
//
// 对标 OpenCV accumulateCostsRight()
//   读取原始 SAD 代价 C，累加到 cost_acc（三路径求和）
//   然后 WTA + 唯一性检查 + 子像素插值 + 伪 LR 检查
//
// 性能说明：原实现中 min_prev 归约以及 WTA/唯一性检查都是由单线程（d==0）
// 串行扫描 numDisp 个候选完成，block 内其余 numDisp-1 个线程在此期间空闲。
// 这里改为用已有的 numDisp 个线程做并行树形归约，逻辑与原始串行版本严格等价：
//   - min_prev/alt_cost 只需要“值”，用普通 sequential-addressing 归约即可
//     （min 满足结合律/交换律，归约顺序不影响最终值）；
//   - min_cost + best_d 是 argmin，原始串行 for 循环在代价相同（tie）时
//     总是保留最左侧（最小 k）的下标。若直接对 (value, index) 做
//     sequential-addressing 树形归约，归约的合并顺序不是严格从左到右，
//     会在有 tie 的情况下选出错误的下标（已用随机仿真验证并复现此 bug）。
//     修复方式：把 (value, index) 打包成单个整数 key = (value << 16) | index
//     再做普通 min 归约——value 在高位，天然保证“先比大小、tie 时比下标”，
//     结果与归约顺序无关，任意归约方式都能得到与串行扫描完全一致的结果。
//   - 唯一性检查原本是第三个串行循环，数学上等价于
//     `alt_cost*(100-ratio) < min_cost*100`（因为该判据关于 cost 单调，
//     alt_cost 是候选里最小、最容易触发的一个），故直接复用 alt_cost
//     归约结果，省掉整整一次 O(numDisp) 扫描。
// ─────────────────────────────────────────────────────────────────────────────

__global__ void kernel_SGM_PathRL_WTA(
    short* cost_acc, const short* C_sad,
    const unsigned char* __restrict__ img, int img_stride,
    short* disp,
    short* disp2,
    short* disp2_cost,
    float* conf_raw,
    int width, int height, int numDisp, int minDisp,
    int P1, int P2, int uniquenessRatio, int disp12MaxDiff,
    int accumulate,
    int adaptiveP2, float p2Alpha, float p2Beta, float p2LowerMult, float p2UpperMult)
{
    // 用 int 类型声明动态共享内存以保证基址 4 字节对齐——
    // 若声明为 `extern __shared__ short smem[]` 只保证 2 字节对齐，
    // 之后把其中一段 reinterpret_cast 成 int* 做归约（red_i）
    // 就可能落在非 4 字节对齐地址上，是未定义行为。
    // 从 4 字节对齐的 int 基址派生 short* 视图则总是安全的（int 对齐更严格）。
    extern __shared__ int smem_i[];
    short* smem = reinterpret_cast<short*>(smem_i);
    int y = blockIdx.x;
    int d = threadIdx.x;
    if (y >= height || d >= numDisp) return;

    short* L_prev = smem_L_prev(smem);
    short* L_cur  = smem_L_cur(smem, numDisp);
    // warp shuffle 归约只需 ceil(numDisp/32) 个 int 的部分最小值缓冲，
    // 三次归约（min_prev / argmin key / alt_cost）串行发生，可复用同一块。
    int* red_i = smem_i + numDisp;   // 2*numDisp short = numDisp int，偏移恰好对齐

    int stride = width * numDisp;
    int row_base = y * stride;
    const short invalid_disp = (short)((minDisp - 1) * 16);
    const int minX1 = dev_max_i(minDisp + numDisp, 0);
    const int maxX1 = width + dev_min_i(minDisp, 0);

    // 初始化：x = width-1
    int x_start = width - 1;
    L_prev[d] = C_sad[row_base + x_start * numDisp + d];

    short min_prev = (short)dev_block_min_i((int)L_prev[d], red_i, d, numDisp);

    // 路径 3 累加到 x=width-1。该列通常位于 OpenCV 有效范围内，
    // 但这里保持 invalid，避免边界未完成 LRC 时输出随机值。
    // accumulate=0 时说明前面的路径（PathLR/PathTB）被跳过了，cost_acc 尚未
    // 初始化，这里改成直接赋值而不是累加，避免读到未初始化的显存内容。
    int idx = row_base + x_start * numDisp + d;
    cost_acc[idx] = (short)dev_min_i(SHRT_MAX, (accumulate ? (int)cost_acc[idx] : 0) + L_prev[d]);
    if (d == 0) {
        disp[y * width + x_start] = invalid_disp;
        conf_raw[y * width + x_start] = 0.0f;
    }
    __syncthreads();

    const unsigned char* img_row = adaptiveP2 ? (img + y * img_stride) : nullptr;
    for (int x = width - 2; x >= 0; --x) {
        short c_val = C_sad[row_base + x * numDisp + d];

        // 右→左路径的“上一个像素”是 x+1
        int p2_eff = adaptiveP2
            ? dev_adaptive_p2(P2, (int)img_row[x], (int)img_row[x + 1],
                              p2Alpha, p2Beta, p2LowerMult, p2UpperMult)
            : P2;
        int L_minus = (d > 0) ? (int)L_prev[d - 1] + P1 : SHRT_MAX;
        int L_self  = (int)L_prev[d];
        int L_plus  = (d < numDisp - 1) ? (int)L_prev[d + 1] + P1 : SHRT_MAX;
        int L_jump  = dev_min_i(SHRT_MAX, (int)min_prev + p2_eff);
        int best = dev_min_i(dev_min_i(L_minus, L_plus), dev_min_i(L_self, L_jump));
        L_cur[d] = dev_saturate_short((int)c_val + best - (int)min_prev);

        // 块级 min 归约求新的 min_prev（归约入口 barrier 同时保证 L_cur 写完）
        min_prev = (short)dev_block_min_i((int)L_cur[d], red_i, d, numDisp);

        // 累加各路径代价（accumulate=0 时为赋值，见函数开头说明）。
        // my_cost 就是本线程刚写入的值，直接用寄存器，无需回读全局内存。
        idx = row_base + x * numDisp + d;
        int total = (accumulate ? (int)cost_acc[idx] : 0) + L_cur[d];
        short my_cost = (short)dev_min_i(SHRT_MAX, total);
        cost_acc[idx] = my_cost;

        // ── 归约 1：求 (min_cost, best_d) ──
        // 打包成单一 key = (value << 16) | index 再做 min 归约：
        // value 在高位保证先比代价大小，index 在低位保证 tie 时选最小下标，
        // 与串行 for 循环的“最左侧最小值”语义严格一致，且与归约顺序无关。
        // 归约入口 barrier 同时保证全 block 的 cost_acc 写入对线程 0 可见
        // （线程 0 在 WTA 段要读 best_d 邻域的 cost_acc 做子像素插值）。
        unsigned int my_key = (((unsigned int)(unsigned short)my_cost) << 16) | (unsigned int)d;
        unsigned int best_key = dev_block_min_u32(my_key, red_i, d, numDisp);
        short min_cost = (short)(best_key >> 16);
        short best_d   = (short)(best_key & 0xFFFFu);

        // ── 归约 2：求 alt_cost（排除 best_d 邻域 |k-best_d|<=1）──
        int my_alt = (abs(d - (int)best_d) > 1) ? (int)my_cost : (int)SHRT_MAX;
        short alt_cost = (short)dev_block_min_i(my_alt, red_i, d, numDisp);

        // ── WTA 结果确定后：唯一性检查 + 子像素插值 + 伪 LR 检查（线程 0）──
        if (d == 0) {
            if (x < minX1 || x >= maxX1) {
                disp[y * width + x] = invalid_disp;
                conf_raw[y * width + x] = 0.0f;
            } else {
                // 唯一性检查（与原三重串行循环数学等价，见函数头注释）
                bool unique = true;
                if (uniquenessRatio > 0 && min_cost < SHRT_MAX / 2) {
                    if ((int)alt_cost * (100 - uniquenessRatio) < (int)min_cost * 100) {
                        unique = false;
                    }
                }

                if (unique && min_cost < SHRT_MAX / 2) {
                    // 3. 子像素插值（对标 OpenCV 公式）
                    // OpenCV: denom2 = max(Sp[d-1]+Sp[d+1]-2*Sp[d], 1)
                    //         d = d*16 + ((Sp[d-1]-Sp[d+1])*16 + denom2) / (denom2*2)
                    int d_int;
                    if (best_d > 0 && best_d < numDisp - 1) {
                        short c0 = cost_acc[row_base + x * numDisp + (best_d - 1)];
                        short c1 = cost_acc[row_base + x * numDisp + best_d];
                        short c2 = cost_acc[row_base + x * numDisp + (best_d + 1)];
                        int denom2 = dev_max_i((int)c0 + c2 - 2 * (int)c1, 1);
                        d_int = best_d * 16 + ((int)(c0 - c2) * 16 + denom2) / (denom2 * 2);
                    } else {
                        d_int = best_d * 16;
                    }

                    short d_val = (short)(d_int + minDisp * 16);

                    // 4. 记录伪右图视差，完整 LRC 在后续 kernel 中执行。
                    int x2 = x - best_d - minDisp;
                    if (x2 >= 0 && x2 < width) {
                        int out = y * width + x2;
                        if (disp2_cost[out] > min_cost) {
                            disp2_cost[out] = min_cost;
                            disp2[out] = (short)(best_d + minDisp);
                        }
                    }

                    disp[y * width + x] = d_val;
                    float denom = fmaxf((float)min_cost, 1.0f);
                    float conf = ((float)alt_cost - (float)min_cost) / denom;
                    conf_raw[y * width + x] = fminf(fmaxf(conf, 0.0f), 1.0f);
                } else {
                    disp[y * width + x] = invalid_disp;
                    conf_raw[y * width + x] = 0.0f;
                }
            }
        }

        // 指针交换是线程局部操作。下一轮迭代对旧 L_prev（新 L_cur）的写入
        // 发生在归约入口 barrier 之后才会被读取，且线程 0 的 WTA 段只读
        // cost_acc/寄存器，不读 L_prev/red_val，因此这里无需额外 barrier。
        short* tmp = L_prev; L_prev = L_cur; L_cur = tmp;
    }

    // 处理 x=0（最左列）的 LR 检查 —— 对标 OpenCV impl() 末尾的 LRC 检查
    // 已在循环中处理，这里不需要额外处理
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 8: 伪左右一致性检查 + 中值滤波 3x3
// ─────────────────────────────────────────────────────────────────────────────

__global__ void kernel_PseudoLRCheck(
    short* disp,
    const short* disp2,
    float* conf_raw,
    int width, int height,
    int minDisp, int disp12MaxDiff)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const short invalid_disp = (short)((minDisp - 1) * 16);
    int idx = y * width + x;
    int d1 = disp[idx];
    if (d1 == invalid_disp) return;

    int _d = d1 >> 4;
    int d_ = (d1 + 15) >> 4;
    int _x = x - _d;
    int x_ = x - d_;
    const short* row_disp2 = disp2 + y * width;
    if (0 <= _x && _x < width && row_disp2[_x] >= minDisp && abs(row_disp2[_x] - _d) > disp12MaxDiff &&
        0 <= x_ && x_ < width && row_disp2[x_] >= minDisp && abs(row_disp2[x_] - d_) > disp12MaxDiff) {
        disp[idx] = invalid_disp;
        conf_raw[idx] = 0.0f;
    }
}

// Pyramid priors are never recovered by a wider fine-level search, so a
// missing right-side projection is not allowed to pass as "unknown".
__global__ void kernel_StrictPyramidLRCheck(
    short* disp,
    const short* disp2,
    float* conf_raw,
    int width, int height,
    int minDisp, int disp12MaxDiff)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const short invalidDisp = (short)((minDisp - 1) * 16);
    int p = y * width + x;
    int d16 = disp[p];
    if (d16 == invalidDisp) return;

    int d0 = d16 >> 4;
    int d1 = (d16 + 15) >> 4;
    int rx0 = x - d0;
    int rx1 = x - d1;
    const short* rightRow = disp2 + y * width;
    bool consistent = false;
    if (rx0 >= 0 && rx0 < width && rightRow[rx0] >= minDisp &&
        abs((int)rightRow[rx0] - d0) <= disp12MaxDiff) {
        consistent = true;
    }
    if (rx1 >= 0 && rx1 < width && rightRow[rx1] >= minDisp &&
        abs((int)rightRow[rx1] - d1) <= disp12MaxDiff) {
        consistent = true;
    }
    if (!consistent) {
        disp[p] = invalidDisp;
        conf_raw[p] = 0.0f;
    }
}

__global__ void kernel_MedianFilter3x3(
    const short* src, short* dst,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    short vals[9];
    int n = 0;

    for (int dy = -1; dy <= 1; ++dy) {
        int yy = y + dy;
        if (yy < 0 || yy >= height) continue;
        for (int dx = -1; dx <= 1; ++dx) {
            int xx = x + dx;
            if (xx < 0 || xx >= width) continue;
            vals[n++] = src[yy * width + xx];
        }
    }

    // 排序取中值（插入排序，n <= 9）
    for (int i = 1; i < n; ++i) {
        short key = vals[i];
        int j = i - 1;
        while (j >= 0 && vals[j] > key) { vals[j + 1] = vals[j]; --j; }
        vals[j + 1] = key;
    }
    dst[y * width + x] = vals[n / 2];
}

__global__ void kernel_ClearChangedMedianConfidence(
    const short* before, const short* after, float* conf,
    int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int p = y * width + x;
    if (before[p] != after[p]) conf[p] = 0.0f;
}

// ─────────────────────────────────────────────────────────────────────────────
// Pyramid coarse-to-fine helpers.
//
// Every refined pixel owns a contiguous local label interval:
//   real disparity = base_disp[p] + local_label.
// Path recursion maps the real disparity back into the previous pixel's local
// interval.  This is essential: equal local indices are not necessarily equal
// disparities when neighboring base values differ.
// ─────────────────────────────────────────────────────────────────────────────

__global__ void kernel_InitCoarseLabels(
    short* base_disp, unsigned char* valid_mask,
    int width, int height, int minDisp)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    int idx = y * width + x;
    base_disp[idx] = (short)minDisp;
    valid_mask[idx] = 1;
}

__global__ void kernel_PrepareRefineLabels(
    const short* parent_disp, const float* parent_conf,
    int parent_width, int parent_height,
    int parent_disp_pitch, int parent_conf_pitch,
    short* base_disp, unsigned char* valid_mask,
    int width, int height,
    int minDisp, int maxDisp, int localD,
    float minConfidence)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int px = dev_min_i(x >> 1, parent_width - 1);
    const int py = dev_min_i(y >> 1, parent_height - 1);
    const short raw = parent_disp[py * parent_disp_pitch + px];
    const float conf = parent_conf[py * parent_conf_pitch + px];
    const short parent_invalid = (short)((minDisp - 1) * 16);
    const int out = y * width + x;

    if (raw == parent_invalid || conf < minConfidence) {
        base_disp[out] = (short)minDisp;
        valid_mask[out] = 0;
        return;
    }

    // Keep the parent's 1/16-pixel fraction while changing level units.
    const int predicted16 = (int)raw * 2;
    const int center = (predicted16 + 8) >> 4;
    const int residual16 = predicted16 - center * 16;
    int base = center + (residual16 < 0 ? -(localD / 2) : -(localD / 2) + 1);

    // Keep a full local interval whenever geometry permits.  At the first
    // columns fewer than localD physical candidates exist; individual labels
    // are masked by the cost/path kernels.
    const int physicalMax = dev_min_i(maxDisp, x);
    const int largestBase = physicalMax - localD + 1;
    if (largestBase >= minDisp) {
        base = dev_min_i(dev_max_i(base, minDisp), largestBase);
    } else {
        base = minDisp;
    }

    base_disp[out] = (short)base;
    valid_mask[out] = 1;
}

__device__ inline int dev_bt_channel_cost(
    const unsigned char* left, const unsigned char* right,
    int stride_left, int stride_right, int width,
    int y, int x1, int x2)
{
    if (x2 < 0 || x2 >= width) return 255;
    int x1m = dev_max_i(x1 - 1, 0);
    int x1p = dev_min_i(x1 + 1, width - 1);
    int x2m = dev_max_i(x2 - 1, 0);
    int x2p = dev_min_i(x2 + 1, width - 1);

    int u = left[y * stride_left + x1];
    int ul = (u + left[y * stride_left + x1m]) / 2;
    int ur = (u + left[y * stride_left + x1p]) / 2;
    int u0 = dev_min_i(dev_min_i(ul, ur), u);
    int u1 = dev_max_i(dev_max_i(ul, ur), u);

    int v = right[y * stride_right + x2];
    int vl = (v + right[y * stride_right + x2m]) / 2;
    int vr = (v + right[y * stride_right + x2p]) / 2;
    int v0 = dev_min_i(dev_min_i(vl, vr), v);
    int v1 = dev_max_i(dev_max_i(vl, vr), v);

    int c0 = dev_max_i(dev_max_i(0, u - v1), v0 - u);
    int c1 = dev_max_i(dev_max_i(0, v - u1), u0 - v);
    return dev_min_i(c0, c1);
}

__device__ inline int dev_bt_pixel_cost(
    const unsigned char* grad1, const unsigned char* grad2,
    const unsigned char* img1, const unsigned char* img2,
    int grad_stride, int img1_stride, int img2_stride,
    int width, int y, int x, int realDisp)
{
    int x2 = x - realDisp;
    if (x2 < 0 || x2 >= width) return 255;
    int cg = dev_bt_channel_cost(grad1, grad2, grad_stride, grad_stride,
                                 width, y, x, x2);
    int ci = dev_bt_channel_cost(img1, img2, img1_stride, img2_stride,
                                 width, y, x, x2);
    return dev_min_i(255, cg + (ci >> 2));
}

__device__ inline bool dev_pyramid_label_geometry_valid(
    const unsigned char* valid_mask, const short* base_disp,
    int p, int x, int d, int numDisp, int minDisp, int maxDisp, int width)
{
    if (!valid_mask[p] || d < 0 || d >= numDisp) return false;
    int realDisp = (int)base_disp[p] + d;
    int x2 = x - realDisp;
    return realDisp >= minDisp && realDisp <= maxDisp &&
           x2 >= 0 && x2 < width;
}

// Local pyramid labels are contiguous in real-disparity space.  Intersect
// [base, base + numDisp - 1] with both the configured disparity range and
// the x-projection range instead of reducing per-label cost sentinels.
__device__ inline int dev_pyramid_valid_label_count(
    const unsigned char* valid_mask, const short* base_disp,
    int p, int x, int numDisp, int minDisp, int maxDisp, int width)
{
    if (!valid_mask[p] || numDisp <= 0) return 0;
    int base = (int)base_disp[p];
    int firstReal = dev_max_i(minDisp, x - (width - 1));
    int lastReal = dev_min_i(maxDisp, x);
    int firstLabel = dev_max_i(0, firstReal - base);
    int lastLabel = dev_min_i(numDisp - 1, lastReal - base);
    return lastLabel >= firstLabel ? lastLabel - firstLabel + 1 : 0;
}

// A separable filter over local index k would mix different real disparities
// when base_disp changes between pixels.  Compute the small BT support window
// directly for each of the 16 real-disparity candidates instead.
__global__ void kernel_BTWindowCostLocal(
    const unsigned char* grad1, const unsigned char* grad2,
    const unsigned char* img1, const unsigned char* img2,
    const short* base_disp, const unsigned char* valid_mask,
    short* cost_volume,
    int width, int height, int grad_stride, int img1_stride, int img2_stride,
    int numDisp, int minDisp, int maxDisp, int blockSize)
{
    int d = threadIdx.x;
    int x = blockIdx.x * blockDim.y + threadIdx.y;
    int y = blockIdx.y;
    if (x >= width || y >= height || d >= numDisp) return;

    int p = y * width + x;
    int realDisp = (int)base_disp[p] + d;
    bool labelValid = dev_pyramid_label_geometry_valid(
        valid_mask, base_disp, p, x, d, numDisp, minDisp, maxDisp, width);
    int out = (p * numDisp) + d;
    if (!labelValid) {
        cost_volume[out] = SHRT_MAX;
        return;
    }

    int half = blockSize / 2;
    int sum = 0;
    for (int dy = -half; dy <= half; ++dy) {
        int yy = dev_min_i(dev_max_i(y + dy, 0), height - 1);
        for (int dx = -half; dx <= half; ++dx) {
            int xx = dev_min_i(dev_max_i(x + dx, 0), width - 1);
            sum += dev_bt_pixel_cost(grad1, grad2, img1, img2,
                                     grad_stride, img1_stride, img2_stride,
                                     width, yy, xx, realDisp);
        }
    }
    cost_volume[out] = (short)dev_min_i(sum, SHRT_MAX - 1);
}

__global__ void kernel_CensusCostLocal(
    const uint32_t* census_left, const uint32_t* census_right,
    const short* base_disp, const unsigned char* valid_mask,
    short* cost_volume,
    int width, int height, int census_stride,
    int numDisp, int minDisp, int maxDisp)
{
    int d = threadIdx.x;
    int x = blockIdx.x * blockDim.y + threadIdx.y;
    int y = blockIdx.y;
    if (x >= width || y >= height || d >= numDisp) return;

    int p = y * width + x;
    int realDisp = (int)base_disp[p] + d;
    int x2 = x - realDisp;
    bool labelValid = dev_pyramid_label_geometry_valid(
        valid_mask, base_disp, p, x, d, numDisp, minDisp, maxDisp, width);
    cost_volume[p * numDisp + d] = labelValid
        ? (short)dev_hamming32(census_left[y * census_stride + x],
                               census_right[y * census_stride + x2])
        : (short)SHRT_MAX;
}

__device__ inline bool dev_pyramid_label_valid(
    const short* C, const unsigned char* valid_mask, const short* base_disp,
    int p, int x, int d, int numDisp, int minDisp, int maxDisp, int width)
{
    return dev_pyramid_label_geometry_valid(
               valid_mask, base_disp, p, x, d,
               numDisp, minDisp, maxDisp, width) &&
           C[p * numDisp + d] < SHRT_MAX;
}

// Pyramid path kernels use a single partial/full warp for the 16/32-label
// refinement cases. Synchronizing that exact active mask makes L_cur/L_prev
// writes visible across lanes without involving inactive lanes. Larger coarse
// searches retain the general block reduction and its block-wide barriers.
__device__ inline int dev_pyramid_min_i(
    int my, int* red, int d, int numDisp)
{
    if (numDisp > 32) {
        return dev_block_min_i(my, red, d, numDisp);
    }

    const unsigned active = __activemask();
    const unsigned mask = __ballot_sync(active, d < numDisp);
    __syncwarp(mask);
    int value = my;
    const int lane = d & 31;
    for (int offset = 16; offset > 0; offset >>= 1) {
        int other = __shfl_down_sync(mask, value, offset);
        if (lane + offset < numDisp) {
            value = dev_min_i(value, other);
        }
    }
    return __shfl_sync(mask, value, 0);
}

__device__ inline unsigned int dev_pyramid_min_u32(
    unsigned int my, int* red, int d, int numDisp)
{
    if (numDisp > 32) {
        return dev_block_min_u32(my, red, d, numDisp);
    }

    const unsigned active = __activemask();
    const unsigned mask = __ballot_sync(active, d < numDisp);
    __syncwarp(mask);
    unsigned int value = my;
    const int lane = d & 31;
    for (int offset = 16; offset > 0; offset >>= 1) {
        unsigned int other = __shfl_down_sync(mask, value, offset);
        if (lane + offset < numDisp) {
            value = value < other ? value : other;
        }
    }
    return __shfl_sync(mask, value, 0);
}

__global__ void kernel_SGM_PathLRPyramid(
    short* cost_acc, const short* C,
    const short* base_disp, const unsigned char* valid_mask,
    const unsigned char* img, int img_stride,
    int width, int height, int numDisp, int minDisp, int maxDisp,
    int P1, int P2,
    int adaptiveP2, float p2Alpha, float p2Beta, float p2LowerMult, float p2UpperMult)
{
    extern __shared__ int smem_i[];
    short* smem = reinterpret_cast<short*>(smem_i);
    short* L_prev = smem_L_prev(smem);
    short* L_cur = smem_L_cur(smem, numDisp);
    int* red_i = smem_i + numDisp;
    int y = blockIdx.x;
    int d = threadIdx.x;
    if (y >= height || d >= numDisp) return;

    bool prevActive = false;
    int prevBase = minDisp;
    short minPrev = SHRT_MAX;
    const int row = y * width;
    const unsigned char* img_row = adaptiveP2 ? img + y * img_stride : nullptr;

    for (int x = 0; x < width; ++x) {
        int p = row + x;
        int base = base_disp[p];
        bool labelValid = dev_pyramid_label_valid(
            C, valid_mask, base_disp, p, x, d,
            numDisp, minDisp, maxDisp, width);
        int validCount = dev_pyramid_valid_label_count(
            valid_mask, base_disp, p, x, numDisp, minDisp, maxDisp, width);
        int out = p * numDisp + d;
        if (validCount == 0) {
            L_cur[d] = SHRT_MAX;
            cost_acc[out] = SHRT_MAX;
            prevActive = false;
            continue;
        }

        int c = labelValid ? (int)C[out] : SHRT_MAX;
        if (!prevActive) {
            L_cur[d] = (short)c;
        } else if (!labelValid) {
            L_cur[d] = SHRT_MAX;
        } else {
            int same = d + base - prevBase;
            int p2eff = adaptiveP2
                ? dev_adaptive_p2(P2, (int)img_row[x], (int)img_row[x - 1],
                                  p2Alpha, p2Beta, p2LowerMult, p2UpperMult)
                : P2;
            int l0 = (same >= 0 && same < numDisp) ? (int)L_prev[same] : SHRT_MAX;
            int minus = same - 1;
            int plus = same + 1;
            int lm = (minus >= 0 && minus < numDisp && L_prev[minus] < SHRT_MAX)
                ? (int)L_prev[minus] + P1 : SHRT_MAX;
            int lp = (plus >= 0 && plus < numDisp && L_prev[plus] < SHRT_MAX)
                ? (int)L_prev[plus] + P1 : SHRT_MAX;
            int lj = dev_min_i(SHRT_MAX, (int)minPrev + p2eff);
            int best = dev_min_i(dev_min_i(l0, lm), dev_min_i(lp, lj));
            L_cur[d] = best < SHRT_MAX
                ? dev_saturate_short(c + best - (int)minPrev) : (short)SHRT_MAX;
        }

        short minCur = (short)dev_pyramid_min_i((int)L_cur[d], red_i, d, numDisp);
        cost_acc[out] = L_cur[d];
        short* tmp = L_prev; L_prev = L_cur; L_cur = tmp;
        minPrev = minCur;
        prevBase = base;
        prevActive = minCur < SHRT_MAX;
    }
}

__global__ void kernel_SGM_PathTBPyramid(
    short* cost_acc, const short* C,
    const short* base_disp, const unsigned char* valid_mask,
    const unsigned char* img, int img_stride,
    int width, int height, int numDisp, int minDisp, int maxDisp,
    int P1, int P2,
    int adaptiveP2, float p2Alpha, float p2Beta, float p2LowerMult, float p2UpperMult)
{
    extern __shared__ int smem_i[];
    short* smem = reinterpret_cast<short*>(smem_i);
    short* L_prev = smem_L_prev(smem);
    short* L_cur = smem_L_cur(smem, numDisp);
    int* red_i = smem_i + numDisp;
    int x = blockIdx.x;
    int d = threadIdx.x;
    if (x >= width || d >= numDisp) return;

    bool prevActive = false;
    int prevBase = minDisp;
    short minPrev = SHRT_MAX;
    for (int y = 0; y < height; ++y) {
        int p = y * width + x;
        int base = base_disp[p];
        bool labelValid = dev_pyramid_label_valid(
            C, valid_mask, base_disp, p, x, d,
            numDisp, minDisp, maxDisp, width);
        int validCount = dev_pyramid_valid_label_count(
            valid_mask, base_disp, p, x, numDisp, minDisp, maxDisp, width);
        int out = p * numDisp + d;
        if (validCount == 0) {
            L_cur[d] = SHRT_MAX;
            cost_acc[out] = SHRT_MAX;
            prevActive = false;
            continue;
        }

        int c = labelValid ? (int)C[out] : SHRT_MAX;
        if (!prevActive) {
            L_cur[d] = (short)c;
        } else if (!labelValid) {
            L_cur[d] = SHRT_MAX;
        } else {
            int same = d + base - prevBase;
            int p2eff = adaptiveP2
                ? dev_adaptive_p2(P2, (int)img[y * img_stride + x],
                                  (int)img[(y - 1) * img_stride + x],
                                  p2Alpha, p2Beta, p2LowerMult, p2UpperMult)
                : P2;
            int l0 = (same >= 0 && same < numDisp) ? (int)L_prev[same] : SHRT_MAX;
            int minus = same - 1;
            int plus = same + 1;
            int lm = (minus >= 0 && minus < numDisp && L_prev[minus] < SHRT_MAX)
                ? (int)L_prev[minus] + P1 : SHRT_MAX;
            int lp = (plus >= 0 && plus < numDisp && L_prev[plus] < SHRT_MAX)
                ? (int)L_prev[plus] + P1 : SHRT_MAX;
            int lj = dev_min_i(SHRT_MAX, (int)minPrev + p2eff);
            int best = dev_min_i(dev_min_i(l0, lm), dev_min_i(lp, lj));
            L_cur[d] = best < SHRT_MAX
                ? dev_saturate_short(c + best - (int)minPrev) : (short)SHRT_MAX;
        }

        short minCur = (short)dev_pyramid_min_i((int)L_cur[d], red_i, d, numDisp);
        int old = cost_acc[out];
        cost_acc[out] = (old < SHRT_MAX && L_cur[d] < SHRT_MAX)
            ? (short)dev_min_i(SHRT_MAX - 1, old + (int)L_cur[d])
            : (short)SHRT_MAX;
        short* tmp = L_prev; L_prev = L_cur; L_cur = tmp;
        minPrev = minCur;
        prevBase = base;
        prevActive = minCur < SHRT_MAX;
    }
}

__global__ void kernel_SGM_PathRLPyramidWTA(
    short* cost_acc, const short* C,
    const short* base_disp, const unsigned char* valid_mask,
    const unsigned char* img, int img_stride,
    short* disp, short* disp2, short* disp2_cost, float* conf_raw,
    int width, int height, int numDisp, int minDisp, int maxDisp,
    int P1, int P2, int uniquenessRatio, float minConfidence,
    int accumulate,
    int adaptiveP2, float p2Alpha, float p2Beta, float p2LowerMult, float p2UpperMult)
{
    extern __shared__ int smem_i[];
    short* smem = reinterpret_cast<short*>(smem_i);
    short* L_prev = smem_L_prev(smem);
    short* L_cur = smem_L_cur(smem, numDisp);
    int* red_i = smem_i + numDisp;
    int y = blockIdx.x;
    int d = threadIdx.x;
    if (y >= height || d >= numDisp) return;

    const short invalidDisp = (short)((minDisp - 1) * 16);
    bool prevActive = false;
    int prevBase = minDisp;
    short minPrev = SHRT_MAX;
    const int row = y * width;
    const unsigned char* img_row = adaptiveP2 ? img + y * img_stride : nullptr;

    for (int x = width - 1; x >= 0; --x) {
        int p = row + x;
        int base = base_disp[p];
        bool labelValid = dev_pyramid_label_valid(
            C, valid_mask, base_disp, p, x, d,
            numDisp, minDisp, maxDisp, width);
        int validCount = dev_pyramid_valid_label_count(
            valid_mask, base_disp, p, x, numDisp, minDisp, maxDisp, width);
        int out = p * numDisp + d;
        if (validCount == 0) {
            L_cur[d] = SHRT_MAX;
            cost_acc[out] = SHRT_MAX;
            if (d == 0) {
                disp[p] = invalidDisp;
                conf_raw[p] = 0.0f;
            }
            prevActive = false;
            continue;
        }

        int c = labelValid ? (int)C[out] : SHRT_MAX;
        if (!prevActive) {
            L_cur[d] = (short)c;
        } else if (!labelValid) {
            L_cur[d] = SHRT_MAX;
        } else {
            int same = d + base - prevBase;
            int p2eff = adaptiveP2
                ? dev_adaptive_p2(P2, (int)img_row[x], (int)img_row[x + 1],
                                  p2Alpha, p2Beta, p2LowerMult, p2UpperMult)
                : P2;
            int l0 = (same >= 0 && same < numDisp) ? (int)L_prev[same] : SHRT_MAX;
            int minus = same - 1;
            int plus = same + 1;
            int lm = (minus >= 0 && minus < numDisp && L_prev[minus] < SHRT_MAX)
                ? (int)L_prev[minus] + P1 : SHRT_MAX;
            int lp = (plus >= 0 && plus < numDisp && L_prev[plus] < SHRT_MAX)
                ? (int)L_prev[plus] + P1 : SHRT_MAX;
            int lj = dev_min_i(SHRT_MAX, (int)minPrev + p2eff);
            int best = dev_min_i(dev_min_i(l0, lm), dev_min_i(lp, lj));
            L_cur[d] = best < SHRT_MAX
                ? dev_saturate_short(c + best - (int)minPrev) : (short)SHRT_MAX;
        }

        short minCur = (short)dev_pyramid_min_i((int)L_cur[d], red_i, d, numDisp);
        int old = accumulate ? (int)cost_acc[out] : 0;
        short myCost = (labelValid && L_cur[d] < SHRT_MAX &&
                        (!accumulate || old < SHRT_MAX))
            ? (short)dev_min_i(SHRT_MAX - 1, old + (int)L_cur[d])
            : (short)SHRT_MAX;
        cost_acc[out] = myCost;

        unsigned int key = (((unsigned int)(unsigned short)myCost) << 16) |
                           (unsigned int)d;
        unsigned int bestKey = dev_pyramid_min_u32(key, red_i, d, numDisp);
        short minCost = (short)(bestKey >> 16);
        int bestD = (int)(bestKey & 0xFFFFu);
        int alt = (labelValid && abs(d - bestD) > 1) ? (int)myCost : SHRT_MAX;
        short altCost = (short)dev_pyramid_min_i(alt, red_i, d, numDisp);

        if (d == 0) {
            bool validResult = validCount >= 3 &&
                               minCost < SHRT_MAX && altCost < SHRT_MAX;
            if (validResult && uniquenessRatio > 0 &&
                (int)altCost * (100 - uniquenessRatio) < (int)minCost * 100) {
                validResult = false;
            }

            const int realBest = base + bestD;
            const int physicalMax = dev_min_i(maxDisp, x);
            const bool artificialLeft = bestD == 0 && base > minDisp;
            const bool artificialRight = bestD == numDisp - 1 &&
                                         base + numDisp - 1 < physicalMax;
            if (artificialLeft || artificialRight) validResult = false;

            float conf = 0.0f;
            if (validResult) {
                float denom = fmaxf((float)minCost, 1.0f);
                conf = fminf(fmaxf(((float)altCost - minCost) / denom, 0.0f), 1.0f);
                if (conf < minConfidence) validResult = false;
            }

            if (validResult) {
                int disp16 = realBest * 16;
                if (bestD > 0 && bestD + 1 < numDisp) {
                    int bestOut = p * numDisp + bestD;
                    short c0 = cost_acc[bestOut - 1];
                    short c1 = cost_acc[bestOut];
                    short c2 = cost_acc[bestOut + 1];
                    if (c0 < SHRT_MAX && c1 < SHRT_MAX && c2 < SHRT_MAX) {
                        int denom2 = dev_max_i((int)c0 + c2 - 2 * (int)c1, 1);
                        disp16 += ((int)(c0 - c2) * 16 + denom2) / (denom2 * 2);
                    }
                }
                disp[p] = (short)disp16;
                conf_raw[p] = conf;

                int x2 = x - realBest;
                if (x2 >= 0 && x2 < width) {
                    int rp = y * width + x2;
                    if (disp2_cost[rp] > minCost) {
                        disp2_cost[rp] = minCost;
                        disp2[rp] = (short)realBest;
                    }
                }
            } else {
                disp[p] = invalidDisp;
                conf_raw[p] = 0.0f;
            }
        }

        short* tmp = L_prev; L_prev = L_cur; L_cur = tmp;
        minPrev = minCur;
        prevBase = base;
        prevActive = minCur < SHRT_MAX;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Host 端封装
// ─────────────────────────────────────────────────────────────────────────────

extern "C" {

struct PyrsgmWorkspace {
    int width = 0;
    int height = 0;
    int disp = 0;
    unsigned char* d_grad1 = nullptr;
    unsigned char* d_grad2 = nullptr;
    uint32_t* d_census1 = nullptr;
    uint32_t* d_census2 = nullptr;
    short* d_cost_sad = nullptr;
    short* d_cost_acc = nullptr;
    short* d_disp_raw = nullptr;
    short* d_disp_med = nullptr;
    short* d_disp2 = nullptr;
    short* d_disp2_cost = nullptr;
    float* d_conf_raw = nullptr;
    short* d_base_disp = nullptr;
    unsigned char* d_valid_mask = nullptr;
};

void* pyrsgm_create_workspace()
{
    return new PyrsgmWorkspace();
}

void pyrsgm_destroy_workspace(void* workspace)
{
    PyrsgmWorkspace* ws = static_cast<PyrsgmWorkspace*>(workspace);
    if (!ws) return;
    if (ws->d_grad1) cudaFree(ws->d_grad1);
    if (ws->d_grad2) cudaFree(ws->d_grad2);
    if (ws->d_census1) cudaFree(ws->d_census1);
    if (ws->d_census2) cudaFree(ws->d_census2);
    if (ws->d_cost_sad) cudaFree(ws->d_cost_sad);
    if (ws->d_cost_acc) cudaFree(ws->d_cost_acc);
    if (ws->d_disp_raw) cudaFree(ws->d_disp_raw);
    if (ws->d_disp_med) cudaFree(ws->d_disp_med);
    if (ws->d_disp2) cudaFree(ws->d_disp2);
    if (ws->d_disp2_cost) cudaFree(ws->d_disp2_cost);
    if (ws->d_conf_raw) cudaFree(ws->d_conf_raw);
    if (ws->d_base_disp) cudaFree(ws->d_base_disp);
    if (ws->d_valid_mask) cudaFree(ws->d_valid_mask);
    delete ws;
}

__global__ void convertDispAndConfKernel(
    const short* disp16,
    const float* conf_raw,
    float* disp32,
    float* conf32,
    int width, int height,
    int disp16_pitch_elems,
    int conf_raw_pitch_elems,
    int disp32_pitch_elems,
    int conf32_pitch_elems,
    short invalid_disp)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    short raw = disp16[y * disp16_pitch_elems + x];
    float d = raw / 16.0f;
    bool valid = raw != invalid_disp && d > 0.0f;
    disp32[y * disp32_pitch_elems + x] = valid ? d : 0.0f;
    conf32[y * conf32_pitch_elems + x] = valid ? conf_raw[y * conf_raw_pitch_elems + x] : 0.0f;
}

static int pyrsgm_ensure_workspace(PyrsgmWorkspace* ws, int width, int height, int D)
{
    if (!ws) return -1;
    if (ws->width == width && ws->height == height && ws->disp == D &&
        ws->d_grad1 && ws->d_grad2 && ws->d_census1 && ws->d_census2 &&
        ws->d_cost_sad && ws->d_cost_acc &&
        ws->d_disp_raw && ws->d_disp_med && ws->d_disp2 && ws->d_disp2_cost && ws->d_conf_raw &&
        ws->d_base_disp && ws->d_valid_mask) {
        return 0;
    }

    if (ws->d_grad1) { cudaFree(ws->d_grad1); ws->d_grad1 = nullptr; }
    if (ws->d_grad2) { cudaFree(ws->d_grad2); ws->d_grad2 = nullptr; }
    if (ws->d_census1) { cudaFree(ws->d_census1); ws->d_census1 = nullptr; }
    if (ws->d_census2) { cudaFree(ws->d_census2); ws->d_census2 = nullptr; }
    if (ws->d_cost_sad) { cudaFree(ws->d_cost_sad); ws->d_cost_sad = nullptr; }
    if (ws->d_cost_acc) { cudaFree(ws->d_cost_acc); ws->d_cost_acc = nullptr; }
    if (ws->d_disp_raw) { cudaFree(ws->d_disp_raw); ws->d_disp_raw = nullptr; }
    if (ws->d_disp_med) { cudaFree(ws->d_disp_med); ws->d_disp_med = nullptr; }
    if (ws->d_disp2) { cudaFree(ws->d_disp2); ws->d_disp2 = nullptr; }
    if (ws->d_disp2_cost) { cudaFree(ws->d_disp2_cost); ws->d_disp2_cost = nullptr; }
    if (ws->d_conf_raw) { cudaFree(ws->d_conf_raw); ws->d_conf_raw = nullptr; }
    if (ws->d_base_disp) { cudaFree(ws->d_base_disp); ws->d_base_disp = nullptr; }
    if (ws->d_valid_mask) { cudaFree(ws->d_valid_mask); ws->d_valid_mask = nullptr; }

    const size_t imgSize = (size_t)width * height * sizeof(unsigned char);
    const size_t censusSize = (size_t)width * height * sizeof(uint32_t);
    const size_t costSize = (size_t)width * height * D * sizeof(short);
    const size_t dispSize = (size_t)width * height * sizeof(short);
    const size_t confSize = (size_t)width * height * sizeof(float);

    cudaError_t err = cudaSuccess;
    #define CUDA_ALLOC_CHECK(call) do { err = (call); if (err != cudaSuccess) return -1; } while (0)
    CUDA_ALLOC_CHECK(cudaMalloc(&ws->d_grad1, imgSize));
    CUDA_ALLOC_CHECK(cudaMalloc(&ws->d_grad2, imgSize));
    CUDA_ALLOC_CHECK(cudaMalloc(&ws->d_census1, censusSize));
    CUDA_ALLOC_CHECK(cudaMalloc(&ws->d_census2, censusSize));
    CUDA_ALLOC_CHECK(cudaMalloc(&ws->d_cost_sad, costSize));
    CUDA_ALLOC_CHECK(cudaMalloc(&ws->d_cost_acc, costSize));
    CUDA_ALLOC_CHECK(cudaMalloc(&ws->d_disp_raw, dispSize));
    CUDA_ALLOC_CHECK(cudaMalloc(&ws->d_disp_med, dispSize));
    CUDA_ALLOC_CHECK(cudaMalloc(&ws->d_disp2, dispSize));
    CUDA_ALLOC_CHECK(cudaMalloc(&ws->d_disp2_cost, dispSize));
    CUDA_ALLOC_CHECK(cudaMalloc(&ws->d_conf_raw, confSize));
    CUDA_ALLOC_CHECK(cudaMalloc(&ws->d_base_disp, dispSize));
    CUDA_ALLOC_CHECK(cudaMalloc(&ws->d_valid_mask, imgSize));
    #undef CUDA_ALLOC_CHECK

    ws->width = width;
    ws->height = height;
    ws->disp = D;
    return 0;
}

int pyrsgm_workspace_reserve(void* workspace, int width, int height, int numDisparities)
{
    if (!workspace || width <= 0 || height <= 0 || numDisparities <= 0 ||
        numDisparities > 1024 || (numDisparities % 16) != 0) {
        return -1;
    }
    return pyrsgm_ensure_workspace(
        static_cast<PyrsgmWorkspace*>(workspace), width, height, numDisparities);
}

int pyrsgm_compute_level(
    void* workspace,
    const unsigned char* d_left_gray,
    const unsigned char* d_right_gray,
    const short* d_parent_disp,
    const float* d_parent_conf,
    short* d_disp,
    float* d_disp_float,
    float* d_conf_float,
    size_t left_step_bytes,
    size_t right_step_bytes,
    size_t parent_disp_step_bytes,
    size_t parent_conf_step_bytes,
    size_t disp_step_bytes,
    size_t disp_float_step_bytes,
    size_t conf_float_step_bytes,
    int width, int height,
    int parent_width, int parent_height,
    int numDisparities, int minDisparity, int maxDisparity,
    int blockSize, int P1, int P2,
    int uniquenessRatio, int disp12MaxDiff,
    int preFilterCap,
    int costMode,
    int postFlags,
    int numPaths,
    int refineLevel,
    float minConfidence,
    int adaptiveP2,
    float p2Alpha, float p2Beta, float p2LowerMult, float p2UpperMult,
    cudaStream_t stream)
{
    if (!workspace || !d_left_gray || !d_right_gray || width <= 0 || height <= 0 ||
        numDisparities <= 0 || numDisparities > 1024 ||
        (numDisparities % 16) != 0 ||
        (refineLevel && numDisparities > 256)) {
        return -1;
    }
    if (refineLevel && (!d_parent_disp || !d_parent_conf ||
                        parent_width <= 0 || parent_height <= 0)) {
        return -1;
    }

    PyrsgmWorkspace* ws = static_cast<PyrsgmWorkspace*>(workspace);
    const int D = numDisparities;
    maxDisparity = std::min(maxDisparity, width - 1);
    maxDisparity = std::min(maxDisparity, SHRT_MAX / 16);
    if (!refineLevel) {
        maxDisparity = std::min(maxDisparity, minDisparity + D - 1);
    }
    if (maxDisparity < minDisparity ||
        pyrsgm_ensure_workspace(ws, width, height, D) != 0) {
        return -1;
    }

    if (p2Alpha < 0.0f) p2Alpha = 0.0f;
    p2Beta = fmaxf(p2Beta, 0.5f);
    p2UpperMult = fmaxf(p2UpperMult, 1.0f);
    p2LowerMult = fminf(fmaxf(p2LowerMult, 0.0f), p2UpperMult);
    if (p2Alpha <= 0.0f && p2LowerMult >= 1.0f) adaptiveP2 = 0;
    minConfidence = fminf(fmaxf(minConfidence, 0.0f), 1.0f);
    if (numPaths < 1) numPaths = 1;
    if (numPaths > 3) numPaths = 3;

    short* d_cost_sad = ws->d_cost_sad;
    short* d_cost_acc = ws->d_cost_acc;
    short* d_disp_raw = ws->d_disp_raw;
    short* d_disp_med = ws->d_disp_med;
    short* d_disp2 = ws->d_disp2;
    short* d_disp2_cost = ws->d_disp2_cost;
    float* d_conf_raw = ws->d_conf_raw;
    short* d_base_disp = ws->d_base_disp;
    unsigned char* d_valid_mask = ws->d_valid_mask;
    uint32_t* d_census1 = ws->d_census1;
    uint32_t* d_census2 = ws->d_census2;
    unsigned char* d_grad1 = ws->d_grad1;
    unsigned char* d_grad2 = ws->d_grad2;

    const size_t path_smem =
        2 * D * sizeof(short) + ((D + 31) / 32) * sizeof(int);
    cudaError_t err;
    #define PYR_CUDA_CHECK(call) do { err = (call); if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA pyramid error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        return -1; } } while (0)

    dim3 block2d(16, 16);
    dim3 grid2d((width + 15) / 16, (height + 15) / 16);
    const short invalidDisp = (short)((minDisparity - 1) * 16);
    kernel_FillDisp<<<grid2d, block2d, 0, stream>>>(
        d_disp_raw, width, height, invalidDisp);
    PYR_CUDA_CHECK(cudaGetLastError());
    kernel_FillDisp<<<grid2d, block2d, 0, stream>>>(
        d_disp2, width, height, invalidDisp);
    PYR_CUDA_CHECK(cudaGetLastError());
    kernel_FillDisp<<<grid2d, block2d, 0, stream>>>(
        d_disp2_cost, width, height, SHRT_MAX);
    PYR_CUDA_CHECK(cudaGetLastError());
    PYR_CUDA_CHECK(cudaMemsetAsync(
        d_conf_raw, 0, (size_t)width * height * sizeof(float), stream));

    if (refineLevel) {
        kernel_PrepareRefineLabels<<<grid2d, block2d, 0, stream>>>(
            d_parent_disp, d_parent_conf,
            parent_width, parent_height,
            (int)(parent_disp_step_bytes / sizeof(short)),
            (int)(parent_conf_step_bytes / sizeof(float)),
            d_base_disp, d_valid_mask,
            width, height, minDisparity, maxDisparity, D, minConfidence);
    } else {
        kernel_InitCoarseLabels<<<grid2d, block2d, 0, stream>>>(
            d_base_disp, d_valid_mask, width, height, minDisparity);
    }
    PYR_CUDA_CHECK(cudaGetLastError());

    if (costMode == PYRSGM_COST_CENSUS) {
        dim3 censusGrid((width + CENSUS_THREADS - 1) / CENSUS_THREADS, height);
        size_t censusSmem = (CENSUS_THREADS + 4) * 5 * sizeof(unsigned char);
        kernel_CensusCompute<<<censusGrid, CENSUS_THREADS, censusSmem, stream>>>(
            d_left_gray, d_census1, width, height, (int)left_step_bytes, width);
        PYR_CUDA_CHECK(cudaGetLastError());
        kernel_CensusCompute<<<censusGrid, CENSUS_THREADS, censusSmem, stream>>>(
            d_right_gray, d_census2, width, height, (int)right_step_bytes, width);
        PYR_CUDA_CHECK(cudaGetLastError());

        if (refineLevel) {
            dim3 localBlock(D, 4);
            dim3 localGrid((width + localBlock.y - 1) / localBlock.y, height);
            kernel_CensusCostLocal<<<localGrid, localBlock, 0, stream>>>(
                d_census1, d_census2, d_base_disp, d_valid_mask, d_cost_sad,
                width, height, width, D, minDisparity, maxDisparity);
        } else {
            dim3 costGrid((width + CENSUS_COST_TILE_X - 1) / CENSUS_COST_TILE_X,
                          height);
            kernel_CensusCost<<<costGrid, D, 0, stream>>>(
                d_census1, d_census2, d_cost_sad,
                width, height, width, D, minDisparity);
        }
        PYR_CUDA_CHECK(cudaGetLastError());
    } else {
        kernel_BTGradPrecompute<<<grid2d, block2d, 0, stream>>>(
            d_left_gray, d_grad1, width, height,
            (int)left_step_bytes, width, preFilterCap);
        PYR_CUDA_CHECK(cudaGetLastError());
        kernel_BTGradPrecompute<<<grid2d, block2d, 0, stream>>>(
            d_right_gray, d_grad2, width, height,
            (int)right_step_bytes, width, preFilterCap);
        PYR_CUDA_CHECK(cudaGetLastError());

        if (refineLevel) {
            dim3 localBlock(D, 4);
            dim3 localGrid((width + localBlock.y - 1) / localBlock.y, height);
            kernel_BTWindowCostLocal<<<localGrid, localBlock, 0, stream>>>(
                d_grad1, d_grad2, d_left_gray, d_right_gray,
                d_base_disp, d_valid_mask, d_cost_sad,
                width, height, width,
                (int)left_step_bytes, (int)right_step_bytes,
                D, minDisparity, maxDisparity, blockSize);
            PYR_CUDA_CHECK(cudaGetLastError());
        } else {
            dim3 btGrid((width + BTCOST_TILE_X - 1) / BTCOST_TILE_X, height);
            size_t btSmem =
                2 * (BTCOST_TILE_X + 2) + 2 * (BTCOST_TILE_X + D + 2);
            kernel_BTCost<<<btGrid, D, btSmem, stream>>>(
                d_grad1, d_grad2, d_left_gray, d_right_gray,
                d_cost_sad, width, height, width,
                (int)left_step_bytes, (int)right_step_bytes,
                D, minDisparity);
            PYR_CUDA_CHECK(cudaGetLastError());
            kernel_BoxFilterH<<<height, D, 0, stream>>>(
                d_cost_sad, d_cost_acc, width, height, D, blockSize);
            PYR_CUDA_CHECK(cudaGetLastError());
            kernel_BoxFilterV<<<width, D, 0, stream>>>(
                d_cost_acc, d_cost_sad, width, height, D, blockSize);
            PYR_CUDA_CHECK(cudaGetLastError());
        }
    }

    if (numPaths >= 2) {
        kernel_SGM_PathLRPyramid<<<height, D, path_smem, stream>>>(
            d_cost_acc, d_cost_sad, d_base_disp, d_valid_mask,
            d_left_gray, (int)left_step_bytes,
            width, height, D, minDisparity, maxDisparity, P1, P2,
            adaptiveP2, p2Alpha, p2Beta, p2LowerMult, p2UpperMult);
        PYR_CUDA_CHECK(cudaGetLastError());
    }
    if (numPaths >= 3) {
        kernel_SGM_PathTBPyramid<<<width, D, path_smem, stream>>>(
            d_cost_acc, d_cost_sad, d_base_disp, d_valid_mask,
            d_left_gray, (int)left_step_bytes,
            width, height, D, minDisparity, maxDisparity, P1, P2,
            adaptiveP2, p2Alpha, p2Beta, p2LowerMult, p2UpperMult);
        PYR_CUDA_CHECK(cudaGetLastError());
    }
    kernel_SGM_PathRLPyramidWTA<<<height, D, path_smem, stream>>>(
        d_cost_acc, d_cost_sad, d_base_disp, d_valid_mask,
        d_left_gray, (int)left_step_bytes,
        d_disp_raw, d_disp2, d_disp2_cost, d_conf_raw,
        width, height, D, minDisparity, maxDisparity,
        P1, P2, uniquenessRatio, minConfidence,
        numPaths >= 2 ? 1 : 0,
        adaptiveP2, p2Alpha, p2Beta, p2LowerMult, p2UpperMult);
    PYR_CUDA_CHECK(cudaGetLastError());

    if (postFlags & PYRSGM_POST_LRCHECK) {
        kernel_StrictPyramidLRCheck<<<grid2d, block2d, 0, stream>>>(
            d_disp_raw, d_disp2, d_conf_raw,
            width, height, minDisparity, disp12MaxDiff);
        PYR_CUDA_CHECK(cudaGetLastError());
    }

    short* d_final = d_disp_raw;
    if (postFlags & PYRSGM_POST_MEDIAN) {
        kernel_MedianFilter3x3<<<grid2d, block2d, 0, stream>>>(
            d_disp_raw, d_disp_med, width, height);
        PYR_CUDA_CHECK(cudaGetLastError());
        kernel_ClearChangedMedianConfidence<<<grid2d, block2d, 0, stream>>>(
            d_disp_raw, d_disp_med, d_conf_raw, width, height);
        PYR_CUDA_CHECK(cudaGetLastError());
        d_final = d_disp_med;
    }
    if (d_disp) {
        PYR_CUDA_CHECK(cudaMemcpy2DAsync(
            d_disp, disp_step_bytes,
            d_final, (size_t)width * sizeof(short),
            (size_t)width * sizeof(short), height,
            cudaMemcpyDeviceToDevice, stream));
    }
    if (d_disp_float && d_conf_float) {
        convertDispAndConfKernel<<<grid2d, block2d, 0, stream>>>(
            d_final, d_conf_raw, d_disp_float, d_conf_float,
            width, height, width, width,
            (int)(disp_float_step_bytes / sizeof(float)),
            (int)(conf_float_step_bytes / sizeof(float)),
            invalidDisp);
        PYR_CUDA_CHECK(cudaGetLastError());
    } else if (d_conf_float) {
        PYR_CUDA_CHECK(cudaMemcpy2DAsync(
            d_conf_float, conf_float_step_bytes,
            d_conf_raw, (size_t)width * sizeof(float),
            (size_t)width * sizeof(float), height,
            cudaMemcpyDeviceToDevice, stream));
    }

    #undef PYR_CUDA_CHECK
    return 0;
}

int pyrsgm_compute_single_scale(
    void* workspace,
    const unsigned char* d_left_gray,
    const unsigned char* d_right_gray,
    short* d_disp,
    float* d_disp_float,
    float* d_conf_float,
    size_t left_step_bytes,
    size_t right_step_bytes,
    size_t disp_step_bytes,
    size_t disp_float_step_bytes,
    size_t conf_float_step_bytes,
    int width, int height,
    int numDisparities, int minDisparity,
    int blockSize, int P1, int P2,
    int uniquenessRatio, int disp12MaxDiff,
    int preFilterCap,
    int costMode,
    int postFlags,
    int numPaths,
    int adaptiveP2,
    float p2Alpha, float p2Beta, float p2LowerMult, float p2UpperMult,
    cudaStream_t stream)
{
    int D = numDisparities;
    // 自适应 P2 参数消毒：beta 下限防除零；upper_mult>=1 保证平坦区可≥基础 P2；
    // lower_mult∈[0,upper]；仅当 alpha<=0 且 lower_mult>=1 时退化为常数 P2。
    if (p2Alpha < 0.0f) p2Alpha = 0.0f;
    p2Beta = fmaxf(p2Beta, 0.5f);
    p2UpperMult = fmaxf(p2UpperMult, 1.0f);
    p2LowerMult = fminf(fmaxf(p2LowerMult, 0.0f), p2UpperMult);
    if (p2Alpha <= 0.0f && p2LowerMult >= 1.0f) adaptiveP2 = 0;
    int minX1 = std::max(minDisparity + numDisparities, 0);
    int maxX1 = width + std::min(minDisparity, 0);
    // numPaths: 3=完整3路径（默认，与原实现一致）；2=跳过竖直路径 PathTB，
    // 仅保留双向水平聚合（PathLR + PathRL_WTA）；1=只保留 PathRL_WTA 单路径。
    // 用于给 LT（左上）这类"辅助/补洞"视差源降低计算量，LR 主路径应始终传 3。
    if (numPaths < 1) numPaths = 1;
    if (numPaths > 3) numPaths = 3;
    PyrsgmWorkspace* ws = static_cast<PyrsgmWorkspace*>(workspace);
    if (pyrsgm_ensure_workspace(ws, width, height, D) != 0) {
        fprintf(stderr, "CUDA workspace allocation failed\n");
        return -1;
    }

    unsigned char* d_grad1 = ws->d_grad1;
    unsigned char* d_grad2 = ws->d_grad2;
    uint32_t* d_census1 = ws->d_census1;
    uint32_t* d_census2 = ws->d_census2;
    short* d_cost_sad = ws->d_cost_sad;
    short* d_cost_acc = ws->d_cost_acc;
    short* d_disp_raw = ws->d_disp_raw;
    short* d_disp_med = ws->d_disp_med;
    short* d_disp2 = ws->d_disp2;
    short* d_disp2_cost = ws->d_disp2_cost;
    float* d_conf_raw = ws->d_conf_raw;

    // 路径核共享内存布局：L_prev + L_cur（2*D short）+ warp shuffle 归约的
    // 部分最小值缓冲（ceil(D/32) 个 int，三次归约串行复用同一块）。
    // 必须在第一个 CUDA_CHECK（可能 goto cleanup）之前声明。
    const size_t path_smem = 2 * D * sizeof(short) + ((D + 31) / 32) * sizeof(int);

    cudaError_t err;
    #define CUDA_CHECK(call) do { err = (call); if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        goto cleanup; } } while(0)

    {
        dim3 block_2d(16, 16);
        dim3 grid_2d((width  + 15) / 16, (height + 15) / 16);
        const short invalid_disp = (short)((minDisparity - 1) * 16);

        kernel_FillDisp<<<grid_2d, block_2d, 0, stream>>>(d_disp_raw, width, height, invalid_disp);
        CUDA_CHECK(cudaGetLastError());

        kernel_FillDisp<<<grid_2d, block_2d, 0, stream>>>(d_disp2, width, height, invalid_disp);
        CUDA_CHECK(cudaGetLastError());

        kernel_FillDisp<<<grid_2d, block_2d, 0, stream>>>(d_disp2_cost, width, height, SHRT_MAX);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemsetAsync(d_conf_raw, 0, (size_t)width * height * sizeof(float), stream));

        if (costMode == PYRSGM_COST_CENSUS) {
            // Step 1-2: 5x5 Census + Hamming 代价（纯 Census，无 AD/CombinedCost）
            dim3 census_grid((width + CENSUS_THREADS - 1) / CENSUS_THREADS, height);
            const size_t census_smem = (CENSUS_THREADS + 2 * 2) * 5 * sizeof(unsigned char);
            kernel_CensusCompute<<<census_grid, CENSUS_THREADS, census_smem, stream>>>(
                d_left_gray, d_census1, width, height,
                (int)left_step_bytes, width);
            CUDA_CHECK(cudaGetLastError());
            kernel_CensusCompute<<<census_grid, CENSUS_THREADS, census_smem, stream>>>(
                d_right_gray, d_census2, width, height,
                (int)right_step_bytes, width);
            CUDA_CHECK(cudaGetLastError());

            dim3 census_cost_grid((width + CENSUS_COST_TILE_X - 1) / CENSUS_COST_TILE_X, height);
            kernel_CensusCost<<<census_cost_grid, D, 0, stream>>>(
                d_census1, d_census2, d_cost_sad,
                width, height, width, D, minDisparity);
            CUDA_CHECK(cudaGetLastError());
        } else {
            // Step 1: BT 梯度采样强度预计算
            kernel_BTGradPrecompute<<<grid_2d, block_2d, 0, stream>>>(
                d_left_gray, d_grad1, width, height, (int)left_step_bytes, width, preFilterCap);
            CUDA_CHECK(cudaGetLastError());

            kernel_BTGradPrecompute<<<grid_2d, block_2d, 0, stream>>>(
                d_right_gray, d_grad2, width, height, (int)right_step_bytes, width, preFilterCap);
            CUDA_CHECK(cudaGetLastError());

            // Step 2: BT 像素匹配代价计算
            // 每 block 处理 BTCOST_TILE_X 个连续 x、D 线程/block；
            // 左右图 tile 缓存进共享内存，输入像素全局内存只读一次（见核函数注释）。
            {
                dim3 bt_grid((width + BTCOST_TILE_X - 1) / BTCOST_TILE_X, height);
                size_t bt_smem = 2 * (BTCOST_TILE_X + 2) + 2 * (BTCOST_TILE_X + D + 2);
                kernel_BTCost<<<bt_grid, D, bt_smem, stream>>>(
                    d_grad1, d_grad2, d_left_gray, d_right_gray,
                    d_cost_sad, width, height, width, (int)left_step_bytes, (int)right_step_bytes, D, minDisparity);
                CUDA_CHECK(cudaGetLastError());
            }

            // Step 3（仅 BT）: SAD box filter 求和（水平 + 垂直，不归一化）。
            // Census 的 5x5 描述子自身已编码局部邻域，Hamming 代价直接送入 SGM。
            kernel_BoxFilterH<<<height, D, 0, stream>>>(
                d_cost_sad, d_cost_acc, width, height, D, blockSize);
            CUDA_CHECK(cudaGetLastError());

            kernel_BoxFilterV<<<width, D, 0, stream>>>(
                d_cost_acc, d_cost_sad, width, height, D, blockSize);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    // d_cost_sad 现在包含 BT-SAD 或 Census-Hamming 代价（只读，后续不再修改）
    // d_cost_acc 用作路径累计缓冲区

    // Step 4: 路径 1 左→右（读取 d_cost_sad，写入 d_cost_acc）——numPaths>=2 才跑
    if (numPaths >= 2) {
        kernel_SGM_PathLR<<<height, D, path_smem, stream>>>(
            d_cost_acc, d_cost_sad, d_left_gray, (int)left_step_bytes,
            width, height, D, minX1, maxX1, P1, P2,
            adaptiveP2, p2Alpha, p2Beta, p2LowerMult, p2UpperMult);
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 5: 路径 2 上→下（读取 d_cost_sad，累加到 d_cost_acc）——numPaths>=3 才跑
    if (numPaths >= 3) {
        kernel_SGM_PathTB<<<width, D, path_smem, stream>>>(
            d_cost_acc, d_cost_sad, d_left_gray, (int)left_step_bytes,
            width, height, D, P1, P2,
            adaptiveP2, p2Alpha, p2Beta, p2LowerMult, p2UpperMult);
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 6: 路径 3 右→左 + WTA（读取 d_cost_sad，累加/初始化 d_cost_acc，然后 WTA）
    // accumulate=0（numPaths==1，前面的路径都被跳过）时改为直接赋值而非累加。
    {
        int accumulate = (numPaths >= 2) ? 1 : 0;
        kernel_SGM_PathRL_WTA<<<height, D, path_smem, stream>>>(
            d_cost_acc, d_cost_sad, d_left_gray, (int)left_step_bytes,
            d_disp_raw, d_disp2, d_disp2_cost, d_conf_raw, width, height, D, minDisparity,
            P1, P2, uniquenessRatio, disp12MaxDiff, accumulate,
            adaptiveP2, p2Alpha, p2Beta, p2LowerMult, p2UpperMult);
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 7: 可配置后处理。LRCheck 可单独保留，median 可用于复现原 3SGM 行为。
    {
        dim3 block_2d(16, 16);
        dim3 grid_2d((width + 15) / 16, (height + 15) / 16);

        if (postFlags & PYRSGM_POST_LRCHECK) {
            kernel_PseudoLRCheck<<<grid_2d, block_2d, 0, stream>>>(
                d_disp_raw, d_disp2, d_conf_raw, width, height, minDisparity, disp12MaxDiff);
            CUDA_CHECK(cudaGetLastError());
        }

        short* d_disp_final = d_disp_raw;
        if (postFlags & PYRSGM_POST_MEDIAN) {
            kernel_MedianFilter3x3<<<grid_2d, block_2d, 0, stream>>>(
                d_disp_raw, d_disp_med, width, height);
            CUDA_CHECK(cudaGetLastError());
            kernel_ClearChangedMedianConfidence<<<grid_2d, block_2d, 0, stream>>>(
                d_disp_raw, d_disp_med, d_conf_raw, width, height);
            CUDA_CHECK(cudaGetLastError());
            d_disp_final = d_disp_med;
        }

        if (d_disp) {
            CUDA_CHECK(cudaMemcpy2DAsync(
                d_disp, disp_step_bytes,
                d_disp_final, (size_t)width * sizeof(short),
                (size_t)width * sizeof(short), height,
                cudaMemcpyDeviceToDevice, stream));
        }
        if (d_disp_float && d_conf_float) {
            convertDispAndConfKernel<<<grid_2d, block_2d, 0, stream>>>(
                d_disp_final, d_conf_raw, d_disp_float, d_conf_float,
                width, height,
                width, width,
                (int)(disp_float_step_bytes / sizeof(float)),
                (int)(conf_float_step_bytes / sizeof(float)),
                (short)((minDisparity - 1) * 16));
            CUDA_CHECK(cudaGetLastError());
        }
    }

    return 0;

cleanup:
    return -1;
}

} // extern "C"