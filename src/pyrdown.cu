/**
 * @file pyrdown.cu
 * @brief Gaussian pyramid down-sampling (cv::pyrDown compatible, 8u C1).
 *
 * Kernel: [1 4 6 4 1]^T x [1 4 6 4 1] / 256, output taken at even source
 * coordinates, borders handled with BORDER_REFLECT_101 (gfedcb|abcdefgh|gfedcba),
 * rounding (sum + 128) >> 8 as in OpenCV's integer path.
 */

#include <cuda_runtime.h>

#include "pyrsgm/pyrsgm_c.h"

namespace {

__device__ __forceinline__ int reflect101(int i, int n)
{
    // Valid for n >= 2 and |i| < 2n, which holds for a 5-tap kernel at 2x.
    if (i < 0) i = -i;
    if (i >= n) i = 2 * n - 2 - i;
    return i < 0 ? 0 : i;
}

__global__ void pyrDownKernel(
    const unsigned char* __restrict__ src, int src_step,
    int src_width, int src_height,
    unsigned char* __restrict__ dst, int dst_step,
    int dst_width, int dst_height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_width || y >= dst_height) return;

    const int cx = 2 * x;
    const int cy = 2 * y;
    const int w[5] = {1, 4, 6, 4, 1};

    int sum = 0;
#pragma unroll
    for (int ky = 0; ky < 5; ++ky) {
        const int sy = reflect101(cy + ky - 2, src_height);
        const unsigned char* row = src + sy * src_step;
        int row_sum = 0;
#pragma unroll
        for (int kx = 0; kx < 5; ++kx) {
            const int sx = reflect101(cx + kx - 2, src_width);
            row_sum += w[kx] * (int)row[sx];
        }
        sum += w[ky] * row_sum;
    }
    dst[y * dst_step + x] = (unsigned char)((sum + 128) >> 8);
}

} // namespace

extern "C" int pyrsgm_pyrdown_u8(
    const unsigned char* d_src, size_t src_step_bytes,
    int src_width, int src_height,
    unsigned char* d_dst, size_t dst_step_bytes,
    int dst_width, int dst_height,
    cudaStream_t stream)
{
    if (!d_src || !d_dst || src_width < 2 || src_height < 2 ||
        dst_width != (src_width + 1) / 2 || dst_height != (src_height + 1) / 2) {
        return -1;
    }
    dim3 block(32, 8);
    dim3 grid((dst_width + block.x - 1) / block.x,
              (dst_height + block.y - 1) / block.y);
    pyrDownKernel<<<grid, block, 0, stream>>>(
        d_src, (int)src_step_bytes, src_width, src_height,
        d_dst, (int)dst_step_bytes, dst_width, dst_height);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
