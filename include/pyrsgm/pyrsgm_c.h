/**
 * @file pyrsgm_c.h
 * @brief Low-level C API of pyrsgm (CUDA pyramid Semi-Global Matching).
 *
 * All image pointers are **device** pointers.  Pitches are given in bytes.
 * Every function enqueues work on `stream` and returns without host
 * synchronisation; the return value only reports launch/argument errors
 * (0 = success, -1 = failure).  Use cudaGetLastError()/cudaStreamSynchronize()
 * to observe asynchronous errors.
 *
 * The 16-bit disparity output uses OpenCV StereoSGBM fixed-point semantics:
 * value = disparity * 16, invalid = (minDisparity - 1) * 16.
 */
#ifndef PYRSGM_C_H
#define PYRSGM_C_H

#include <stddef.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Matching cost selection (costMode). */
#define PYRSGM_COST_BT       0   /* Birchfield-Tomasi + gradient, SAD window   */
#define PYRSGM_COST_CENSUS   1   /* 5x5 Census + Hamming                        */

/* Post-processing bit flags (postFlags). */
#define PYRSGM_POST_LRCHECK  1   /* pseudo left-right consistency check          */
#define PYRSGM_POST_MEDIAN   2   /* 3x3 median on the final disparity            */

/**
 * Persistent device workspace.  One workspace holds the cost volumes and
 * scratch images for a single (width, height, numDisparities) triple and is
 * re-allocated lazily when those change.  Use one workspace per pyramid
 * level and per concurrently running stream.
 */
void* pyrsgm_create_workspace(void);
void  pyrsgm_destroy_workspace(void* workspace);

/**
 * Allocate (or re-allocate) the workspace for the given geometry up front so
 * that the compute functions never allocate.  Optional; compute functions
 * call it implicitly.  Returns 0 on success.
 */
int pyrsgm_workspace_reserve(void* workspace, int width, int height, int numDisparities);

/**
 * Full-resolution, single-scale 3-path SGM (legacy path; identical to the
 * Tri3SGM `pyramid_levels=0` behaviour).
 *
 * @param d_disp        optional 16-bit output (may be NULL)
 * @param d_disp_float  optional float output in pixels, invalid = 0
 *                      (written only when d_conf_float is also non-NULL)
 * @param d_conf_float  optional confidence output in [0,1]
 * @param numDisparities multiple of 16, <= 1024
 * @param numPaths      1..3 (3 = left->right, top->bottom, right->left)
 * @param adaptiveP2    non-zero enables P2' = P2 * (lower + (1+alpha-lower) *
 *                      exp(-|dI|/beta)) clamped to [P2*lower, P2*upper]
 */
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
    cudaStream_t stream);

/**
 * One pyramid level.
 *
 * refineLevel == 0: coarse search over [minDisparity, minDisparity+numDisparities)
 *                   (numDisparities multiple of 16, <= 1024), limited by
 *                   maxDisparity.  Parent inputs are ignored.
 * refineLevel != 0: refinement around the parent level (half resolution).
 *                   numDisparities must be 16 (<= 256 accepted) and is the
 *                   number of contiguous local labels evaluated per pixel.
 *                   Parent pixels that failed uniqueness / LR check or whose
 *                   confidence is below minConfidence stay invalid.
 *
 * Intermediate levels should pass postFlags = PYRSGM_POST_LRCHECK only.
 */
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
    cudaStream_t stream);

/**
 * Gaussian 5x5 ([1 4 6 4 1]/16 separable) down-sampling by two with
 * BORDER_REFLECT_101, matching cv::pyrDown for 8-bit single-channel images.
 * dst size must be ((src_width + 1) / 2, (src_height + 1) / 2).
 */
int pyrsgm_pyrdown_u8(
    const unsigned char* d_src, size_t src_step_bytes,
    int src_width, int src_height,
    unsigned char* d_dst, size_t dst_step_bytes,
    int dst_width, int dst_height,
    cudaStream_t stream);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* PYRSGM_C_H */
