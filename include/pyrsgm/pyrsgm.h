/**
 * @file pyrsgm.h
 * @brief pyrsgm — CUDA coarse-to-fine (pyramid) Semi-Global Matching.
 *
 * Header-only dependency surface: CUDA runtime.  No OpenCV required.
 *
 * Usage:
 * @code
 *   pyrsgm::Params p;
 *   p.num_disparities = 128;        // full-resolution search range
 *   p.pyramid_levels  = 2;          // coarse search at 1/4, refine x2 with 16 labels
 *   pyrsgm::StereoMatcher sgm(width, height, p);
 *   sgm.compute(d_left, left_pitch, d_right, right_pitch,
 *               d_disp16, disp_pitch, nullptr, 0, nullptr, 0, stream);
 * @endcode
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include <cuda_runtime.h>

namespace pyrsgm {

enum class CostMode {
    BirchfieldTomasi = 0,  ///< BT sampling-insensitive cost + gradient, SAD block aggregation
    Census = 1             ///< 5x5 Census transform + Hamming distance
};

struct Params {
    // ---- matching ---------------------------------------------------------
    /// Disparity search range at full resolution.  Rounded up to a multiple
    /// of 16; <= 1024.  With pyramid_levels > 0 this is still the range that
    /// is covered (pyramid_max_disparity defaults to num_disparities - 1).
    int num_disparities = 128;
    /// Minimum disparity.  Pyramid mode currently requires 0.
    int min_disparity = 0;
    /// SAD window for the BT cost (odd, >= 3).  Ignored by Census.
    /// Pyramid levels >= 2 use a 3x3 window regardless.
    int block_size = 5;
    CostMode cost_mode = CostMode::BirchfieldTomasi;
    /// Smoothness penalties.  BT costs are block_size^2 SAD sums of values
    /// in [0,255]; Census costs are Hamming distances in [0,24].
    int p1 = 300;
    int p2 = 2400;
    /// Uniqueness margin in percent (0 disables).  Same meaning as OpenCV.
    int uniqueness_ratio = 10;
    /// Max allowed LR difference in pixels (pyramid requires >= 0).
    int disp12_max_diff = 1;
    bool lr_check = true;
    /// 3x3 median on the final level.
    bool median_filter = true;
    /// Aggregation paths: 3 = L->R, T->B, R->L; 2 drops T->B; 1 keeps R->L.
    int num_paths = 3;
    /// BT gradient clipping (OpenCV preFilterCap).
    int pre_filter_cap = 63;

    // ---- coarse-to-fine pyramid -----------------------------------------
    /// 0 = legacy full-resolution single-scale SGM.
    /// L in 1..4 runs the full search at 1/2^L resolution, then refines each
    /// finer level with 16 local labels around the up-scaled parent disparity.
    int pyramid_levels = 0;
    /// Upper bound of labels evaluated at the coarsest level.  The effective
    /// value is align16(min(pyramid_coarse_disparities, level_max + 1)).
    int pyramid_coarse_disparities = 128;
    /// Largest real disparity (full resolution) the pyramid may produce.
    /// <= 0 selects num_disparities - 1.
    int pyramid_max_disparity = 0;
    /// Parent pixels below this confidence are not propagated to finer levels.
    float pyramid_min_confidence = 0.05f;

    // ---- adaptive P2 ------------------------------------------------------
    /// P2' = P2 * (lower + (1 + alpha - lower) * exp(-|I(p) - I(p-r)| / beta)),
    /// clamped to [max(1, P2*lower), P2*upper].
    bool adaptive_p2 = false;
    float p2_alpha = 3.041f;
    float p2_beta = 16.137f;
    float p2_lower_mult = 0.601f;
    float p2_upper_mult = 1.77f;
};

/// Resolve derived values (alignment, pyramid_max_disparity) and throw
/// std::invalid_argument for unsupported combinations.
Params validateParams(const Params& params);

/// Fixed-point invalid marker of the 16-bit disparity output.
inline int16_t invalidDisparity16(const Params& params)
{
    return static_cast<int16_t>((params.min_disparity - 1) * 16);
}

/// Device buffer sizes per level, for diagnostics.
struct LevelInfo {
    int level = 0;
    int width = 0;
    int height = 0;
    int num_labels = 0;      ///< labels evaluated per pixel at this level
    int max_disparity = 0;   ///< real disparity cap at this level's scale
    int block_size = 0;
    bool refine = false;
};

/**
 * Stereo matcher for a fixed image size.  All device memory (cost volumes,
 * pyramid images, intermediate disparities) is allocated once in the
 * constructor; compute() is allocation-free and fully asynchronous.
 *
 * Inputs are 8-bit grayscale **rectified** left/right images in device memory.
 * Outputs (all optional, pass nullptr to skip):
 *   - disp16 : int16, disparity * 16, invalid = invalidDisparity16()
 *   - disp32 : float, disparity in pixels, invalid = 0
 *   - conf   : float in [0,1], 0 for invalid pixels
 */
class StereoMatcher {
public:
    StereoMatcher(int width, int height, const Params& params);
    ~StereoMatcher();
    StereoMatcher(StereoMatcher&&) noexcept;
    StereoMatcher& operator=(StereoMatcher&&) noexcept;
    StereoMatcher(const StereoMatcher&) = delete;
    StereoMatcher& operator=(const StereoMatcher&) = delete;

    /// Enqueue one stereo computation on `stream`.  Throws std::runtime_error
    /// on launch failure.  Pitches are in bytes.
    void compute(const uint8_t* d_left, size_t left_pitch,
                 const uint8_t* d_right, size_t right_pitch,
                 int16_t* d_disp16, size_t disp16_pitch,
                 float* d_disp32, size_t disp32_pitch,
                 float* d_conf, size_t conf_pitch,
                 cudaStream_t stream = nullptr);

    int width() const;
    int height() const;
    const Params& params() const;       ///< validated parameters
    int numLevels() const;              ///< pyramid_levels + 1 (1 for legacy)
    LevelInfo levelInfo(int level) const;
    size_t deviceMemoryBytes() const;   ///< total device memory owned

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

const char* version();

} // namespace pyrsgm
