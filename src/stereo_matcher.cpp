/**
 * @file stereo_matcher.cpp
 * @brief Host-side orchestration of the pyramid: Gaussian down-sampling,
 *        coarse full-range search, 16-label refinement per finer level.
 *
 * Ported from Tri3SgmEngine::runPyramid (Tri3SGM) without the OpenCV
 * dependency; cv::cuda::pyrDown is replaced by pyrsgm_pyrdown_u8.
 */

#include "pyrsgm/pyrsgm.h"

#include <algorithm>
#include <array>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "pyrsgm/pyrsgm_c.h"

namespace pyrsgm {
namespace {

constexpr int kMaxPyramidLevels = 4;
constexpr int kRefineLabels = 16;
constexpr int kMaxCoarseLabels = 1024;

int align16(int v)
{
    return std::max(16, ((v + 15) / 16) * 16);
}

[[noreturn]] void throwCuda(const char* what, cudaError_t err)
{
    std::ostringstream oss;
    oss << "pyrsgm: " << what << ": " << cudaGetErrorString(err);
    throw std::runtime_error(oss.str());
}

void check(cudaError_t err, const char* what)
{
    if (err != cudaSuccess) throwCuda(what, err);
}

/// Pitched 2-D device buffer.
template <class T>
struct DeviceImage {
    T* data = nullptr;
    size_t pitch = 0;  // bytes
    int width = 0;
    int height = 0;

    DeviceImage() = default;
    DeviceImage(const DeviceImage&) = delete;
    DeviceImage& operator=(const DeviceImage&) = delete;
    DeviceImage(DeviceImage&& o) noexcept { *this = std::move(o); }
    DeviceImage& operator=(DeviceImage&& o) noexcept
    {
        if (this != &o) {
            release();
            data = o.data; pitch = o.pitch; width = o.width; height = o.height;
            o.data = nullptr; o.pitch = 0; o.width = o.height = 0;
        }
        return *this;
    }
    ~DeviceImage() { release(); }

    void create(int w, int h)
    {
        release();
        if (w <= 0 || h <= 0) return;
        void* p = nullptr;
        check(cudaMallocPitch(&p, &pitch, (size_t)w * sizeof(T), (size_t)h),
              "cudaMallocPitch");
        data = static_cast<T*>(p);
        width = w;
        height = h;
    }
    void release()
    {
        if (data) cudaFree(data);
        data = nullptr; pitch = 0; width = height = 0;
    }
    bool empty() const { return data == nullptr; }
    size_t bytes() const { return pitch * (size_t)height; }
};

size_t workspaceBytes(int w, int h, int D)
{
    const size_t px = (size_t)w * h;
    return px * 1 * 3                    // grad1, grad2, valid_mask
         + px * 4 * 2                    // census1, census2
         + px * (size_t)D * 2 * 2        // cost_sad, cost_acc
         + px * 2 * 5                    // disp_raw, disp_med, disp2, disp2_cost, base_disp
         + px * 4;                       // conf_raw
}

} // namespace

Params validateParams(const Params& in)
{
    Params p = in;
    if (p.num_disparities <= 0 || p.num_disparities > kMaxCoarseLabels) {
        throw std::invalid_argument("pyrsgm: num_disparities must be in (0, 1024]");
    }
    p.num_disparities = align16(p.num_disparities);
    if (p.num_disparities > kMaxCoarseLabels) {
        throw std::invalid_argument("pyrsgm: num_disparities must be <= 1024 after 16-alignment");
    }
    if (p.block_size < 1) throw std::invalid_argument("pyrsgm: block_size must be >= 1");
    p.block_size = std::max(3, p.block_size | 1);
    if (p.p1 < 0 || p.p2 < p.p1) {
        throw std::invalid_argument("pyrsgm: require 0 <= p1 <= p2");
    }
    if (p.uniqueness_ratio < 0 || p.uniqueness_ratio >= 100) {
        throw std::invalid_argument("pyrsgm: uniqueness_ratio must be in [0, 100)");
    }
    if (p.pre_filter_cap <= 0) throw std::invalid_argument("pyrsgm: pre_filter_cap must be > 0");
    p.num_paths = std::clamp(p.num_paths, 1, 3);

    if (p.pyramid_levels < 0 || p.pyramid_levels > kMaxPyramidLevels) {
        throw std::invalid_argument("pyrsgm: pyramid_levels must be in [0, 4]");
    }
    if (p.pyramid_levels > 0) {
        if (p.min_disparity != 0) {
            throw std::invalid_argument("pyrsgm: pyramid mode requires min_disparity = 0");
        }
        if (p.disp12_max_diff < 0) {
            throw std::invalid_argument("pyrsgm: pyramid mode requires disp12_max_diff >= 0");
        }
        if (p.pyramid_coarse_disparities <= 0) {
            throw std::invalid_argument("pyrsgm: pyramid_coarse_disparities must be > 0");
        }
        p.pyramid_coarse_disparities =
            std::min(kMaxCoarseLabels, align16(p.pyramid_coarse_disparities));
        if (p.pyramid_max_disparity <= 0) {
            p.pyramid_max_disparity = p.min_disparity + p.num_disparities - 1;
        }
        if (p.pyramid_max_disparity < p.min_disparity) {
            throw std::invalid_argument("pyrsgm: pyramid_max_disparity < min_disparity");
        }
        p.pyramid_min_confidence = std::clamp(p.pyramid_min_confidence, 0.0f, 1.0f);
    }
    if (p.adaptive_p2 && p.p2_beta <= 0.0f) {
        throw std::invalid_argument("pyrsgm: adaptive_p2 requires p2_beta > 0");
    }
    return p;
}

struct StereoMatcher::Impl {
    int width = 0;
    int height = 0;
    Params params;
    int levels = 0;  // pyramid_levels

    std::array<void*, kMaxPyramidLevels + 1> workspaces{};
    std::array<DeviceImage<uint8_t>, kMaxPyramidLevels + 1> pyr_left;   // [0] unused (input)
    std::array<DeviceImage<uint8_t>, kMaxPyramidLevels + 1> pyr_right;
    std::array<DeviceImage<int16_t>, kMaxPyramidLevels + 1> disp16;     // [0] unused (output)
    std::array<DeviceImage<float>, kMaxPyramidLevels + 1> conf;         // [0] scratch when needed
    std::array<LevelInfo, kMaxPyramidLevels + 1> info{};
    size_t workspace_bytes = 0;

    Impl(int w, int h, const Params& p)
        : width(w), height(h), params(validateParams(p)), levels(params.pyramid_levels)
    {
        if (w <= 0 || h <= 0) throw std::invalid_argument("pyrsgm: empty image size");

        // Level geometry (mirrors Tri3SgmEngine::runPyramid).
        int lw = w, lh = h;
        for (int level = 0; level <= levels; ++level) {
            if (level > 0) {
                lw = (lw + 1) / 2;
                lh = (lh + 1) / 2;
            }
            LevelInfo& li = info[level];
            li.level = level;
            li.width = lw;
            li.height = lh;
            if (levels == 0) {
                li.refine = false;
                li.num_labels = params.num_disparities;
                li.max_disparity = params.min_disparity + params.num_disparities - 1;
                li.block_size = params.block_size;
            } else {
                li.refine = level != levels;
                li.max_disparity = std::min(params.pyramid_max_disparity >> level, lw - 1);
                li.num_labels = li.refine
                    ? kRefineLabels
                    : align16(std::min(params.pyramid_coarse_disparities, li.max_disparity + 1));
                li.block_size = level >= 2 ? 3 : params.block_size;
            }
        }

        try {
            for (int level = 0; level <= levels; ++level) {
                workspaces[level] = pyrsgm_create_workspace();
                if (!workspaces[level]) throw std::runtime_error("pyrsgm: workspace allocation failed");
                const LevelInfo& li = info[level];
                if (pyrsgm_workspace_reserve(workspaces[level], li.width, li.height, li.num_labels) != 0) {
                    throwCuda("workspace reserve", cudaGetLastError());
                }
                workspace_bytes += workspaceBytes(li.width, li.height, li.num_labels);
                if (level > 0) {
                    pyr_left[level].create(li.width, li.height);
                    pyr_right[level].create(li.width, li.height);
                    disp16[level].create(li.width, li.height);
                    conf[level].create(li.width, li.height);
                }
            }
            // Scratch confidence for level 0 when the caller wants disp32 but no conf.
            conf[0].create(w, h);
        } catch (...) {
            destroy();
            throw;
        }
    }

    ~Impl() { destroy(); }

    void destroy()
    {
        for (auto& ws : workspaces) {
            if (ws) pyrsgm_destroy_workspace(ws);
            ws = nullptr;
        }
    }

    int postFlags(int level) const
    {
        if (level != 0) return PYRSGM_POST_LRCHECK;  // intermediate: LR check only, no median
        int flags = 0;
        if (params.lr_check) flags |= PYRSGM_POST_LRCHECK;
        if (params.median_filter) flags |= PYRSGM_POST_MEDIAN;
        return flags;
    }

    void compute(const uint8_t* d_left, size_t left_pitch,
                 const uint8_t* d_right, size_t right_pitch,
                 int16_t* d_disp16, size_t disp16_pitch,
                 float* d_disp32, size_t disp32_pitch,
                 float* d_conf, size_t conf_pitch,
                 cudaStream_t stream)
    {
        if (!d_left || !d_right) throw std::invalid_argument("pyrsgm: null input image");
        if (!d_disp16 && !d_disp32 && !d_conf) return;  // nothing requested

        float* out_conf = d_conf;
        size_t out_conf_pitch = conf_pitch;
        if (!out_conf && d_disp32) {
            out_conf = conf[0].data;
            out_conf_pitch = conf[0].pitch;
        }
        const int cost_mode = params.cost_mode == CostMode::Census ? PYRSGM_COST_CENSUS : PYRSGM_COST_BT;

        if (levels == 0) {
            const int ret = pyrsgm_compute_single_scale(
                workspaces[0], d_left, d_right,
                d_disp16, d_disp32, out_conf,
                left_pitch, right_pitch, disp16_pitch, disp32_pitch, out_conf_pitch,
                width, height,
                params.num_disparities, params.min_disparity,
                params.block_size, params.p1, params.p2,
                params.uniqueness_ratio, params.disp12_max_diff,
                params.pre_filter_cap, cost_mode, postFlags(0), params.num_paths,
                params.adaptive_p2 ? 1 : 0,
                params.p2_alpha, params.p2_beta, params.p2_lower_mult, params.p2_upper_mult,
                stream);
            if (ret != 0) throwCuda("single-scale SGM", cudaGetLastError());
            return;
        }

        // Gaussian pyramid of both views.
        const uint8_t* prev_l = d_left;  size_t prev_lp = left_pitch;
        const uint8_t* prev_r = d_right; size_t prev_rp = right_pitch;
        for (int level = 1; level <= levels; ++level) {
            const LevelInfo& parent = info[level - 1];
            const LevelInfo& li = info[level];
            if (pyrsgm_pyrdown_u8(prev_l, prev_lp, parent.width, parent.height,
                                  pyr_left[level].data, pyr_left[level].pitch,
                                  li.width, li.height, stream) != 0 ||
                pyrsgm_pyrdown_u8(prev_r, prev_rp, parent.width, parent.height,
                                  pyr_right[level].data, pyr_right[level].pitch,
                                  li.width, li.height, stream) != 0) {
                throwCuda("pyrDown", cudaGetLastError());
            }
            prev_l = pyr_left[level].data;  prev_lp = pyr_left[level].pitch;
            prev_r = pyr_right[level].data; prev_rp = pyr_right[level].pitch;
        }

        // Coarse-to-fine.
        for (int level = levels; level >= 0; --level) {
            const LevelInfo& li = info[level];
            const bool refine = li.refine;
            const uint8_t* lvl_left  = level == 0 ? d_left  : pyr_left[level].data;
            const uint8_t* lvl_right = level == 0 ? d_right : pyr_right[level].data;
            const size_t lvl_lp = level == 0 ? left_pitch  : pyr_left[level].pitch;
            const size_t lvl_rp = level == 0 ? right_pitch : pyr_right[level].pitch;

            const DeviceImage<int16_t>* parent_disp = refine ? &disp16[level + 1] : nullptr;
            const DeviceImage<float>*   parent_conf = refine ? &conf[level + 1] : nullptr;

            int16_t* out16 = level == 0 ? d_disp16 : disp16[level].data;
            size_t out16_pitch = level == 0 ? disp16_pitch : disp16[level].pitch;
            float* out32 = level == 0 ? d_disp32 : nullptr;
            size_t out32_pitch = level == 0 ? disp32_pitch : 0;
            float* outc = level == 0 ? out_conf : conf[level].data;
            size_t outc_pitch = level == 0 ? out_conf_pitch : conf[level].pitch;

            const int ret = pyrsgm_compute_level(
                workspaces[level],
                lvl_left, lvl_right,
                parent_disp ? parent_disp->data : nullptr,
                parent_conf ? parent_conf->data : nullptr,
                out16, out32, outc,
                lvl_lp, lvl_rp,
                parent_disp ? parent_disp->pitch : 0,
                parent_conf ? parent_conf->pitch : 0,
                out16_pitch, out32_pitch, outc_pitch,
                li.width, li.height,
                parent_disp ? parent_disp->width : 0,
                parent_disp ? parent_disp->height : 0,
                li.num_labels, params.min_disparity, li.max_disparity,
                li.block_size, params.p1, params.p2,
                params.uniqueness_ratio, params.disp12_max_diff,
                params.pre_filter_cap, cost_mode, postFlags(level), params.num_paths,
                refine ? 1 : 0, params.pyramid_min_confidence,
                params.adaptive_p2 ? 1 : 0,
                params.p2_alpha, params.p2_beta, params.p2_lower_mult, params.p2_upper_mult,
                stream);
            if (ret != 0) {
                std::ostringstream oss;
                oss << "pyramid level " << level << " (" << li.width << "x" << li.height
                    << ", D=" << li.num_labels << ", refine=" << refine << ")";
                throwCuda(oss.str().c_str(), cudaGetLastError());
            }
        }
    }

    size_t deviceBytes() const
    {
        size_t total = workspace_bytes;
        for (int level = 0; level <= levels; ++level) {
            total += pyr_left[level].bytes() + pyr_right[level].bytes() +
                     disp16[level].bytes() + conf[level].bytes();
        }
        return total;
    }
};

StereoMatcher::StereoMatcher(int width, int height, const Params& params)
    : impl_(new Impl(width, height, params)) {}
StereoMatcher::~StereoMatcher() = default;
StereoMatcher::StereoMatcher(StereoMatcher&&) noexcept = default;
StereoMatcher& StereoMatcher::operator=(StereoMatcher&&) noexcept = default;

void StereoMatcher::compute(const uint8_t* d_left, size_t left_pitch,
                            const uint8_t* d_right, size_t right_pitch,
                            int16_t* d_disp16, size_t disp16_pitch,
                            float* d_disp32, size_t disp32_pitch,
                            float* d_conf, size_t conf_pitch,
                            cudaStream_t stream)
{
    impl_->compute(d_left, left_pitch, d_right, right_pitch,
                   d_disp16, disp16_pitch, d_disp32, disp32_pitch,
                   d_conf, conf_pitch, stream);
}

int StereoMatcher::width() const { return impl_->width; }
int StereoMatcher::height() const { return impl_->height; }
const Params& StereoMatcher::params() const { return impl_->params; }
int StereoMatcher::numLevels() const { return impl_->levels + 1; }
LevelInfo StereoMatcher::levelInfo(int level) const
{
    if (level < 0 || level > impl_->levels) throw std::out_of_range("pyrsgm: level");
    return impl_->info[level];
}
size_t StereoMatcher::deviceMemoryBytes() const { return impl_->deviceBytes(); }

const char* version() { return "0.1.0"; }

} // namespace pyrsgm
