// C++ API test: parameter validation, pyrDown against a CPU reference, and
// StereoMatcher at pyramid levels 0..4 with BT / Census on synthetic scenes
// (constant disparity and a slanted plane).

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "pyrsgm/pyrsgm.h"
#include "pyrsgm/pyrsgm_c.h"
#include "test_utils.h"

using testutil::DevImage;

namespace {

int failures = 0;

void expect(bool cond, const std::string& what)
{
    if (!cond) {
        ++failures;
        std::cerr << "FAIL: " << what << "\n";
    }
}

bool throwsInvalid(const pyrsgm::Params& p)
{
    try {
        pyrsgm::validateParams(p);
    } catch (const std::invalid_argument&) {
        return true;
    }
    return false;
}

void testValidation()
{
    pyrsgm::Params p;
    pyrsgm::Params v = pyrsgm::validateParams(p);
    expect(v.num_disparities == 128 && v.block_size == 5, "defaults survive validation");
    expect(v.pyramid_max_disparity == 0, "legacy mode leaves pyramid_max_disparity alone");

    p.num_disparities = 100;
    v = pyrsgm::validateParams(p);
    expect(v.num_disparities == 112, "num_disparities aligned to 16");
    p.block_size = 4;
    expect(pyrsgm::validateParams(p).block_size == 5, "block_size made odd");

    p = pyrsgm::Params();
    p.pyramid_levels = 2;
    v = pyrsgm::validateParams(p);
    expect(v.pyramid_max_disparity == 127, "pyramid_max_disparity derived from num_disparities");

    p = pyrsgm::Params(); p.pyramid_levels = 5;
    expect(throwsInvalid(p), "pyramid_levels > 4 rejected");
    p = pyrsgm::Params(); p.pyramid_levels = 1; p.min_disparity = 8;
    expect(throwsInvalid(p), "pyramid with min_disparity != 0 rejected");
    p = pyrsgm::Params(); p.pyramid_levels = 1; p.disp12_max_diff = -1;
    expect(throwsInvalid(p), "pyramid with disp12_max_diff < 0 rejected");
    p = pyrsgm::Params(); p.num_disparities = 2048;
    expect(throwsInvalid(p), "num_disparities > 1024 rejected");
    p = pyrsgm::Params(); p.p1 = 100; p.p2 = 50;
    expect(throwsInvalid(p), "p2 < p1 rejected");
    p = pyrsgm::Params(); p.adaptive_p2 = true; p.p2_beta = 0.0f;
    expect(throwsInvalid(p), "adaptive_p2 with beta 0 rejected");
}

int reflect101(int i, int n)
{
    if (i < 0) i = -i;
    if (i >= n) i = 2 * n - 2 - i;
    return i;
}

std::vector<uint8_t> pyrDownRef(const std::vector<uint8_t>& src, int w, int h, int& dw, int& dh)
{
    dw = (w + 1) / 2;
    dh = (h + 1) / 2;
    const int k[5] = {1, 4, 6, 4, 1};
    std::vector<uint8_t> dst((size_t)dw * dh);
    for (int y = 0; y < dh; ++y) {
        for (int x = 0; x < dw; ++x) {
            int sum = 0;
            for (int ky = 0; ky < 5; ++ky) {
                const int sy = reflect101(2 * y + ky - 2, h);
                for (int kx = 0; kx < 5; ++kx) {
                    const int sx = reflect101(2 * x + kx - 2, w);
                    sum += k[ky] * k[kx] * src[(size_t)sy * w + sx];
                }
            }
            dst[(size_t)y * dw + x] = static_cast<uint8_t>((sum + 128) >> 8);
        }
    }
    return dst;
}

void testPyrDown()
{
    for (int w : {64, 97}) {
        for (int h : {48, 33}) {
            std::vector<uint8_t> src((size_t)w * h);
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x)
                    src[(size_t)y * w + x] = static_cast<uint8_t>((x * 13 + y * 29 + (x * y) / 7) & 0xff);
            int dw = 0, dh = 0;
            const std::vector<uint8_t> ref = pyrDownRef(src, w, h, dw, dh);
            DevImage<uint8_t> d_src(w, h), d_dst(dw, dh);
            d_src.upload(src);
            const int ret = pyrsgm_pyrdown_u8(d_src.data, d_src.pitch, w, h,
                                              d_dst.data, d_dst.pitch, dw, dh, nullptr);
            expect(ret == 0 && cudaDeviceSynchronize() == cudaSuccess, "pyrDown launch");
            const std::vector<uint8_t> got = d_dst.download();
            int mismatches = 0;
            for (size_t i = 0; i < got.size(); ++i) mismatches += got[i] != ref[i];
            expect(mismatches == 0, "pyrDown matches CPU reference " + std::to_string(w) + "x" +
                                        std::to_string(h) + " (mismatches=" +
                                        std::to_string(mismatches) + ")");
        }
    }
}

struct Scene {
    int width, height;
    std::vector<uint8_t> left, right;
    std::vector<float> gt;  // per-pixel disparity
};

Scene constantScene(int w, int h, int disparity)
{
    Scene s{w, h, testutil::makePattern(w, h, disparity, false),
            testutil::makePattern(w, h, disparity, true), std::vector<float>((size_t)w * h, (float)disparity)};
    return s;
}

// Slanted plane expressed in *left* image coordinates: left pixel x has
// disparity d(x) = d0 + slope * x and matches right pixel xr = x - d(x).
// Inverting, right pixel xr samples the texture at x = (xr + d0) / (1 - slope).
// The left image is the texture at integer positions, the right at sub-pixel
// positions (bilinear).
Scene slantedScene(int w, int h, float d0, float slope)
{
    Scene s;
    s.width = w;
    s.height = h;
    s.left.resize((size_t)w * h);
    s.right.resize((size_t)w * h);
    s.gt.resize((size_t)w * h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            s.left[(size_t)y * w + x] = testutil::sampleTexture((float)x, y);
            s.right[(size_t)y * w + x] =
                testutil::sampleTexture(((float)x + d0) / (1.0f - slope), y);
            s.gt[(size_t)y * w + x] = d0 + slope * (float)x;
        }
    }
    return s;
}

struct RunResult {
    testutil::DispStats stats;
    std::vector<int16_t> disp16;
    std::vector<float> disp32, conf;
};

RunResult run(const Scene& scene, const pyrsgm::Params& params, cudaStream_t stream,
              bool want16 = true, bool want32 = true, bool wantConf = true)
{
    pyrsgm::StereoMatcher sgm(scene.width, scene.height, params);
    DevImage<uint8_t> left(scene.width, scene.height), right(scene.width, scene.height);
    left.upload(scene.left);
    right.upload(scene.right);
    DevImage<int16_t> disp16(scene.width, scene.height);
    DevImage<float> disp32(scene.width, scene.height), conf(scene.width, scene.height);
    sgm.compute(left.data, left.pitch, right.data, right.pitch,
                want16 ? disp16.data : nullptr, disp16.pitch,
                want32 ? disp32.data : nullptr, disp32.pitch,
                wantConf ? conf.data : nullptr, conf.pitch, stream);
    expect(cudaStreamSynchronize(stream) == cudaSuccess, "compute stream sync");
    RunResult r;
    r.disp16 = disp16.download();
    r.disp32 = disp32.download();
    r.conf = conf.download();
    // Margin keeps the unmatched left border (max disparity) and the pattern
    // wrap-around out of the statistics.
    const int margin = params.num_disparities + 8;
    r.stats = testutil::compareDisparity(
        r.disp32, scene.width, scene.height,
        [&](int x, int y) { return scene.gt[(size_t)y * scene.width + x]; }, 1.0f, margin);
    return r;
}

void testMatcher()
{
    cudaStream_t stream = nullptr;
    expect(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) == cudaSuccess, "stream create");

    const Scene constant = constantScene(640, 480, 40);
    const Scene slanted = slantedScene(640, 480, 12.0f, 0.06f);  // 12 .. 50 px

    for (int levels = 0; levels <= 4; ++levels) {
        for (pyrsgm::CostMode cost : {pyrsgm::CostMode::BirchfieldTomasi, pyrsgm::CostMode::Census}) {
            pyrsgm::Params p;
            p.num_disparities = 64;
            p.pyramid_levels = levels;
            p.cost_mode = cost;
            p.p1 = cost == pyrsgm::CostMode::Census ? 8 : 300;
            p.p2 = cost == pyrsgm::CostMode::Census ? 32 : 2400;
            const std::string tag = "L" + std::to_string(levels) +
                                    (cost == pyrsgm::CostMode::Census ? " Census" : " BT");

            const RunResult c = run(constant, p, stream);
            const double density = c.stats.total ? (double)c.stats.valid / c.stats.total : 0.0;
            const double good = c.stats.valid ? (double)c.stats.good / c.stats.valid : 0.0;
            std::cout << "  " << tag << " constant: density=" << density
                      << " good(<=1px)=" << good << " mae=" << c.stats.mean_abs_err << "\n";
            expect(density > 0.5, tag + " constant density > 50%");
            expect(good > 0.95, tag + " constant >95% within 1px");

            // 16-bit and float outputs agree; invalid marker respected.
            int disagreements = 0, invalid16 = 0;
            const int16_t invalid = pyrsgm::invalidDisparity16(p);
            for (size_t i = 0; i < c.disp16.size(); ++i) {
                if (c.disp16[i] == invalid) {
                    ++invalid16;
                    if (c.disp32[i] != 0.0f) ++disagreements;
                } else if (std::fabs(c.disp16[i] / 16.0f - c.disp32[i]) > 1e-6f && c.disp32[i] != 0.0f) {
                    ++disagreements;
                }
            }
            expect(disagreements == 0, tag + " disp16/disp32 consistent");
            expect(invalid16 > 0, tag + " border pixels are invalid");

            const RunResult s = run(slanted, p, stream);
            const double sgood = s.stats.valid ? (double)s.stats.good / s.stats.valid : 0.0;
            std::cout << "  " << tag << " slanted : density="
                      << (s.stats.total ? (double)s.stats.valid / s.stats.total : 0.0)
                      << " good(<=1px)=" << sgood << " mae=" << s.stats.mean_abs_err << "\n";
            expect(sgood > 0.9, tag + " slanted >90% within 1px");
            expect(s.stats.mean_abs_err < 0.5, tag + " slanted mean abs error < 0.5px");
        }
    }

    // Output subsets.
    {
        pyrsgm::Params p;
        p.num_disparities = 64;
        p.pyramid_levels = 2;
        const RunResult only16 = run(constant, p, stream, true, false, false);
        int valid = 0;
        for (int16_t v : only16.disp16) valid += v != pyrsgm::invalidDisparity16(p);
        expect(valid > 0, "disp16-only output produced");
        const RunResult only32 = run(constant, p, stream, false, true, false);
        expect(only32.stats.valid > 0, "disp32-only output produced (scratch confidence)");
        const RunResult onlyConf = run(constant, p, stream, false, false, true);
        int nonzero = 0;
        for (float v : onlyConf.conf) nonzero += v > 0.0f;
        expect(nonzero > 0, "confidence-only output produced");
    }

    // Level info / memory accounting.
    {
        pyrsgm::Params p;
        p.num_disparities = 128;
        p.pyramid_levels = 3;
        pyrsgm::StereoMatcher sgm(961, 541, p);
        expect(sgm.numLevels() == 4, "numLevels");
        const pyrsgm::LevelInfo l3 = sgm.levelInfo(3);
        expect(l3.width == 121 && l3.height == 68, "coarsest level size uses ceil(w/2)");
        expect(!l3.refine && l3.num_labels == 16 && l3.max_disparity == 15,
               "coarsest labels = align16(min(128, (127>>3)+1)) = 16");
        expect(sgm.levelInfo(0).refine && sgm.levelInfo(0).num_labels == 16, "fine level refines with 16 labels");
        expect(sgm.levelInfo(2).block_size == 3 && sgm.levelInfo(1).block_size == 5,
               "coarse levels use 3x3 BT window");
        expect(sgm.deviceMemoryBytes() > 0, "device memory accounted");
    }

    cudaStreamDestroy(stream);
}

} // namespace

int main()
{
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cout << "CUDA device unavailable; skipping\n";
        return 77;
    }
    std::cout << "pyrsgm " << pyrsgm::version() << "\n";
    testValidation();
    testPyrDown();
    testMatcher();
    if (failures) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "pyrsgm matcher test passed\n";
    return 0;
}
