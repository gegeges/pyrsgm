// Kernel-level smoke test of the pyramid C API.
//
// Chains a real coarse result through four refinement levels (1/16 .. 1/1),
// exercises spatially varying parent bases (both bounds of same±1 in every
// path kernel), BT and Census costs, and 1/2/3 aggregation paths.
// Ported from Tri3SGM tests/cuda_pyramid_smoke.cpp (OpenCV removed).

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <vector>

#include "pyrsgm/pyrsgm_c.h"
#include "test_utils.h"

using testutil::DevImage;

namespace {

bool runLevel(void* workspace,
              const DevImage<uint8_t>& left, const DevImage<uint8_t>& right,
              const DevImage<int16_t>* parent_disp, const DevImage<float>* parent_conf,
              DevImage<int16_t>& disp, DevImage<float>& disp_float, DevImage<float>& conf,
              int labels, int max_disparity, bool refine,
              int cost_mode = PYRSGM_COST_BT, int paths = 3)
{
    disp.create(left.width, left.height);
    disp_float.create(left.width, left.height);
    conf.create(left.width, left.height);
    const int ret = pyrsgm_compute_level(
        workspace,
        left.data, right.data,
        parent_disp ? parent_disp->data : nullptr,
        parent_conf ? parent_conf->data : nullptr,
        disp.data, disp_float.data, conf.data,
        left.pitch, right.pitch,
        parent_disp ? parent_disp->pitch : 0,
        parent_conf ? parent_conf->pitch : 0,
        disp.pitch, disp_float.pitch, conf.pitch,
        left.width, left.height,
        parent_disp ? parent_disp->width : 0,
        parent_disp ? parent_disp->height : 0,
        labels, 0, max_disparity,
        3, 24, 96,
        0, 1, 63,
        cost_mode, PYRSGM_POST_LRCHECK, paths,
        refine ? 1 : 0, 0.0f,
        1, 2.0f, 14.0f, 0.75f, 2.0f,
        nullptr);
    return ret == 0 && cudaDeviceSynchronize() == cudaSuccess;
}

bool expectDisparity(const DevImage<float>& disp, float expected, float tolerance,
                     const char* label)
{
    const bool ok = testutil::hasExpectedDisparity(disp.download(), disp.width, disp.height,
                                                   expected, tolerance);
    if (!ok) std::cerr << "  [" << label << "] expected disparity " << expected << " not found\n";
    return ok;
}

struct Level {
    DevImage<uint8_t> left, right;
    DevImage<int16_t> disp;
    DevImage<float> disp_float, conf;
    void* workspace = nullptr;

    Level(int w, int h, int disparity)
    {
        left.create(w, h);
        right.create(w, h);
        left.upload(testutil::makePattern(w, h, disparity, false));
        right.upload(testutil::makePattern(w, h, disparity, true));
        workspace = pyrsgm_create_workspace();
    }
    ~Level() { pyrsgm_destroy_workspace(workspace); }
};

} // namespace

int main()
{
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cout << "CUDA device unavailable; skipping\n";
        return 77;
    }

    bool ok = true;

    // 1/16 -> 1/8 -> 1/4 -> 1/2 -> 1/1 with doubling disparity.
    Level l4(40, 16, 2), l3(80, 32, 4), l2(160, 64, 8), l1(320, 128, 16), l0(640, 256, 32);

    ok = ok && runLevel(l4.workspace, l4.left, l4.right, nullptr, nullptr,
                        l4.disp, l4.disp_float, l4.conf, 32, 31, false);
    ok = ok && expectDisparity(l4.disp_float, 2.0f, 1.5f, "coarsest");

    ok = ok && runLevel(l3.workspace, l3.left, l3.right, &l4.disp, &l4.conf,
                        l3.disp, l3.disp_float, l3.conf, 16, 63, true);
    ok = ok && expectDisparity(l3.disp_float, 4.0f, 1.5f, "refine 1/8");

    ok = ok && runLevel(l2.workspace, l2.left, l2.right, &l3.disp, &l3.conf,
                        l2.disp, l2.disp_float, l2.conf, 16, 127, true);
    ok = ok && expectDisparity(l2.disp_float, 8.0f, 1.5f, "refine 1/4");

    ok = ok && runLevel(l1.workspace, l1.left, l1.right, &l2.disp, &l2.conf,
                        l1.disp, l1.disp_float, l1.conf, 16, 255, true);
    ok = ok && expectDisparity(l1.disp_float, 16.0f, 1.5f, "refine 1/2");

    ok = ok && runLevel(l0.workspace, l0.left, l0.right, &l1.disp, &l1.conf,
                        l0.disp, l0.disp_float, l0.conf, 16, 511, true);
    ok = ok && expectDisparity(l0.disp_float, 32.0f, 2.0f, "refine 1/1");

    // Large base jumps between adjacent parent pixels exercise both bounds of
    // same±1 in all pyramid path kernels (1 and 3 paths).
    {
        std::vector<int16_t> parent_disp((size_t)160 * 64);
        std::vector<float> parent_conf((size_t)160 * 64, 1.0f);
        for (int y = 0; y < 64; ++y)
            for (int x = 0; x < 160; ++x)
                parent_disp[(size_t)y * 160 + x] = static_cast<int16_t>(((x & 1) ? 30 : 2) * 16);
        DevImage<int16_t> d_parent(160, 64);
        DevImage<float> d_conf(160, 64);
        d_parent.upload(parent_disp);
        d_conf.upload(parent_conf);
        DevImage<int16_t> out;
        DevImage<float> out_float, out_conf;
        ok = ok && runLevel(l1.workspace, l1.left, l1.right, &d_parent, &d_conf,
                            out, out_float, out_conf, 16, 255, true);
        ok = ok && runLevel(l1.workspace, l1.left, l1.right, &d_parent, &d_conf,
                            out, out_float, out_conf, 16, 255, true, PYRSGM_COST_BT, 1);
        if (!ok) std::cerr << "  [varying base] launch failed\n";
    }

    // Census cost, two paths, coarse search.
    {
        DevImage<int16_t> disp;
        DevImage<float> disp_float, conf;
        ok = ok && runLevel(l2.workspace, l2.left, l2.right, nullptr, nullptr,
                            disp, disp_float, conf, 128, 127, false, PYRSGM_COST_CENSUS, 2);
        ok = ok && expectDisparity(disp_float, 8.0f, 2.0f, "census coarse");
    }

    // Single-scale legacy entry point, BT and Census.
    for (int cost_mode : {PYRSGM_COST_BT, PYRSGM_COST_CENSUS}) {
        DevImage<int16_t> disp(l0.left.width, l0.left.height);
        DevImage<float> disp_float(l0.left.width, l0.left.height);
        DevImage<float> conf(l0.left.width, l0.left.height);
        const int ret = pyrsgm_compute_single_scale(
            l0.workspace, l0.left.data, l0.right.data,
            disp.data, disp_float.data, conf.data,
            l0.left.pitch, l0.right.pitch, disp.pitch, disp_float.pitch, conf.pitch,
            l0.left.width, l0.left.height,
            64, 0, 5,
            cost_mode == PYRSGM_COST_BT ? 300 : 8,
            cost_mode == PYRSGM_COST_BT ? 2400 : 32,
            10, 1, 63, cost_mode,
            PYRSGM_POST_LRCHECK | PYRSGM_POST_MEDIAN, 3,
            0, 0.0f, 1.0f, 1.0f, 1.0f, nullptr);
        ok = ok && ret == 0 && cudaDeviceSynchronize() == cudaSuccess;
        ok = ok && expectDisparity(disp_float, 32.0f, 1.0f,
                                   cost_mode == PYRSGM_COST_BT ? "single-scale BT" : "single-scale Census");
    }

    if (!ok) {
        std::cerr << "pyrsgm pyramid smoke test FAILED: "
                  << cudaGetErrorString(cudaGetLastError()) << "\n";
        return 1;
    }
    std::cout << "pyrsgm pyramid smoke test passed\n";
    return 0;
}
