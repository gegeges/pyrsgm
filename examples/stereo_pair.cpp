// Minimal example: rectified left/right images -> disparity PNG + raw PFM.
//
//   pyrsgm_stereo_pair left.png right.png out_prefix [num_disparities] [pyramid_levels] [bt|census]
//
// Writes <out_prefix>_disp.png (colour-mapped) and <out_prefix>_disp.pfm (float
// disparity, 0 = invalid) plus <out_prefix>_conf.png.

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "pyrsgm/pyrsgm.h"

namespace {

struct DeviceBuffer {
    void* ptr = nullptr;
    size_t pitch = 0;
    DeviceBuffer(size_t width_bytes, int height)
    {
        if (cudaMallocPitch(&ptr, &pitch, width_bytes, (size_t)height) != cudaSuccess) {
            throw std::runtime_error("cudaMallocPitch failed");
        }
    }
    ~DeviceBuffer() { cudaFree(ptr); }
};

void writePfm(const std::string& path, const cv::Mat& disp)
{
    std::ofstream f(path, std::ios::binary);
    f << "Pf\n" << disp.cols << " " << disp.rows << "\n-1.0\n";
    for (int y = disp.rows - 1; y >= 0; --y) {
        f.write(reinterpret_cast<const char*>(disp.ptr<float>(y)), (std::streamsize)(disp.cols * sizeof(float)));
    }
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::cerr << "usage: " << argv[0]
                  << " left.png right.png out_prefix [num_disparities=128] [pyramid_levels=2] [bt|census]\n";
        return 2;
    }
    const cv::Mat left = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    const cv::Mat right = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
    if (left.empty() || right.empty() || left.size() != right.size()) {
        std::cerr << "cannot read images or sizes differ\n";
        return 1;
    }
    const std::string prefix = argv[3];

    pyrsgm::Params params;
    params.num_disparities = argc > 4 ? std::atoi(argv[4]) : 128;
    params.pyramid_levels = argc > 5 ? std::atoi(argv[5]) : 2;
    if (argc > 6 && std::string(argv[6]) == "census") {
        params.cost_mode = pyrsgm::CostMode::Census;
        params.p1 = 8;
        params.p2 = 32;
    }

    try {
        pyrsgm::StereoMatcher sgm(left.cols, left.rows, params);
        std::cout << "pyrsgm " << pyrsgm::version() << ": " << left.cols << "x" << left.rows
                  << ", D=" << sgm.params().num_disparities << ", levels=" << sgm.numLevels() - 1
                  << ", device memory " << sgm.deviceMemoryBytes() / (1024 * 1024) << " MiB\n";
        for (int l = 0; l < sgm.numLevels(); ++l) {
            const pyrsgm::LevelInfo li = sgm.levelInfo(l);
            std::cout << "  level " << l << ": " << li.width << "x" << li.height << ", labels " << li.num_labels
                      << ", max disparity " << li.max_disparity << (li.refine ? " (refine)" : " (coarse)") << "\n";
        }

        DeviceBuffer d_left(left.cols, left.rows), d_right(left.cols, left.rows);
        DeviceBuffer d_disp(left.cols * sizeof(float), left.rows), d_conf(left.cols * sizeof(float), left.rows);
        cudaMemcpy2D(d_left.ptr, d_left.pitch, left.data, left.step, left.cols, left.rows, cudaMemcpyHostToDevice);
        cudaMemcpy2D(d_right.ptr, d_right.pitch, right.data, right.step, right.cols, right.rows, cudaMemcpyHostToDevice);

        // Warm-up, then time a few runs.
        sgm.compute((const uint8_t*)d_left.ptr, d_left.pitch, (const uint8_t*)d_right.ptr, d_right.pitch,
                    nullptr, 0, (float*)d_disp.ptr, d_disp.pitch, (float*)d_conf.ptr, d_conf.pitch);
        cudaDeviceSynchronize();
        const int runs = 10;
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < runs; ++i) {
            sgm.compute((const uint8_t*)d_left.ptr, d_left.pitch, (const uint8_t*)d_right.ptr, d_right.pitch,
                        nullptr, 0, (float*)d_disp.ptr, d_disp.pitch, (float*)d_conf.ptr, d_conf.pitch);
        }
        cudaDeviceSynchronize();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / runs;
        std::cout << "  " << ms << " ms per frame (" << 1000.0 / ms << " fps)\n";

        cv::Mat disp(left.rows, left.cols, CV_32FC1), conf(left.rows, left.cols, CV_32FC1);
        cudaMemcpy2D(disp.data, disp.step, d_disp.ptr, d_disp.pitch, left.cols * sizeof(float), left.rows, cudaMemcpyDeviceToHost);
        cudaMemcpy2D(conf.data, conf.step, d_conf.ptr, d_conf.pitch, left.cols * sizeof(float), left.rows, cudaMemcpyDeviceToHost);

        cv::Mat u8, color;
        disp.convertTo(u8, CV_8UC1, 255.0 / params.num_disparities);
        cv::applyColorMap(u8, color, cv::COLORMAP_TURBO);
        color.setTo(cv::Scalar(0, 0, 0), disp <= 0.0f);
        cv::imwrite(prefix + "_disp.png", color);
        cv::Mat conf8;
        conf.convertTo(conf8, CV_8UC1, 255.0);
        cv::imwrite(prefix + "_conf.png", conf8);
        writePfm(prefix + "_disp.pfm", disp);
        std::cout << "  wrote " << prefix << "_disp.png / _conf.png / _disp.pfm\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    return 0;
}
