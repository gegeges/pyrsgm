// Minimal CUDA helpers shared by the pyrsgm tests (no OpenCV dependency).
#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace testutil {

inline bool cudaOk(cudaError_t err, const char* what)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
        return false;
    }
    return true;
}

template <class T>
struct DevImage {
    T* data = nullptr;
    size_t pitch = 0;
    int width = 0;
    int height = 0;

    DevImage() = default;
    DevImage(int w, int h) { create(w, h); }
    DevImage(const DevImage&) = delete;
    DevImage& operator=(const DevImage&) = delete;
    ~DevImage() { release(); }

    void create(int w, int h)
    {
        release();
        void* p = nullptr;
        if (!cudaOk(cudaMallocPitch(&p, &pitch, (size_t)w * sizeof(T), (size_t)h), "cudaMallocPitch")) {
            std::abort();
        }
        data = static_cast<T*>(p);
        width = w;
        height = h;
    }
    void release()
    {
        if (data) cudaFree(data);
        data = nullptr;
        pitch = 0;
        width = height = 0;
    }
    void upload(const std::vector<T>& host)
    {
        cudaOk(cudaMemcpy2D(data, pitch, host.data(), (size_t)width * sizeof(T),
                            (size_t)width * sizeof(T), (size_t)height, cudaMemcpyHostToDevice),
               "upload");
    }
    std::vector<T> download() const
    {
        std::vector<T> host((size_t)width * height);
        cudaOk(cudaMemcpy2D(host.data(), (size_t)width * sizeof(T), data, pitch,
                            (size_t)width * sizeof(T), (size_t)height, cudaMemcpyDeviceToHost),
               "download");
        return host;
    }
};

/// Deterministic textured pattern; the right image is the left shifted by
/// `disparity` pixels (right(x) = left(x + disparity)).
inline std::vector<uint8_t> makePattern(int width, int height, int disparity, bool right)
{
    std::vector<uint8_t> image((size_t)width * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int source_x = right ? x + disparity : x;
            image[(size_t)y * width + x] = static_cast<uint8_t>(
                (source_x * 37 + y * 17 + ((source_x / 7) ^ (y / 5)) * 29) & 0xff);
        }
    }
    return image;
}

/// Smooth, noise-free natural-looking texture with a planar disparity ramp
/// is harder for SGM than a hash pattern; we keep the hash pattern plus a
/// slanted-plane variant: disparity(x) = d0 + slope * x (sub-pixel shifts via
/// bilinear sampling of the hash texture).
inline uint8_t sampleTexture(float x, int y)
{
    auto tex = [](int sx, int sy) {
        return (sx * 37 + sy * 17 + ((sx / 7) ^ (sy / 5)) * 29) & 0xff;
    };
    const int x0 = (int)std::floor(x);
    const float f = x - (float)x0;
    const float v = (1.0f - f) * (float)tex(x0, y) + f * (float)tex(x0 + 1, y);
    return static_cast<uint8_t>(v + 0.5f);
}

struct DispStats {
    int valid = 0;
    int good = 0;
    int total = 0;
    double mean_abs_err = 0.0;
};

/// Compares float disparity (invalid = 0) with a per-pixel expected value.
template <class Expected>
inline DispStats compareDisparity(const std::vector<float>& disp, int width, int height,
                                  Expected expected, float tolerance, int margin)
{
    DispStats s;
    double err_sum = 0.0;
    for (int y = margin; y < height - margin; ++y) {
        for (int x = margin; x < width - margin; ++x) {
            ++s.total;
            const float d = disp[(size_t)y * width + x];
            if (!(d > 0.0f)) continue;
            ++s.valid;
            const float e = std::fabs(d - expected(x, y));
            err_sum += e;
            if (e <= tolerance) ++s.good;
        }
    }
    s.mean_abs_err = s.valid ? err_sum / s.valid : 0.0;
    return s;
}

inline bool hasExpectedDisparity(const std::vector<float>& disp, int width, int height,
                                 float expected, float tolerance)
{
    const DispStats s = compareDisparity(
        disp, width, height, [expected](int, int) { return expected; }, tolerance, 0);
    return s.valid > 0 && s.good >= s.valid / 2;
}

} // namespace testutil
