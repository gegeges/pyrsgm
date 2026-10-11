// Middlebury (MiddEval3) I/O and disparity metrics for the pyrsgm benchmark.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace mb {

/// Reads a PFM (grayscale "Pf") file into CV_32FC1.  Unknown pixels (inf) are
/// kept as +inf.  Handles the little-endian scale sign and bottom-up row order.
inline cv::Mat readPFM(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::string magic;
    f >> magic;
    if (magic != "Pf") throw std::runtime_error(path + ": only grayscale PFM supported");
    int w = 0, h = 0;
    double scale = 0.0;
    f >> w >> h >> scale;
    f.get();  // single whitespace after the header
    if (w <= 0 || h <= 0) throw std::runtime_error(path + ": bad PFM header");
    const bool little_endian = scale < 0.0;
    cv::Mat img(h, w, CV_32FC1);
    std::vector<float> row((size_t)w);
    for (int y = h - 1; y >= 0; --y) {  // PFM stores bottom row first
        f.read(reinterpret_cast<char*>(row.data()), (std::streamsize)(w * sizeof(float)));
        if (!f) throw std::runtime_error(path + ": truncated PFM");
        float* dst = img.ptr<float>(y);
        for (int x = 0; x < w; ++x) {
            float v = row[x];
            if (!little_endian) {
                uint32_t u;
                std::memcpy(&u, &v, 4);
                u = ((u & 0xff) << 24) | ((u & 0xff00) << 8) | ((u >> 8) & 0xff00) | (u >> 24);
                std::memcpy(&v, &u, 4);
            }
            dst[x] = v;
        }
    }
    return img;
}

/// Parses "key=value" lines of a MiddEval3 calib.txt; returns ndisp (0 if absent).
inline int readNdisp(const std::string& path)
{
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("ndisp=", 0) == 0) return std::atoi(line.c_str() + 6);
    }
    return 0;
}

struct Metrics {
    int evaluated = 0;      ///< GT pixels used (non-occluded mask)
    int valid = 0;          ///< evaluated pixels with a valid estimate
    double avg_err = 0.0;   ///< mean |d - gt| over valid pixels
    double rms_err = 0.0;
    double bad05 = 0.0, bad1 = 0.0, bad2 = 0.0, bad4 = 0.0;  ///< % of *valid* pixels
    double density = 0.0;   ///< valid / evaluated %
    double invalid = 0.0;   ///< 100 - density
    double bad1_all = 0.0;  ///< % of evaluated pixels that are invalid or err > 1 (MiddEval3 "bad" with invalid counted)
    double bad2_all = 0.0;
};

/// disp: float disparity in pixels, <= 0 means invalid.  gt: float, inf = unknown.
/// mask: optional 8-bit nocc mask (255 = evaluate) — pass empty to evaluate all known GT.
inline Metrics evaluate(const cv::Mat& disp, const cv::Mat& gt, const cv::Mat& mask, float max_disp)
{
    CV_Assert(disp.type() == CV_32FC1 && gt.type() == CV_32FC1 && disp.size() == gt.size());
    Metrics m;
    double sum = 0.0, sq = 0.0;
    int b05 = 0, b1 = 0, b2 = 0, b4 = 0;
    for (int y = 0; y < gt.rows; ++y) {
        const float* g = gt.ptr<float>(y);
        const float* d = disp.ptr<float>(y);
        const uint8_t* mk = mask.empty() ? nullptr : mask.ptr<uint8_t>(y);
        for (int x = 0; x < gt.cols; ++x) {
            if (!std::isfinite(g[x]) || g[x] <= 0.0f || g[x] > max_disp) continue;
            if (mk && mk[x] != 255) continue;
            ++m.evaluated;
            if (!(d[x] > 0.0f)) continue;
            ++m.valid;
            const double e = std::fabs((double)d[x] - (double)g[x]);
            sum += e;
            sq += e * e;
            b05 += e > 0.5;
            b1 += e > 1.0;
            b2 += e > 2.0;
            b4 += e > 4.0;
        }
    }
    if (m.valid) {
        m.avg_err = sum / m.valid;
        m.rms_err = std::sqrt(sq / m.valid);
        m.bad05 = 100.0 * b05 / m.valid;
        m.bad1 = 100.0 * b1 / m.valid;
        m.bad2 = 100.0 * b2 / m.valid;
        m.bad4 = 100.0 * b4 / m.valid;
    }
    if (m.evaluated) {
        m.density = 100.0 * m.valid / m.evaluated;
        m.invalid = 100.0 - m.density;
        m.bad1_all = 100.0 * (b1 + (m.evaluated - m.valid)) / m.evaluated;
        m.bad2_all = 100.0 * (b2 + (m.evaluated - m.valid)) / m.evaluated;
    }
    return m;
}

} // namespace mb
