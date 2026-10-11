// pyrsgm benchmark
//
// Compares pyrsgm (single-scale and pyramid, BT / Census) against
//   * OpenCV cv::cuda::StereoSGM (libSGM port inside opencv_contrib, MODE_HH / MODE_HH4)
//   * fixstars libSGM (SCAN_8PATH / SCAN_4PATH)
//   * OpenCV CPU StereoSGBM MODE_SGBM_3WAY as the algorithmic reference of the BT path
// on Middlebury MiddEval3 training scenes (accuracy + GPU time) or on resized
// images (timing only).
//
// All GPU algorithms read inputs from and write outputs to device memory; the
// reported "gpu_ms" is CUDA-event time on the default stream around one call,
// "wall_ms" is host wall-clock including the final device synchronisation.

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#ifdef PYRSGM_HAVE_CV_CUDASTEREO
#include <opencv2/cudastereo.hpp>
#endif
#ifdef PYRSGM_HAVE_LIBSGM
#include <libsgm.h>
#endif

#include "middlebury.h"
#include "pyrsgm/pyrsgm.h"

namespace {

namespace fs_compat {
// Tiny helpers so we do not depend on <filesystem> on older toolchains.
bool exists(const std::string& p)
{
    std::ifstream f(p);
    return f.good();
}
std::string join(const std::string& a, const std::string& b) { return a + "/" + b; }
} // namespace fs_compat

struct Options {
    std::string dataset;                 // MiddEval3 trainingQ directory
    std::vector<std::string> scenes;     // subset (empty = all found)
    std::string left, right, gt, mask;   // single pair mode
    int ndisp_override = 0;
    std::vector<std::string> algos;      // empty = all available
    int iters = 20;
    int warmup = 3;
    double warmup_ms = 300.0;            // keep warming up until this much wall time has passed
    int rounds = 1;                      // repeat the whole sweep; samples are pooled
    std::string csv = "results.csv";
    std::string md;
    std::string save_dir;                // disparity visualisations
    std::vector<cv::Size> sizes;         // timing-only mode
    bool list = false;
    bool verbose = false;
};

const char* kDefaultScenes[] = {
    "Adirondack", "ArtL", "Jadeplant", "Motorcycle", "MotorcycleE", "Piano", "PianoL",
    "Pipes", "Playroom", "Playtable", "PlaytableP", "Recycle", "Shelves", "Teddy", "Vintage"};

std::vector<std::string> split(const std::string& s, char sep)
{
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, sep)) {
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

void usage()
{
    std::cout <<
        "pyrsgm_bench [options]\n"
        "  --dataset DIR        MiddEval3 trainingQ directory (scene/im0.png im1.png calib.txt disp0GT.pfm mask0nocc.png)\n"
        "  --scenes a,b,c       subset of scenes (default: all 15 training scenes present)\n"
        "  --left L --right R   single rectified pair (8-bit or colour PNG/JPG)\n"
        "  --gt PFM --mask PNG  optional ground truth / non-occluded mask for the single pair\n"
        "  --ndisp N            disparity range; rounded up to 64/128/256 (default: calib.txt ndisp)\n"
        "  --algos a,b          subset of algorithms (see --list)\n"
        "  --iters N            timed iterations per scene/algorithm (default 20)\n"
        "  --warmup N           minimum untimed warm-up iterations (default 3)\n"
        "  --warmup-ms T        keep warming up for at least T ms so clocks settle (default 300)\n"
        "  --rounds R           repeat the whole scene x algorithm sweep R times, pooling samples (default 1)\n"
        "  --sizes WxH,...      timing-only mode: resize the first scene / pair to these sizes\n"
        "  --csv FILE           per-scene CSV output (default results.csv)\n"
        "  --md FILE            markdown summary output\n"
        "  --save-disp DIR      write colour-mapped disparity PNGs per scene/algorithm\n"
        "  --list               list algorithms and exit\n"
        "  --verbose            per-iteration output\n";
}

Options parseArgs(int argc, char** argv)
{
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << name << " requires a value\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--dataset") o.dataset = next("--dataset");
        else if (a == "--scenes") o.scenes = split(next("--scenes"), ',');
        else if (a == "--left") o.left = next("--left");
        else if (a == "--right") o.right = next("--right");
        else if (a == "--gt") o.gt = next("--gt");
        else if (a == "--mask") o.mask = next("--mask");
        else if (a == "--ndisp") o.ndisp_override = std::atoi(next("--ndisp").c_str());
        else if (a == "--algos") o.algos = split(next("--algos"), ',');
        else if (a == "--iters") o.iters = std::atoi(next("--iters").c_str());
        else if (a == "--warmup") o.warmup = std::atoi(next("--warmup").c_str());
        else if (a == "--warmup-ms") o.warmup_ms = std::atof(next("--warmup-ms").c_str());
        else if (a == "--rounds") o.rounds = std::max(1, std::atoi(next("--rounds").c_str()));
        else if (a == "--csv") o.csv = next("--csv");
        else if (a == "--md") o.md = next("--md");
        else if (a == "--save-disp") o.save_dir = next("--save-disp");
        else if (a == "--sizes") {
            for (const std::string& s : split(next("--sizes"), ',')) {
                int w = 0, h = 0;
                if (std::sscanf(s.c_str(), "%dx%d", &w, &h) == 2 && w > 0 && h > 0) {
                    o.sizes.emplace_back(w, h);
                } else {
                    std::cerr << "bad size: " << s << "\n";
                    std::exit(2);
                }
            }
        }
        else if (a == "--list") o.list = true;
        else if (a == "--verbose") o.verbose = true;
        else if (a == "-h" || a == "--help") { usage(); std::exit(0); }
        else {
            std::cerr << "unknown option " << a << "\n";
            usage();
            std::exit(2);
        }
    }
    return o;
}

// ---------------------------------------------------------------------------
// Algorithms
// ---------------------------------------------------------------------------

struct Algorithm {
    virtual ~Algorithm() = default;
    virtual std::string name() const = 0;
    virtual std::string description() const = 0;
    virtual bool onGpu() const { return true; }
    /// Allocate everything for this geometry; D is 64/128/256.
    virtual void setup(int width, int height, int D) = 0;
    /// One computation; inputs/outputs are device images (CV_8UC1 / CV_16SC1 x16).
    virtual void run(const cv::cuda::GpuMat& left, const cv::cuda::GpuMat& right,
                     cv::cuda::GpuMat& disp16) = 0;
    virtual void teardown() {}
};

struct PyrsgmAlgo : Algorithm {
    int levels;
    pyrsgm::CostMode cost;
    std::unique_ptr<pyrsgm::StereoMatcher> sgm;
    pyrsgm::Params used;

    PyrsgmAlgo(int l, pyrsgm::CostMode c) : levels(l), cost(c) {}
    std::string name() const override
    {
        return "pyrsgm-L" + std::to_string(levels) +
               (cost == pyrsgm::CostMode::Census ? "-census" : "-bt");
    }
    std::string description() const override
    {
        std::ostringstream s;
        s << "pyrsgm " << pyrsgm::version() << ", 3 paths, "
          << (cost == pyrsgm::CostMode::Census ? "5x5 Census/Hamming P1=8 P2=32"
                                               : "BT+gradient 5x5 SAD P1=300 P2=2400")
          << ", uniq 10%, LR diff 1, median";
        if (levels > 0) s << ", pyramid " << levels << " level(s): coarse 1/" << (1 << levels)
                          << " + 16-label refine";
        else s << ", single scale";
        return s.str();
    }
    void setup(int width, int height, int D) override
    {
        pyrsgm::Params p;
        p.num_disparities = D;
        p.pyramid_levels = levels;
        p.cost_mode = cost;
        p.block_size = 5;
        p.p1 = cost == pyrsgm::CostMode::Census ? 8 : 300;
        p.p2 = cost == pyrsgm::CostMode::Census ? 32 : 2400;
        p.uniqueness_ratio = 10;
        p.disp12_max_diff = 1;
        p.lr_check = true;
        p.median_filter = true;
        p.pyramid_coarse_disparities = 128;
        p.pyramid_min_confidence = 0.05f;
        sgm.reset(new pyrsgm::StereoMatcher(width, height, p));
        used = sgm->params();
    }
    void run(const cv::cuda::GpuMat& left, const cv::cuda::GpuMat& right,
             cv::cuda::GpuMat& disp16) override
    {
        sgm->compute(left.ptr<uint8_t>(), left.step, right.ptr<uint8_t>(), right.step,
                     disp16.ptr<int16_t>(), disp16.step, nullptr, 0, nullptr, 0, nullptr);
    }
    void teardown() override { sgm.reset(); }
};

#ifdef PYRSGM_HAVE_CV_CUDASTEREO
struct CvCudaSgmAlgo : Algorithm {
    int mode;
    cv::Ptr<cv::cuda::StereoSGM> sgm;
    explicit CvCudaSgmAlgo(int m) : mode(m) {}
    std::string name() const override
    {
        return mode == cv::cuda::StereoSGM::MODE_HH ? "opencv-cuda-sgm-hh" : "opencv-cuda-sgm-hh4";
    }
    std::string description() const override
    {
        return std::string("cv::cuda::StereoSGM (OpenCV ") + CV_VERSION + ", libSGM port), " +
               (mode == cv::cuda::StereoSGM::MODE_HH ? "8 paths" : "4 paths") +
               ", 9x7 census, P1=10 P2=120, uniq 5%, LR diff 1, median, subpixel";
    }
    void setup(int, int, int D) override
    {
        sgm = cv::cuda::createStereoSGM(0, D, 10, 120, 5, mode);
        sgm->setDisp12MaxDiff(1);
    }
    void run(const cv::cuda::GpuMat& left, const cv::cuda::GpuMat& right,
             cv::cuda::GpuMat& disp16) override
    {
        sgm->compute(left, right, disp16, cv::cuda::Stream::Null());
    }
    void teardown() override { sgm.release(); }
};
#endif

#ifdef PYRSGM_HAVE_LIBSGM
struct LibSgmAlgo : Algorithm {
    sgm::PathType path;
    std::unique_ptr<sgm::StereoSGM> ssgm;
    int width = 0, height = 0, D = 0;
    size_t src_pitch = 0, dst_pitch = 0;
    explicit LibSgmAlgo(sgm::PathType p) : path(p) {}
    std::string name() const override
    {
        return path == sgm::PathType::SCAN_8PATH ? "libsgm-8path" : "libsgm-4path";
    }
    std::string description() const override
    {
        return std::string("fixstars libSGM 3.1, ") +
               (path == sgm::PathType::SCAN_8PATH ? "8 paths" : "4 paths") +
               ", symmetric 9x7 census, P1=10 P2=120, uniq 0.95, LR diff 1, median, subpixel";
    }
    void setup(int w, int h, int d) override
    {
        width = w; height = h; D = d;
        ssgm.reset();  // (re)created lazily on first run when pitches are known
    }
    void run(const cv::cuda::GpuMat& left, const cv::cuda::GpuMat& right,
             cv::cuda::GpuMat& disp16) override
    {
        if (!ssgm || src_pitch != left.step || dst_pitch != disp16.step / 2) {
            if (left.step != right.step) throw std::runtime_error("libSGM needs equal input pitches");
            src_pitch = left.step;
            dst_pitch = disp16.step / 2;
            sgm::StereoSGM::Parameters p(10, 120, 0.95f, /*subpixel=*/true, path, 0, 1,
                                          sgm::CensusType::SYMMETRIC_CENSUS_9x7);
            ssgm.reset(new sgm::StereoSGM(width, height, D, 8, 16, (int)src_pitch, (int)dst_pitch,
                                          sgm::EXECUTE_INOUT_CUDA2CUDA, p));
        }
        ssgm->execute(left.ptr<uint8_t>(), right.ptr<uint8_t>(), disp16.ptr<uint16_t>());
    }
    void teardown() override { ssgm.reset(); }
};
#endif

struct CvCpuSgbmAlgo : Algorithm {
    cv::Ptr<cv::StereoSGBM> sgbm;
    cv::Mat l, r, d;
    std::string name() const override { return "opencv-cpu-sgbm-3way"; }
    std::string description() const override
    {
        return std::string("cv::StereoSGBM MODE_SGBM_3WAY (OpenCV ") + CV_VERSION +
               ", CPU reference of the BT path), 3 paths, BT 5x5, P1=300 P2=2400, uniq 10%, LR diff 1, no speckle";
    }
    bool onGpu() const override { return false; }
    void setup(int, int, int D) override
    {
        sgbm = cv::StereoSGBM::create(0, D, 5, 300, 2400, 1, 63, 10, 0, 0, cv::StereoSGBM::MODE_SGBM_3WAY);
    }
    void run(const cv::cuda::GpuMat& left, const cv::cuda::GpuMat& right,
             cv::cuda::GpuMat& disp16) override
    {
        // Download once; timing wraps compute only (see benchmark loop).
        if (l.empty()) { left.download(l); right.download(r); }
        sgbm->compute(l, r, d);
        disp16.upload(d);
    }
    void teardown() override { sgbm.release(); l.release(); r.release(); d.release(); }
};

std::vector<std::unique_ptr<Algorithm>> makeAlgorithms()
{
    std::vector<std::unique_ptr<Algorithm>> v;
    v.emplace_back(new PyrsgmAlgo(0, pyrsgm::CostMode::BirchfieldTomasi));
    v.emplace_back(new PyrsgmAlgo(1, pyrsgm::CostMode::BirchfieldTomasi));
    v.emplace_back(new PyrsgmAlgo(2, pyrsgm::CostMode::BirchfieldTomasi));
    v.emplace_back(new PyrsgmAlgo(3, pyrsgm::CostMode::BirchfieldTomasi));
    v.emplace_back(new PyrsgmAlgo(0, pyrsgm::CostMode::Census));
    v.emplace_back(new PyrsgmAlgo(2, pyrsgm::CostMode::Census));
#ifdef PYRSGM_HAVE_CV_CUDASTEREO
    v.emplace_back(new CvCudaSgmAlgo(cv::cuda::StereoSGM::MODE_HH));
    v.emplace_back(new CvCudaSgmAlgo(cv::cuda::StereoSGM::MODE_HH4));
#endif
#ifdef PYRSGM_HAVE_LIBSGM
    v.emplace_back(new LibSgmAlgo(sgm::PathType::SCAN_8PATH));
    v.emplace_back(new LibSgmAlgo(sgm::PathType::SCAN_4PATH));
#endif
    v.emplace_back(new CvCpuSgbmAlgo());
    return v;
}

// ---------------------------------------------------------------------------
// Scenes
// ---------------------------------------------------------------------------

struct Scene {
    std::string name;
    cv::Mat left, right;   // CV_8UC1
    cv::Mat gt;            // CV_32FC1 or empty
    cv::Mat mask;          // CV_8UC1 or empty
    int ndisp = 0;         // scene disparity range (from calib.txt / --ndisp)
};

int roundDisparity(int ndisp)
{
    if (ndisp <= 64) return 64;
    if (ndisp <= 128) return 128;
    return 256;
}

cv::Mat loadGray(const std::string& path)
{
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) throw std::runtime_error("cannot read image " + path);
    return img;
}

std::vector<Scene> loadScenes(const Options& o)
{
    std::vector<Scene> scenes;
    if (!o.left.empty()) {
        Scene s;
        s.name = "pair";
        s.left = loadGray(o.left);
        s.right = loadGray(o.right);
        if (!o.gt.empty()) s.gt = mb::readPFM(o.gt);
        if (!o.mask.empty()) s.mask = loadGray(o.mask);
        s.ndisp = o.ndisp_override > 0 ? o.ndisp_override : 128;
        scenes.push_back(std::move(s));
        return scenes;
    }
    if (o.dataset.empty()) throw std::runtime_error("either --dataset or --left/--right is required");
    std::vector<std::string> names = o.scenes;
    if (names.empty()) {
        for (const char* n : kDefaultScenes) {
            if (fs_compat::exists(fs_compat::join(fs_compat::join(o.dataset, n), "im0.png"))) names.push_back(n);
        }
    }
    for (const std::string& n : names) {
        const std::string dir = fs_compat::join(o.dataset, n);
        Scene s;
        s.name = n;
        s.left = loadGray(fs_compat::join(dir, "im0.png"));
        s.right = loadGray(fs_compat::join(dir, "im1.png"));
        if (fs_compat::exists(fs_compat::join(dir, "disp0GT.pfm"))) s.gt = mb::readPFM(fs_compat::join(dir, "disp0GT.pfm"));
        if (fs_compat::exists(fs_compat::join(dir, "mask0nocc.png"))) s.mask = loadGray(fs_compat::join(dir, "mask0nocc.png"));
        s.ndisp = o.ndisp_override > 0 ? o.ndisp_override : mb::readNdisp(fs_compat::join(dir, "calib.txt"));
        if (s.ndisp <= 0) s.ndisp = 128;
        scenes.push_back(std::move(s));
    }
    if (scenes.empty()) throw std::runtime_error("no scenes found in " + o.dataset);
    return scenes;
}

// ---------------------------------------------------------------------------
// Benchmark loop
// ---------------------------------------------------------------------------

double median(std::vector<double> v)
{
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

struct Result {
    std::string scene, algo;
    int width = 0, height = 0, D = 0;
    std::vector<double> gpu_ms, wall_ms;   // pooled samples over rounds
    double gpu_ms_median = 0, gpu_ms_mean = 0, gpu_ms_min = 0, wall_ms_median = 0;
    bool has_metrics = false;
    mb::Metrics m;
    bool failed = false;
    std::string error;

    void finalize()
    {
        if (gpu_ms.empty()) return;
        gpu_ms_median = median(gpu_ms);
        gpu_ms_mean = std::accumulate(gpu_ms.begin(), gpu_ms.end(), 0.0) / gpu_ms.size();
        gpu_ms_min = *std::min_element(gpu_ms.begin(), gpu_ms.end());
        wall_ms_median = median(wall_ms);
    }
};

cv::Mat toFloatDisparity(const cv::Mat& disp16)
{
    // Fixed-point x16 int16; values <= 0 are invalid for every library here
    // (pyrsgm/OpenCV: (minDisp-1)*16 = -16; libSGM: 0xFFF0 reinterpreted = -16).
    cv::Mat out(disp16.size(), CV_32FC1);
    for (int y = 0; y < disp16.rows; ++y) {
        const int16_t* s = disp16.ptr<int16_t>(y);
        float* d = out.ptr<float>(y);
        for (int x = 0; x < disp16.cols; ++x) d[x] = s[x] > 0 ? s[x] / 16.0f : 0.0f;
    }
    return out;
}

cv::Mat colorize(const cv::Mat& disp, float max_disp)
{
    cv::Mat u8;
    disp.convertTo(u8, CV_8UC1, 255.0 / max_disp);
    cv::Mat color;
    cv::applyColorMap(u8, color, cv::COLORMAP_TURBO);
    color.setTo(cv::Scalar(0, 0, 0), disp <= 0.0f);
    return color;
}

/// Runs one (algorithm, scene) measurement round and appends the samples to r.
/// Accuracy metrics and visualisations are produced on the first round only.
void benchmarkOne(Algorithm& algo, const Scene& scene, const Options& o, Result& r, bool first_round)
{
    r.scene = scene.name;
    r.algo = algo.name();
    r.width = scene.left.cols;
    r.height = scene.left.rows;
    r.D = roundDisparity(scene.ndisp);
    if (r.failed) return;

    cv::cuda::GpuMat d_left, d_right, d_disp(scene.left.size(), CV_16SC1);
    d_left.upload(scene.left);
    d_right.upload(scene.right);

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    try {
        algo.setup(r.width, r.height, r.D);
        // Warm-up: at least o.warmup iterations and at least o.warmup_ms of
        // wall time, so that power-managed GPUs settle at the clock they can
        // sustain for this workload before anything is recorded.
        const auto warm_start = std::chrono::steady_clock::now();
        int warm = 0;
        for (;;) {
            algo.run(d_left, d_right, d_disp);
            cudaDeviceSynchronize();
            ++warm;
            const double elapsed =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - warm_start).count();
            if (warm >= o.warmup && elapsed >= o.warmup_ms) break;
        }
        const cudaError_t warm_err = cudaGetLastError();
        if (warm_err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(warm_err));

        for (int i = 0; i < o.iters; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            cudaEventRecord(ev0, 0);
            algo.run(d_left, d_right, d_disp);
            cudaEventRecord(ev1, 0);
            cudaEventSynchronize(ev1);
            const auto t1 = std::chrono::steady_clock::now();
            const cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
            float ms = 0.0f;
            cudaEventElapsedTime(&ms, ev0, ev1);
            const double wall = std::chrono::duration<double, std::milli>(t1 - t0).count();
            r.gpu_ms.push_back(algo.onGpu() ? (double)ms : wall);  // CPU reference: host time
            r.wall_ms.push_back(wall);
            if (o.verbose) std::cout << "    iter " << i << ": " << ms << " ms\n";
        }

        if (first_round) {
            cv::Mat disp16;
            d_disp.download(disp16);
            const cv::Mat disp = toFloatDisparity(disp16);
            if (!scene.gt.empty()) {
                r.has_metrics = true;
                r.m = mb::evaluate(disp, scene.gt, scene.mask, (float)scene.ndisp);
            }
            if (!o.save_dir.empty()) {
                const std::string file = fs_compat::join(o.save_dir, scene.name + "_" + algo.name() + ".png");
                cv::imwrite(file, colorize(disp, (float)r.D));
            }
        }
    } catch (const std::exception& e) {
        r.failed = true;
        r.error = e.what();
        cudaGetLastError();
    }
    algo.teardown();
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
}

std::string gpuName()
{
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return "unknown GPU";
    std::ostringstream s;
    s << prop.name << " (sm_" << prop.major << prop.minor << ")";
    return s.str();
}

std::string fmt(double v, int prec = 2)
{
    std::ostringstream s;
    s << std::fixed << std::setprecision(prec) << v;
    return s.str();
}

void writeCsv(const std::string& path, const std::vector<Result>& results)
{
    std::ofstream f(path);
    f << "scene,algo,width,height,D,gpu_ms_median,gpu_ms_mean,gpu_ms_min,wall_ms_median,"
         "density_pct,bad0.5_pct,bad1_pct,bad2_pct,bad4_pct,avg_err_px,rms_err_px,bad1_all_pct,bad2_all_pct,error\n";
    for (const Result& r : results) {
        f << r.scene << ',' << r.algo << ',' << r.width << ',' << r.height << ',' << r.D << ','
          << fmt(r.gpu_ms_median, 3) << ',' << fmt(r.gpu_ms_mean, 3) << ',' << fmt(r.gpu_ms_min, 3) << ','
          << fmt(r.wall_ms_median, 3) << ',';
        if (r.has_metrics) {
            f << fmt(r.m.density) << ',' << fmt(r.m.bad05) << ',' << fmt(r.m.bad1) << ',' << fmt(r.m.bad2) << ','
              << fmt(r.m.bad4) << ',' << fmt(r.m.avg_err, 3) << ',' << fmt(r.m.rms_err, 3) << ','
              << fmt(r.m.bad1_all) << ',' << fmt(r.m.bad2_all);
        } else {
            f << ",,,,,,,,";
        }
        f << ',' << (r.failed ? r.error : "") << '\n';
    }
}

struct Summary {
    std::string algo, description;
    int scenes = 0;
    double gpu_ms = 0, wall_ms = 0, fps = 0;
    double density = 0, bad1 = 0, bad2 = 0, avg_err = 0, bad1_all = 0, bad2_all = 0;
    bool has_metrics = false;
};

std::vector<Summary> summarize(const std::vector<Result>& results,
                               const std::vector<std::unique_ptr<Algorithm>>& algos)
{
    std::vector<Summary> out;
    for (const auto& a : algos) {
        Summary s;
        s.algo = a->name();
        s.description = a->description();
        for (const Result& r : results) {
            if (r.algo != s.algo || r.failed) continue;
            ++s.scenes;
            s.gpu_ms += r.gpu_ms_median;
            s.wall_ms += r.wall_ms_median;
            if (r.has_metrics) {
                s.has_metrics = true;
                s.density += r.m.density;
                s.bad1 += r.m.bad1;
                s.bad2 += r.m.bad2;
                s.avg_err += r.m.avg_err;
                s.bad1_all += r.m.bad1_all;
                s.bad2_all += r.m.bad2_all;
            }
        }
        if (s.scenes) {
            s.gpu_ms /= s.scenes;
            s.wall_ms /= s.scenes;
            s.fps = s.gpu_ms > 0 ? 1000.0 / s.gpu_ms : 0.0;
            s.density /= s.scenes;
            s.bad1 /= s.scenes;
            s.bad2 /= s.scenes;
            s.avg_err /= s.scenes;
            s.bad1_all /= s.scenes;
            s.bad2_all /= s.scenes;
        }
        out.push_back(s);
    }
    return out;
}

void writeMarkdown(std::ostream& f, const std::vector<Result>& results,
                   const std::vector<std::unique_ptr<Algorithm>>& algos,
                   const std::vector<Scene>& scenes, const Options& o)
{
    const std::vector<Summary> sums = summarize(results, algos);
    const bool metrics = std::any_of(sums.begin(), sums.end(), [](const Summary& s) { return s.has_metrics; });
    f << "GPU: " << gpuName() << "  \n";
    f << "CUDA runtime: ";
    int rt = 0;
    cudaRuntimeGetVersion(&rt);
    f << rt / 1000 << "." << (rt % 1000) / 10 << ", OpenCV " << CV_VERSION << "  \n";
    f << "Scenes: " << scenes.size() << ", rounds: " << o.rounds << " x " << o.iters
      << " timed iterations (median of pooled samples), warm-up: >= " << o.warmup
      << " iterations and >= " << fmt(o.warmup_ms, 0) << " ms per algorithm\n\n";

    f << "| algorithm | paths | avg GPU ms | FPS | wall ms";
    if (metrics) f << " | density % | bad1 % (valid) | bad2 % (valid) | avg err px | bad1 % (all) | bad2 % (all)";
    f << " |\n|---|---|---:|---:|---:";
    if (metrics) f << "|---:|---:|---:|---:|---:|---:";
    f << "|\n";
    for (const Summary& s : sums) {
        if (!s.scenes) continue;
        std::string paths = "3";
        if (s.description.find("8 paths") != std::string::npos) paths = "8";
        else if (s.description.find("4 paths") != std::string::npos) paths = "4";
        f << "| " << s.algo << " | " << paths << " | " << fmt(s.gpu_ms) << " | " << fmt(s.fps, 1)
          << " | " << fmt(s.wall_ms);
        if (metrics) {
            if (s.has_metrics) {
                f << " | " << fmt(s.density, 1) << " | " << fmt(s.bad1) << " | " << fmt(s.bad2) << " | "
                  << fmt(s.avg_err) << " | " << fmt(s.bad1_all) << " | " << fmt(s.bad2_all);
            } else {
                f << " | - | - | - | - | - | -";
            }
        }
        f << " |\n";
    }
    f << "\n";
    if (metrics) {
        f << "`bad N % (valid)` is the share of evaluated non-occluded GT pixels with a valid estimate whose error exceeds N px; "
             "`bad N % (all)` additionally counts invalid (hole) pixels as bad, which is the MiddEval3 convention for sparse outputs.  "
             "Time columns are the per-scene medians averaged over scenes; the CPU reference reports host time.\n\n";
    }

    // Per-scene time table.
    f << "### Per-scene GPU time (ms, median)\n\n| scene | size | D |";
    for (const auto& a : algos) f << " " << a->name() << " |";
    f << "\n|---|---|---:|";
    for (size_t i = 0; i < algos.size(); ++i) f << "---:|";
    f << "\n";
    for (const Scene& sc : scenes) {
        const Result* any = nullptr;
        for (const Result& r : results) if (r.scene == sc.name) { any = &r; break; }
        if (!any) continue;
        f << "| " << sc.name << " | " << any->width << "x" << any->height << " | " << any->D << " |";
        for (const auto& a : algos) {
            const Result* found = nullptr;
            for (const Result& r : results) if (r.scene == sc.name && r.algo == a->name()) { found = &r; break; }
            if (!found) f << " - |";
            else if (found->failed) f << " fail |";
            else f << " " << fmt(found->gpu_ms_median) << " |";
        }
        f << "\n";
    }
    if (metrics) {
        f << "\n### Per-scene bad1 % (all, invalid counted as bad)\n\n| scene |";
        for (const auto& a : algos) f << " " << a->name() << " |";
        f << "\n|---|";
        for (size_t i = 0; i < algos.size(); ++i) f << "---:|";
        f << "\n";
        for (const Scene& sc : scenes) {
            bool row = false;
            std::ostringstream line;
            line << "| " << sc.name << " |";
            for (const auto& a : algos) {
                const Result* found = nullptr;
                for (const Result& r : results) if (r.scene == sc.name && r.algo == a->name()) { found = &r; break; }
                if (!found || !found->has_metrics) line << " - |";
                else if (found->failed) line << " fail |";
                else { line << " " << fmt(found->m.bad1_all, 1) << " |"; row = true; }
            }
            if (row) f << line.str() << "\n";
        }
    }
    f << "\n### Algorithm settings\n\n";
    for (const Summary& s : sums) f << "- `" << s.algo << "`: " << s.description << "\n";
    for (const Result& r : results) {
        if (r.failed) f << "- `" << r.algo << "` failed on " << r.scene << ": " << r.error << "\n";
    }
}

} // namespace

int main(int argc, char** argv)
{
    Options o = parseArgs(argc, argv);
    std::vector<std::unique_ptr<Algorithm>> algos = makeAlgorithms();
    if (o.list) {
        for (const auto& a : algos) std::cout << a->name() << "\t" << a->description() << "\n";
        return 0;
    }
    if (!o.algos.empty()) {
        std::vector<std::unique_ptr<Algorithm>> filtered;
        for (auto& a : algos) {
            if (std::find(o.algos.begin(), o.algos.end(), a->name()) != o.algos.end()) filtered.push_back(std::move(a));
        }
        for (const std::string& n : o.algos) {
            bool found = false;
            for (const auto& a : filtered) found |= a->name() == n;
            if (!found) std::cerr << "warning: unknown/unavailable algorithm " << n << "\n";
        }
        algos = std::move(filtered);
    }
    if (algos.empty()) {
        std::cerr << "no algorithms selected\n";
        return 2;
    }

    std::vector<Scene> scenes;
    try {
        scenes = loadScenes(o);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }

    if (!o.sizes.empty()) {
        // Timing-only mode: resize the first scene to each requested size.
        const Scene base = scenes.front();
        scenes.clear();
        for (const cv::Size& sz : o.sizes) {
            Scene s;
            s.name = std::to_string(sz.width) + "x" + std::to_string(sz.height);
            cv::resize(base.left, s.left, sz, 0, 0, cv::INTER_AREA);
            cv::resize(base.right, s.right, sz, 0, 0, cv::INTER_AREA);
            s.ndisp = o.ndisp_override > 0 ? o.ndisp_override : 128;
            scenes.push_back(std::move(s));
        }
    }

    std::cout << "GPU: " << gpuName() << "\n";
    // results[scene * algos.size() + algo]
    std::vector<Result> results(scenes.size() * algos.size());
    for (int round = 0; round < o.rounds; ++round) {
        if (o.rounds > 1) std::cout << "---- round " << round + 1 << "/" << o.rounds << "\n";
        for (size_t si = 0; si < scenes.size(); ++si) {
            const Scene& sc = scenes[si];
            std::cout << "== " << sc.name << " " << sc.left.cols << "x" << sc.left.rows
                      << " ndisp=" << sc.ndisp << " D=" << roundDisparity(sc.ndisp) << "\n";
            for (size_t ai = 0; ai < algos.size(); ++ai) {
                Result& r = results[si * algos.size() + ai];
                benchmarkOne(*algos[ai], sc, o, r, round == 0);
                r.finalize();
                std::cout << "  " << std::left << std::setw(24) << r.algo << std::right;
                if (r.failed) {
                    std::cout << " FAILED: " << r.error << "\n";
                } else {
                    std::cout << " gpu " << std::setw(8) << fmt(r.gpu_ms_median, 3) << " ms"
                              << "  wall " << std::setw(8) << fmt(r.wall_ms_median, 3) << " ms";
                    if (r.has_metrics) {
                        std::cout << "  density " << fmt(r.m.density, 1) << "%"
                                  << "  bad1 " << fmt(r.m.bad1) << "%"
                                  << "  avgErr " << fmt(r.m.avg_err) << "px"
                                  << "  bad1(all) " << fmt(r.m.bad1_all) << "%";
                    }
                    std::cout << "\n";
                }
            }
        }
    }

    if (!o.csv.empty()) writeCsv(o.csv, results);
    std::cout << "\n";
    writeMarkdown(std::cout, results, algos, scenes, o);
    if (!o.md.empty()) {
        std::ofstream f(o.md);
        writeMarkdown(f, results, algos, scenes, o);
    }
    return 0;
}
