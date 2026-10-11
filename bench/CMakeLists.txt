# Benchmark: pyrsgm vs OpenCV cuda::StereoSGM vs fixstars libSGM (+ OpenCV CPU SGBM).
# Requires an OpenCV build with CUDA support (core GpuMat); cudastereo and
# libSGM are optional and detected automatically.

find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs calib3d)
if(NOT OpenCV_FOUND)
    message(WARNING "pyrsgm: OpenCV not found, benchmark disabled "
                    "(set OpenCV_DIR or -DPYRSGM_BUILD_BENCH=OFF)")
    return()
endif()
message(STATUS "pyrsgm: benchmark OpenCV ${OpenCV_VERSION} (${OpenCV_DIR})")

set(_bench_cv_targets opencv_core opencv_imgproc opencv_imgcodecs opencv_calib3d)
set(_bench_defs "")

if(TARGET opencv_cudastereo)
    list(APPEND _bench_cv_targets opencv_cudastereo)
    list(APPEND _bench_defs PYRSGM_HAVE_CV_CUDASTEREO)
    message(STATUS "pyrsgm: benchmark includes OpenCV cuda::StereoSGM")
else()
    message(STATUS "pyrsgm: OpenCV cudastereo module not found, cuda::StereoSGM skipped")
endif()
if(TARGET opencv_cudaarithm)
    list(APPEND _bench_cv_targets opencv_cudaarithm)
endif()

# fixstars libSGM (https://github.com/fixstars/libSGM), optional.
set(_libsgm_libs "")
if(PYRSGM_BENCH_WITH_LIBSGM)
    find_path(LIBSGM_INCLUDE_DIR libsgm.h
        HINTS ${LIBSGM_ROOT} $ENV{LIBSGM_ROOT}
              ${CMAKE_SOURCE_DIR}/third_party/libSGM/install
        PATH_SUFFIXES include)
    find_library(LIBSGM_LIBRARY NAMES sgm libsgm
        HINTS ${LIBSGM_ROOT} $ENV{LIBSGM_ROOT}
              ${CMAKE_SOURCE_DIR}/third_party/libSGM/install
        PATH_SUFFIXES lib lib64)
    if(LIBSGM_INCLUDE_DIR AND LIBSGM_LIBRARY)
        list(APPEND _bench_defs PYRSGM_HAVE_LIBSGM)
        set(_libsgm_libs ${LIBSGM_LIBRARY})
        message(STATUS "pyrsgm: benchmark includes libSGM (${LIBSGM_LIBRARY})")
    else()
        message(STATUS "pyrsgm: libSGM not found (run scripts/build_libsgm.sh or set LIBSGM_ROOT); skipped")
    endif()
endif()

add_executable(pyrsgm_bench pyrsgm_bench.cpp)
target_link_libraries(pyrsgm_bench PRIVATE pyrsgm ${_bench_cv_targets} ${_libsgm_libs} CUDA::cudart)
target_compile_definitions(pyrsgm_bench PRIVATE ${_bench_defs})
if(LIBSGM_INCLUDE_DIR AND PYRSGM_BENCH_WITH_LIBSGM)
    target_include_directories(pyrsgm_bench PRIVATE ${LIBSGM_INCLUDE_DIR})
endif()
target_compile_options(pyrsgm_bench PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall -Wextra>)
