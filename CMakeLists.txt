cmake_minimum_required(VERSION 3.18)

# Pick CUDA architectures before project() so the CUDA language check uses them.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Try the local GPU first (keeps developer builds fast), otherwise a
    # portable set covering Turing .. Ada.  Pass -DCMAKE_CUDA_ARCHITECTURES
    # explicitly for release builds.
    execute_process(
        COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
        OUTPUT_VARIABLE _pyrsgm_cc
        RESULT_VARIABLE _pyrsgm_cc_res
        ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(_pyrsgm_cc_res EQUAL 0 AND _pyrsgm_cc MATCHES "^([0-9]+)\\.([0-9]+)")
        # First GPU only; "12.0" -> "120"
        string(REGEX REPLACE "\n.*" "" _pyrsgm_cc "${_pyrsgm_cc}")
        string(REPLACE "." "" _pyrsgm_cc "${_pyrsgm_cc}")
        set(CMAKE_CUDA_ARCHITECTURES "${_pyrsgm_cc}")
    else()
        set(CMAKE_CUDA_ARCHITECTURES "75;86;89")
    endif()
endif()

project(pyrsgm VERSION 0.1.0 LANGUAGES CXX CUDA)
message(STATUS "pyrsgm: CUDA architectures = ${CMAKE_CUDA_ARCHITECTURES}")

option(PYRSGM_BUILD_TESTS "Build unit / smoke tests" ON)
option(PYRSGM_BUILD_BENCH "Build the benchmark tool (needs OpenCV with CUDA)" ON)
option(PYRSGM_BUILD_EXAMPLES "Build examples (needs OpenCV)" ON)
option(PYRSGM_BENCH_WITH_LIBSGM "Link fixstars libSGM into the benchmark if found" ON)
option(PYRSGM_SHARED "Build pyrsgm as a shared library" OFF)
option(PYRSGM_WERROR "Treat warnings as errors" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

find_package(CUDAToolkit REQUIRED)

# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
set(PYRSGM_LIB_TYPE STATIC)
if(PYRSGM_SHARED)
    set(PYRSGM_LIB_TYPE SHARED)
endif()

add_library(pyrsgm ${PYRSGM_LIB_TYPE}
    src/sgm_kernels.cu
    src/pyrdown.cu
    src/stereo_matcher.cpp)
add_library(pyrsgm::pyrsgm ALIAS pyrsgm)

target_include_directories(pyrsgm
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
target_link_libraries(pyrsgm PUBLIC CUDA::cudart)
target_compile_features(pyrsgm PUBLIC cxx_std_17)
set_target_properties(pyrsgm PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION OFF
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR})
target_compile_options(pyrsgm PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall -Wextra>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -Xcompiler=-Wall,-Wextra>)
if(PYRSGM_WERROR)
    target_compile_options(pyrsgm PRIVATE
        $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Werror>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Werror>)
endif()

# ---------------------------------------------------------------------------
# Install / export
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS pyrsgm EXPORT pyrsgmTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/pyrsgm DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT pyrsgmTargets
    NAMESPACE pyrsgm::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pyrsgm)
configure_package_config_file(
    cmake/pyrsgmConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/pyrsgmConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pyrsgm)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/pyrsgmConfigVersion.cmake
    COMPATIBILITY SameMajorVersion)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/pyrsgmConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/pyrsgmConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/pyrsgm)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(PYRSGM_BUILD_TESTS)
    enable_testing()
    add_executable(pyrsgm_pyramid_smoke tests/pyramid_smoke.cpp)
    target_link_libraries(pyrsgm_pyramid_smoke PRIVATE pyrsgm)
    add_test(NAME pyramid_smoke COMMAND pyrsgm_pyramid_smoke)

    add_executable(pyrsgm_matcher_test tests/matcher_test.cpp)
    target_link_libraries(pyrsgm_matcher_test PRIVATE pyrsgm)
    add_test(NAME matcher_test COMMAND pyrsgm_matcher_test)

    set_tests_properties(pyramid_smoke matcher_test PROPERTIES SKIP_RETURN_CODE 77)
endif()

# ---------------------------------------------------------------------------
# Benchmark / examples (both need OpenCV; the benchmark needs OpenCV with CUDA)
# ---------------------------------------------------------------------------
if(PYRSGM_BUILD_BENCH)
    add_subdirectory(bench)
endif()

if(PYRSGM_BUILD_EXAMPLES)
    find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs)
    if(OpenCV_FOUND)
        add_executable(pyrsgm_stereo_pair examples/stereo_pair.cpp)
        target_link_libraries(pyrsgm_stereo_pair PRIVATE pyrsgm opencv_core opencv_imgproc opencv_imgcodecs)
    else()
        message(STATUS "pyrsgm: OpenCV not found, examples skipped")
    endif()
endif()
