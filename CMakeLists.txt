cmake_minimum_required(VERSION 3.25)
project(DspLinalg VERSION 0.1.0
  DESCRIPTION "GPU linear algebra: vector operations, Capon/MVDR"
  LANGUAGES CXX HIP)

# ── Опции ──────────────────────────────────────────────────────────
option(DSP_LINALG_BUILD_TESTS  "Build tests"            ON)
option(DSP_LINALG_BUILD_PYTHON "Build Python bindings"  OFF)

# ── Зависимости ────────────────────────────────────────────────────
find_package(hip      REQUIRED)   # lowercase — Linux case-sensitive!
find_package(rocblas  REQUIRED)
find_package(rocsolver REQUIRED)
find_package(hiprtc   REQUIRED)

include(cmake/fetch_deps.cmake)
fetch_dsp_core()

# ── Библиотека ─────────────────────────────────────────────────────
add_library(DspLinalg STATIC)
add_library(DspLinalg::DspLinalg ALIAS DspLinalg)

target_sources(DspLinalg PRIVATE
  # vector_algebra
  src/vector_algebra/src/matrix_ops_rocm.cpp
  src/vector_algebra/src/cholesky_inverter_rocm.cpp
  src/vector_algebra/src/diagonal_load_regularizer.cpp
  src/vector_algebra/src/symmetrize_gpu_rocm.cpp
  # capon
  src/capon/src/capon_processor.cpp
)

target_include_directories(DspLinalg
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    src/vector_algebra/include
    src/capon/include
    kernels/
    src
)
target_link_libraries(DspLinalg
  PUBLIC  DspCore::DspCore roc::rocblas roc::rocsolver
  PRIVATE hiprtc
)
target_compile_features(DspLinalg PUBLIC cxx_std_17)

# ── Install + Export ────────────────────────────────────────────────
include(GNUInstallDirs)
install(TARGETS DspLinalg EXPORT DspLinalgTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/DspLinalgConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/DspLinalgConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DspLinalg
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/DspLinalgConfigVersion.cmake"
  VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion
)
install(EXPORT DspLinalgTargets NAMESPACE DspLinalg::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DspLinalg
)
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/DspLinalgConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/DspLinalgConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DspLinalg
)

if(DSP_LINALG_BUILD_TESTS)
  add_subdirectory(tests)
endif()
if(DSP_LINALG_BUILD_PYTHON)
  add_subdirectory(python)
endif()
