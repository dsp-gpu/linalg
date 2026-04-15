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
    # Стандарт AMD: #include <linalg/operations/...>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    src
)

# hiprtc: portable через find_library + CMAKE_PREFIX_PATH
find_library(HIPRTC_LIB NAMES hiprtc PATH_SUFFIXES lib lib64 REQUIRED)
message(STATUS "[DspLinalg] hiprtc: ${HIPRTC_LIB}")

target_link_libraries(DspLinalg
  PUBLIC  DspCore::DspCore roc::rocblas roc::rocsolver
  PRIVATE ${HIPRTC_LIB}
)
target_compile_features(DspLinalg PUBLIC cxx_std_17)

# ENABLE_ROCM для #if блоков
if(ENABLE_ROCM)
  target_compile_definitions(DspLinalg PUBLIC ENABLE_ROCM=1)
endif()

# ── Git Version ────────────────────────────────────────────────────
find_package(Git QUIET)
if(NOT GIT_EXECUTABLE)
    set(GIT_EXECUTABLE git)
endif()

add_custom_target(git_version_DspLinalg ALL
    COMMAND ${CMAKE_COMMAND}
        -D SRC_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -D BIN_DIR=${CMAKE_CURRENT_BINARY_DIR}
        -D GIT_EXECUTABLE=${GIT_EXECUTABLE}
        -D MODULE_PREFIX=DSPLINALG
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.cmake
    COMMENT "[DspLinalg] Checking git version..."
    VERBATIM
)
add_dependencies(DspLinalg git_version_DspLinalg)

# ── Install + Export ────────────────────────────────────────────────
include(GNUInstallDirs)
install(TARGETS DspLinalg EXPORT DspLinalgTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/generated/version.h"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/DspLinalg
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/generated/version.json"
  DESTINATION ${CMAKE_INSTALL_DATADIR}/DspLinalg
)

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
