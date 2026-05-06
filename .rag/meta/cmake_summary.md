<!-- type:meta_cmake_specific repo:linalg inherits:dsp_gpu__root__meta_cmake_common__v1 -->

# CMake Specific — linalg

```yaml
inherits: dsp_gpu__root__meta_cmake_common__v1
specific_only: true
target: DspLinalg
description: "GPU linear algebra: vector operations, Capon/MVDR"
adds_find_package: [hip, hiprtc, rocblas, rocsolver]
adds_links: [DspCore::DspCore, roc::rocblas, roc::rocsolver]
```

## Project

- **Target**: `DspLinalg`
- **Описание**: GPU linear algebra: vector operations, Capon/MVDR

## Уникальные find_package

```cmake
find_package(hip REQUIRED)
find_package(hiprtc REQUIRED)
find_package(rocblas REQUIRED)
find_package(rocsolver REQUIRED)
```

## Линкуемые библиотеки

```cmake
target_link_libraries(DspLinalg PUBLIC
  DspCore::DspCore
  roc::rocblas
  roc::rocsolver
)
```

## Исходники (5 файлов)

```cmake
target_sources(DspLinalg PRIVATE
  src/vector_algebra/src/matrix_ops_rocm.cpp
  src/vector_algebra/src/cholesky_inverter_rocm.cpp
  src/vector_algebra/src/diagonal_load_regularizer.cpp
  src/vector_algebra/src/symmetrize_gpu_rocm.cpp
  src/capon/src/capon_processor.cpp
)
```

## Прочие специфичные строки (17)

```cmake
DESCRIPTION "GPU linear algebra: vector operations, Capon/MVDR"
PRIVATE ${HIPRTC_LIB}
PUBLIC  <TARGET>::<TARGET> roc::rocblas roc::rocsolver
find_library(HIPRTC_LIB NAMES hiprtc PATH_SUFFIXES lib lib64 REQUIRED)
find_package(hip      REQUIRED)
find_package(hiprtc   REQUIRED)
find_package(rocblas  REQUIRED)
find_package(rocsolver REQUIRED)
message(STATUS "[<TARGET>] hiprtc: ${HIPRTC_LIB}")
src/capon/src/capon_processor.cpp
src/vector_algebra/src/cholesky_inverter_rocm.cpp
src/vector_algebra/src/diagonal_load_regularizer.cpp
src/vector_algebra/src/matrix_ops_rocm.cpp
src/vector_algebra/src/symmetrize_gpu_rocm.cpp
target_compile_definitions(<TARGET> PUBLIC ENABLE_ROCBLAS=1)
target_compile_definitions(<TARGET> PUBLIC ENABLE_ROCM=1 ENABLE_ROCBLAS=1)
target_link_libraries(<TARGET>
```

