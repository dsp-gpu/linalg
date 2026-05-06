<!-- type:meta_targets repo:linalg source:linalg/CMakeLists.txt -->

# Build Targets — linalg

## Targets

- **`DspLinalg`** (library)
  - PUBLIC: `DspCore::DspCore`, `roc::rocblas`, `roc::rocsolver`
  - PRIVATE: `${HIPRTC_LIB}`

## BUILD-флаги (option)

- `DSP_LINALG_BUILD_TESTS` (default `ON`) — Build tests
- `DSP_LINALG_BUILD_PYTHON` (default `OFF`) — Build Python bindings

## Зависимости от DSP репо

- `core` — через `fetch_dsp_core()`

## External find_package

- `hip` (required)
- `rocblas` (required)
- `rocsolver` (required)
- `hiprtc` (required)
