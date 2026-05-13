---
schema_version: 1
repo: linalg
arch_level: c4
tags:
  - "#level:c4"
  - "#repo:linalg"
  - "#layer:compute"
  - "#pattern:Pipeline:CaponProcessor"
description: "C4 Code — реальные классы с паттернами GoF/SOLID для репо linalg."
---

# C4 Code — `linalg`

## Классы с паттернами проектирования

| Класс | Паттерн | Brief |
|-------|---------|-------|
| `CaponProcessor` | **Pipeline** |  |

## HIP-ядра (`kernels/rocm/`)

*Репо без HIP-ядер.*

## Все key_classes (FQN список)

- `dsp::linalg::MatrixOpsROCm` (13 методов)
- `dsp::linalg::CaponProcessor` (28 методов)
- `drv_gpu_lib::GpuContext` (14 методов)
- `dsp::linalg::CholeskyInverterROCm` (39 методов)
- `dsp::linalg::DiagonalLoadRegularizer` (8 методов)
- `dsp::linalg::AdaptBeamformOp` (2 методов)
- `dsp::linalg::ComputeWeightsOp` (2 методов)
- `dsp::linalg::CholeskyResult` (13 методов)
- `dsp::linalg::CaponInvertOp` (6 методов)
- `dsp::linalg::CovarianceMatrixOp` (2 методов)
