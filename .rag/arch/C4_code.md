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

- `vector_algebra::MatrixOpsROCm` (13 методов)
- `capon::CaponProcessor` (28 методов)
- `drv_gpu_lib::GpuContext` (14 методов)
- `vector_algebra::CholeskyInverterROCm` (39 методов)
- `vector_algebra::DiagonalLoadRegularizer` (8 методов)
- `capon::AdaptBeamformOp` (2 методов)
- `capon::ComputeWeightsOp` (2 методов)
- `vector_algebra::CholeskyResult` (13 методов)
- `capon::CaponInvertOp` (6 методов)
- `capon::CovarianceMatrixOp` (2 методов)
