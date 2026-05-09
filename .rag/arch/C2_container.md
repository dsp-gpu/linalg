---
schema_version: 1
repo: linalg
arch_level: c2
tags:
  - "#level:c2"
  - "#repo:linalg"
  - "#layer:compute"
  - "#namespace:capon"
  - "#namespace:vector_algebra"
  - "#namespace:vector_algebra::tests"
description: "C2 Container — namespace tree и зависимости репо linalg."
---

# C2 Container — `linalg` (layer=compute)

## Namespaces (top по числу классов)

- `capon`
- `vector_algebra`
- `vector_algebra::tests`
- `drv_gpu_lib`
- `test_capon_rocm_bench`

## Public modules (`include/linalg/`)

- `kernels/`
- `operations/`

## Зависимости (depends_on)

`core`

## Используется (used_by)

`radar`, `DSP`

## Top key_classes

| Class | Namespace | Methods | TestParams |
|-------|-----------|--------:|-----------:|
| `MatrixOpsROCm` | `vector_algebra` | 13 | 41 |
| `CaponProcessor` | `capon` | 28 | 19 |
| `GpuContext` | `drv_gpu_lib` | 14 | 7 |
| `CholeskyInverterROCm` | `vector_algebra` | 39 | 4 |
| `DiagonalLoadRegularizer` | `vector_algebra` | 8 | 4 |
