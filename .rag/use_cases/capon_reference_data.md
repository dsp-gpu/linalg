---
schema_version: 1
kind: use_case
id: capon_reference_data
repo: linalg
title: "Вычисление спектра направленности антенн на GPU"
synonyms:
  ru:
    - "расчет направленности антенн"
    - "fft для антенных массивов"
    - "обработка сигналов на GPU"
    - "алгоритм Капона для массивов"
    - "вычисление спектра направленности"
    - "ускорение обработки сигналов"
    - "аналитика антенных массивов"
    - "параллельная обработка сигналов"
  en:
    - "capon algorithm implementation"
    - "beamforming on gpu"
    - "fft for antenna arrays"
    - "signal processing acceleration"
    - "directed reception calculation"
    - "parallel signal analysis"
    - "antenna array optimization"
    - "gpu-based spectral analysis"
primary_class: (unknown)
primary_method: (unknown)
related_classes:
related_use_cases:
  - linalg__capon_hip_opencl_to_rocm__usecase__v1
  - linalg__capon_opencl_to_rocm__usecase__v1
  - linalg__capon_benchmark_rocm__usecase__v1
maturity: stable
language: cpp
tags: [linalg, rocm, fft, antenna, beamforming, gpu, batch, parallel_processing, signal_processing, directed_reception]
ai_generated: true
human_verified: false
operator: ai
updated_at: 2026-05-06
---

# Use-case: Вычисление спектра направленности антенн на GPU

## Когда применять

Когда требуется ускорить обработку сигналов антенных массивов с использованием алгоритма Капона на GPU для больших объемов данных.

## Решение

Класс — `(unknown)`, метод `(unknown)`.

```cpp
  TestPrint("=== test_capon_reference_data ===");
  test_01_load_files();
  test_02_physical_relief_properties();
  test_03_cpu_vs_gpu_small_p();
  TestPrint("=== test_capon_reference_data DONE ===");
```

## Граничные случаи

_Не определены (нет `@throws` в Doxygen primary_method)._

## Что делать дальше

- См. [linalg__capon_hip_opencl_to_rocm__usecase__v1](./capon_hip_opencl_to_rocm.md)
- См. [linalg__capon_opencl_to_rocm__usecase__v1](./capon_opencl_to_rocm.md)
- См. [linalg__capon_benchmark_rocm__usecase__v1](./capon_benchmark_rocm.md)

## Ссылки

- Источник кода: `E:/DSP-GPU/linalg/tests/test_capon_reference_data.hpp:1`
