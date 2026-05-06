---
schema_version: 1
kind: use_case
id: capon_rocm
repo: linalg
title: "Адаптивное формирование луча на GPU с методом Capon"
synonyms:
  ru:
    - "адаптивное формирование луча на GPU"
    - "метод Capon для антенн"
    - "обработка сигналов с помехами на ROCm"
    - "Capon батчем на GPU"
    - "адаптация луча с регуляризацией"
    - "GPU обработка антенн"
    - "Capon для радиолокации"
    - "адаптивная обработка сигналов ROCm"
  en:
    - "adaptive beamforming with Capon on GPU"
    - "Capon method for antenna arrays"
    - "GPU signal processing with Capon"
    - "batch Capon on ROCm"
    - "regularized beamforming GPU"
    - "antenna array processing ROCm"
    - "Capon for radar applications"
    - "GPU adaptive filtering"
primary_class: (unknown)
primary_method: (unknown)
related_classes:
related_use_cases:
  - linalg__capon_opencl_to_rocm__usecase__v1
  - linalg__capon_benchmark_rocm__usecase__v1
  - linalg__capon_hip_opencl_to_rocm__usecase__v1
maturity: stable
language: cpp
tags: [linalg, capon, rocm, beamforming, gpu, antenna, regularization, signal_processing, adaptive, batch]
ai_generated: true
human_verified: false
operator: ai
updated_at: 2026-05-06
---

# Use-case: Адаптивное формирование луча на GPU с методом Capon

## Когда применять

Когда требуется обработка сигналов с антенной решеткой на GPU с регуляризацией и передачей данных между устройствами

## Решение

Класс — `(unknown)`, метод `(unknown)`.

```cpp
  drv_gpu_lib::ConsoleOutput::GetInstance().Start();
  TestPrint("=== test_capon_rocm ===");
  test_01_relief_noise_only();
  test_02_relief_with_interference();
  test_03_adaptive_beamform_dims();
  test_04_regularization();
  test_05_gpu_to_gpu();
  TestPrint("=== test_capon_rocm DONE ===");
```

## Граничные случаи

_Не определены (нет `@throws` в Doxygen primary_method)._

## Что делать дальше

- См. [linalg__capon_opencl_to_rocm__usecase__v1](./capon_opencl_to_rocm.md)
- См. [linalg__capon_benchmark_rocm__usecase__v1](./capon_benchmark_rocm.md)
- См. [linalg__capon_hip_opencl_to_rocm__usecase__v1](./capon_hip_opencl_to_rocm.md)

## Ссылки

- Источник кода: `E:/DSP-GPU/linalg/tests/test_capon_rocm.hpp:1`
