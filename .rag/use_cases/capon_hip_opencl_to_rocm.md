---
schema_version: 1
kind: use_case
id: capon_hip_opencl_to_rocm
repo: linalg
title: "Как выполнить обработку сигналов Capon с HIP и OpenCL на GPU"
synonyms:
  ru:
    - "Обработка сигналов Capon на GPU"
    - "Capon алгоритм HIP OpenCL"
    - "GPU обработка сигналов для антенн"
    - "Миграция OpenCL к ROCm для Capon"
    - "Обработка сигналов батчем на GPU"
    - "Capon с HIP и OpenCL"
    - "Радиолокационная обработка сигналов на GPU"
    - "Алгоритм Capon для антенных массивов"
  en:
    - "Capon processing with HIP and OpenCL"
    - "GPU signal processing for Capon"
    - "Antenna array signal processing"
    - "Migrate OpenCL to ROCm for Capon"
    - "Batch signal processing on GPU"
    - "Capon algorithm HIP OpenCL"
    - "Radar signal processing on GPU"
    - "Capon for phased array antennas"
primary_class: (unknown)
primary_method: (unknown)
related_classes:
related_use_cases:
  - core__zero_copy__usecase__v1
  - core__hybrid_backend__usecase__v1
  - linalg__capon_benchmark_rocm__usecase__v1
maturity: stable
language: cpp
tags: [linalg, hip, opencl, rocm, capon, batch, antenna_array, signal_processing, gpu_computing, radar]
ai_generated: true
human_verified: false
operator: ai
updated_at: 2026-05-06
---

# Use-case: Как выполнить обработку сигналов Capon с HIP и OpenCL на GPU

## Когда применять

Когда требуется миграция обработки сигналов с OpenCL на ROCm с использованием HIP для Capon-алгоритма

## Решение

Класс — `(unknown)`, метод `(unknown)`.

```cpp
  ConsoleOutput::GetInstance().Start();
  TestPrint("========================================================");
  TestPrint("  test_capon_hip_opencl_to_rocm");
  TestPrint("  HIP выделяет VRAM, OpenCL пишет данные, Capon считает");
  TestPrint("  NO clSVMAlloc  NO staging  NO ZeroCopyBridge");
  TestPrint("========================================================");

  test_01_detect_hip_svm();
  test_02_hip_opencl_capon_pipeline();
  test_03_hip_opencl_matches_direct();

  TestPrint("=== test_capon_hip_opencl_to_rocm DONE ===");
```

## Граничные случаи

_Не определены (нет `@throws` в Doxygen primary_method)._

## Что делать дальше

- См. [core__zero_copy__usecase__v1](./zero_copy.md)
- См. [core__hybrid_backend__usecase__v1](./hybrid_backend.md)
- См. [linalg__capon_benchmark_rocm__usecase__v1](./capon_benchmark_rocm.md)

## Ссылки

- Источник кода: `E:/DSP-GPU/linalg/tests/test_capon_hip_opencl_to_rocm.hpp:1`
