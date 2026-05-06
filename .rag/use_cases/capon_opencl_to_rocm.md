---
schema_version: 1
kind: use_case
id: capon_opencl_to_rocm
repo: linalg
title: "Перенос обработки сигналов Capon с OpenCL на ROCm"
synonyms:
  ru:
    - "Перенос Capon с OpenCL на ROCm"
    - "Ускорение обработки сигналов Capon на ROCm"
    - "Интеграция OpenCL и ROCm для Capon"
    - "Обработка сигналов Capon через ROCm"
    - "Перенос данных между OpenCL и ROCm для Capon"
    - "Оптимизация Capon на ROCm с OpenCL"
    - "Совместимость OpenCL/ROCm для обработки сигналов"
    - "Обработка сигналов Capon с нулекопи"
  en:
    - "Transfer capon from opencl to rocm"
    - "Accelerate capon processing on rocm"
    - "Opencl to rocm interop for capon"
    - "Capon signal processing via rocm"
    - "Zero copy data transfer between opencl and rocm"
    - "Optimize capon on rocm with opencl"
    - "Opencl rocm compatibility for signal processing"
    - "Capon processing with zero copy"
primary_class: (unknown)
primary_method: (unknown)
related_classes:
related_use_cases:
  - linalg__capon_hip_opencl_to_rocm__usecase__v1
  - core__hybrid_backend__usecase__v1
  - core__zero_copy__usecase__v1
maturity: stable
language: cpp
tags: [linalg, capon, rocm, opencl, gpu, signal_processing, batch, interop, zero_copy, svm]
ai_generated: true
human_verified: false
operator: ai
updated_at: 2026-05-06
---

# Use-case: Перенос обработки сигналов Capon с OpenCL на ROCm

## Когда применять

Когда требуется ускорить обработку сигналов Capon на GPU с использованием ROCm, особенно при наличии нулекопи между OpenCL и ROCm.

## Решение

Класс — `(unknown)`, метод `(unknown)`.

```cpp
  ConsoleOutput::GetInstance().Start();
  TestPrint("========================================================");
  TestPrint("  test_capon_opencl_to_rocm");
  TestPrint("  Данные заказчика -> OpenCL -> Zero Copy -> Capon ROCm");
  TestPrint("========================================================");

  test_01_detect_interop();
  test_02_customer_data_pipeline();
  test_03_zerocopy_matches_direct();
  test_04_beamform_customer_data();
  test_05_svm_customer_data();

  TestPrint("=== test_capon_opencl_to_rocm DONE ===");
```

## Граничные случаи

_Не определены (нет `@throws` в Doxygen primary_method)._

## Что делать дальше

- См. [linalg__capon_hip_opencl_to_rocm__usecase__v1](./capon_hip_opencl_to_rocm.md)
- См. [core__hybrid_backend__usecase__v1](./hybrid_backend.md)
- См. [core__zero_copy__usecase__v1](./zero_copy.md)

## Ссылки

- Источник кода: `E:/DSP-GPU/linalg/tests/test_capon_opencl_to_rocm.hpp:1`
