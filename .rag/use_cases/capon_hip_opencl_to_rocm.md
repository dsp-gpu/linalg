---
schema_version: 1
kind: use_case
id: capon_hip_opencl_to_rocm
repo: linalg
title: "Capon Hip Opencl To Rocm"
synonyms:
  ru:
    - []
  en:
    - []
primary_class: (unknown)
primary_method: (unknown)
related_classes:
related_use_cases:
  - linalg__capon_benchmark_rocm__usecase__v1
  - core__rocm_backend__usecase__v1
  - core__rocm_external_context__usecase__v1
maturity: stable
language: cpp
tags: []
ai_generated: false
human_verified: false
operator: alex
updated_at: 2026-05-13
---

# Use-case: Capon Hip Opencl To Rocm

## Когда применять

_LLM-fallback: см. описание класса._

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

- См. [linalg__capon_benchmark_rocm__usecase__v1](./capon_benchmark_rocm.md)
- См. [core__rocm_backend__usecase__v1](./rocm_backend.md)
- См. [core__rocm_external_context__usecase__v1](./rocm_external_context.md)

## Ссылки

- Источник кода: `/home/alex/DSP-GPU/linalg/tests/test_capon_hip_opencl_to_rocm.hpp:1`
