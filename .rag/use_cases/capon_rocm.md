---
schema_version: 1
kind: use_case
id: capon_rocm
repo: linalg
title: "Capon Rocm"
synonyms:
  ru:
    - []
  en:
    - []
primary_class: (unknown)
primary_method: (unknown)
related_classes:
related_use_cases:
  - linalg__capon_opencl_to_rocm__usecase__v1
  - linalg__capon_benchmark_rocm__usecase__v1
  - linalg__capon_hip_opencl_to_rocm__usecase__v1
maturity: stable
language: cpp
tags: []
ai_generated: false
human_verified: false
operator: alex
updated_at: 2026-05-13
---

# Use-case: Capon Rocm

## Когда применять

_LLM-fallback: см. описание класса._

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

- Источник кода: `/home/alex/DSP-GPU/linalg/tests/test_capon_rocm.hpp:1`
