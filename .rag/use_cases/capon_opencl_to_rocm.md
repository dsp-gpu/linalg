---
schema_version: 1
kind: use_case
id: capon_opencl_to_rocm
repo: linalg
title: "Capon Opencl To Rocm"
synonyms:
  ru:
    - []
  en:
    - []
primary_class: (unknown)
primary_method: (unknown)
related_classes:
related_use_cases:
  - linalg__capon_hip_opencl_to_rocm__usecase__v1
  - linalg__capon_benchmark_rocm__usecase__v1
  - core__rocm_backend__usecase__v1
maturity: stable
language: cpp
tags: []
ai_generated: false
human_verified: false
operator: alex
updated_at: 2026-05-13
---

# Use-case: Capon Opencl To Rocm

## Когда применять

_LLM-fallback: см. описание класса._

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
- См. [linalg__capon_benchmark_rocm__usecase__v1](./capon_benchmark_rocm.md)
- См. [core__rocm_backend__usecase__v1](./rocm_backend.md)

## Ссылки

- Источник кода: `/home/alex/DSP-GPU/linalg/tests/test_capon_opencl_to_rocm.hpp:1`
