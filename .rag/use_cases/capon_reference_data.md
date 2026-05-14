---
schema_version: 1
kind: use_case
id: capon_reference_data
repo: linalg
title: "Capon Reference Data"
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
  - linalg__capon_opencl_to_rocm__usecase__v1
  - linalg__capon_hip_opencl_to_rocm__usecase__v1
maturity: stable
language: cpp
tags: []
ai_generated: false
human_verified: false
operator: alex
updated_at: 2026-05-13
---

# Use-case: Capon Reference Data

## Когда применять

_LLM-fallback: см. описание класса._

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

- См. [linalg__capon_benchmark_rocm__usecase__v1](./capon_benchmark_rocm.md)
- См. [linalg__capon_opencl_to_rocm__usecase__v1](./capon_opencl_to_rocm.md)
- См. [linalg__capon_hip_opencl_to_rocm__usecase__v1](./capon_hip_opencl_to_rocm.md)

## Ссылки

- Источник кода: `/home/alex/DSP-GPU/linalg/tests/test_capon_reference_data.hpp:1`
