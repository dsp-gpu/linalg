---
schema_version: 1
kind: use_case
id: capon_benchmark_rocm
repo: linalg
title: "Capon Benchmark Rocm"
synonyms:
  ru:
    - []
  en:
    - []
primary_class: (unknown)
primary_method: (unknown)
related_classes:
related_use_cases:
  - spectrum__filters_benchmark_rocm__usecase__v1
  - stats__statistics_rocm__usecase__v1
  - spectrum__moving_average_rocm__usecase__v1
maturity: stable
language: cpp
tags: []
ai_generated: false
human_verified: false
operator: alex
updated_at: 2026-05-13
---

# Use-case: Capon Benchmark Rocm

## Когда применять

_LLM-fallback: см. описание класса._

## Решение

Класс — `(unknown)`, метод `(unknown)`.

```cpp
  TestPrint("============================================================");
  TestPrint("  Capon Benchmark (ComputeRelief / AdaptiveBeamform) — ROCm");
  TestPrint("============================================================");

  // Проверить AMD GPU
  if (drv_gpu_lib::ROCmCore::GetAvailableDeviceCount() == 0) {
    TestPrint("  [SKIP] No AMD GPU available");
    return;
  }

  try {
    // ── ROCm backend ──────────────────────────────────────────────────────
    auto* backend = &capon_test_helpers::GetROCmBackend();

    // ── Параметры Кейпона ─────────────────────────────────────────────────
    dsp::linalg::CaponParams params;
    params.n_channels   = 16;   // P — число антенных каналов
    params.n_samples    = 256;  // N — число временных отсчётов
    params.n_directions = 64;   // M — число направлений сканирования
    params.mu           = 0.01f;

    // ── Тестовые данные ───────────────────────────────────────────────────
    const auto signal   = capon_test_helpers::MakeNoise(
        static_cast<size_t>(params.n_channels) * params.n_samples, 1.0f, 42u);
    const auto steering = capon_test_helpers::MakeSteeringMatrix(
        params.n_channels, params.n_directions,
        -static_cast<float>(M_PI) / 3.0f,
         static_cast<float>(M_PI) / 3.0f);

    // ── Создать процессор (компилирует HIP kernels один раз) ──────────────
// ... (truncated)
```

## Граничные случаи

_Не определены (нет `@throws` в Doxygen primary_method)._

## Что делать дальше

- См. [spectrum__filters_benchmark_rocm__usecase__v1](./filters_benchmark_rocm.md)
- См. [stats__statistics_rocm__usecase__v1](./statistics_rocm.md)
- См. [spectrum__moving_average_rocm__usecase__v1](./moving_average_rocm.md)

## Ссылки

- Источник кода: `/home/alex/DSP-GPU/linalg/tests/test_capon_benchmark_rocm.hpp:1`
