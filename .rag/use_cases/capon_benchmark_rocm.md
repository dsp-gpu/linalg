---
schema_version: 1
kind: use_case
id: capon_benchmark_rocm
repo: linalg
title: "Как выполнить алгоритм Кейпона на GPU с использованием ROCm"
synonyms:
  ru:
    - "алгоритм кейпона gpu"
    - "обработка сигналов антенный массив"
    - "rocм бенчмарк кейпона"
    - "кейпон на amd gpu"
    - "оптимизация кейпона roc"
    - "hip kernels кейпон"
    - "бенчмарк антенный массив"
    - "gpu обработка сигналов кейпон"
  en:
    - "capon algorithm gpu"
    - "antenna array signal processing"
    - "rocml benchmark capon"
    - "capon on amd gpu"
    - "optimize capon roc"
    - "hip kernels capon"
    - "antenna array benchmark"
    - "gpu signal processing capon"
primary_class: (unknown)
primary_method: (unknown)
related_classes:
related_use_cases:
  - heterodyne__heterodyne_benchmark_rocm__usecase__v1
  - spectrum__lch_farrow_rocm__usecase__v1
  - spectrum__lch_farrow_benchmark_rocm__usecase__v1
maturity: stable
language: cpp
tags: [linalg, capon, roc, gpu, hip, antenna_array, batch_processing, signal_processing, optimization, benchmark]
ai_generated: true
human_verified: false
operator: ai
updated_at: 2026-05-06
---

# Use-case: Как выполнить алгоритм Кейпона на GPU с использованием ROCm

## Когда применять

Когда требуется оптимизация обработки сигналов антенных массивов на AMD GPU с использованием ROCm и HIP kernels

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
    capon::CaponParams params;
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

- См. [heterodyne__heterodyne_benchmark_rocm__usecase__v1](./heterodyne_benchmark_rocm.md)
- См. [spectrum__lch_farrow_rocm__usecase__v1](./lch_farrow_rocm.md)
- См. [spectrum__lch_farrow_benchmark_rocm__usecase__v1](./lch_farrow_benchmark_rocm.md)

## Ссылки

- Источник кода: `E:/DSP-GPU/linalg/tests/test_capon_benchmark_rocm.hpp:1`
