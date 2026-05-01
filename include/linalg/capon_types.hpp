#pragma once

/**
 * @brief Типы данных модуля Capon — параметры, результаты, индексы shared-буферов.
 *
 * @note Тип B (technical header): POD-структуры без логики, только default'ы и enum-индексы.
 *       Валидация (P>0, N>=P, M>0) — в CaponProcessor::ComputeRelief / AdaptiveBeamform.
 *
 * Алгоритм Кейпона (MVDR — Minimum Variance Distortionless Response):
 *   1. R = (1/N)·Y·Y^H + μ·I        — ковариационная матрица + регуляризация
 *   2. R^{-1}                        — rocSOLVER POTRF + POTRI (Cholesky-инверсия)
 *   3. z[m] = 1 / Re(u_m^H·R^{-1}·u_m)  — рельеф (angular power spectrum)
 *   4. W = R^{-1}·U,  Y_out = W^H·Y  — адаптивное диаграммообразование
 *
 *   Y — матрица сигнала            [n_channels × n_samples]
 *   U — управляющие векторы        [n_channels × n_directions]
 *   R — ковариационная матрица     [n_channels × n_channels]
 *   W — весовая матрица            [n_channels × n_directions]
 *
 * История:
 *   - Создан:  2026-03-16
 *   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
 */

#include <complex>
#include <cstdint>
#include <vector>

namespace capon {

// ============================================================================
// Параметры алгоритма
// ============================================================================

struct CaponParams {
  uint32_t n_channels;    ///< P — число антенных каналов (строки Y и U)
  uint32_t n_samples;     ///< N — число временных отсчётов (столбцы Y)
  uint32_t n_directions;  ///< M — число направлений (столбцы U)
  float    mu = 0.0f;     ///< Коэффициент регуляризации (диагональная загрузка)
};

// ============================================================================
// Результаты
// ============================================================================

/// Рельеф Кейпона — M вещественных значений (пространственный спектр)
struct CaponReliefResult {
  std::vector<float> relief;  ///< Re(1 / (u^H * R^{-1} * u)) — размер n_directions
};

/// Адаптивное диаграммообразование — матрица [n_directions × n_samples]
struct CaponBeamResult {
  uint32_t n_directions = 0;
  uint32_t n_samples    = 0;
  std::vector<std::complex<float>> output;  ///< n_directions × n_samples (row-major)
};

// Индексы разделяемых буферов (GpuContext shared buffers) — только ROCm
#if ENABLE_ROCM
namespace shared_buf {
  static constexpr size_t kSignal    = 0;
  static constexpr size_t kSteering  = 1;
  static constexpr size_t kCovMatrix = 2;
  static constexpr size_t kWeight    = 3;
  static constexpr size_t kOutput    = 4;
  static constexpr size_t kCount     = 5;
}  // namespace shared_buf
#endif  // ENABLE_ROCM

}  // namespace capon
