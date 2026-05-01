#pragma once
#if ENABLE_ROCM

/**
 * @file diagonal_load_kernel_rocm.hpp
 * @brief HIP kernel source для diagonal loading (A[i,i].re += mu) — для hiprtc.
 *
 * @note Тип B (technical header): R"HIP(...)HIP" source для компиляции через
 *       GpuContext::CompileModule (disk cache v2). Kernel: diagonal_load
 *         - 1D grid: n threads (один поток = один диагональный элемент)
 *         - column-major: A[i,i] → A[i*n + i]
 *         - Только Re-часть: mu вещественное, A[i,i].im не меняется
 * @note Применение: регуляризация ковариационной матрицы перед инверсией
 *       (Тихонов / Ridge), Capon (MVDR) beamformer.
 * @note Launch: grid = (n + 255) / 256, block = 256.
 *
 * История:
 *   - Создан:  2026-03-16
 *   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
 */

namespace vector_algebra {
namespace kernels {

inline const char* GetDiagonalLoadKernelSource() {
  return R"HIP(

// ============================================================================
// diagonal_load — добавить mu к вещественной части диагонали матрицы A
// ============================================================================
// A         — [n × n], column-major, complex<float> хранится как float2
// mu        — вещественный коэффициент регуляризации
// n         — размер матрицы
//
// Диагональный элемент [i,i] в column-major: индекс = i*n + i
// Мнимая часть не меняется (mu вещественное → A[i,i].im не затрагивается)
//
// Launch: grid = (n + 255) / 256, block = 256
// Каждый thread обрабатывает один диагональный элемент.
// ============================================================================

extern "C" __global__ void diagonal_load(
    float2*      A,    // [n × n], column-major, complex<float>
    float        mu,   // коэффициент регуляризации
    unsigned int n)    // размер матрицы
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  // column-major: A[i, i] → A[i*n + i]
  A[i * n + i].x += mu;
}

)HIP";
}

}  // namespace kernels
}  // namespace vector_algebra

#endif  // ENABLE_ROCM
