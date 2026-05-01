#pragma once
#if ENABLE_ROCM

/**
 * @file symmetrize_kernel_sources_rocm.hpp
 * @brief HIP kernel source для симметризации upper → full (после POTRI), для hiprtc.
 *
 * @note Тип B (technical header): R"HIP(...)HIP" source. Kernel:
 *       symmetrize_upper_to_full
 *         - 2D grid: (ceil(n/16), ceil(n/16)), block (16, 16)
 *         - один поток = один элемент (row, col)
 *         - copy upper → lower с conjugate: A[col][row] = conj(A[row][col])
 *           (float2: {.x = re, .y = -im})
 * @note Применение: после rocSOLVER POTRI результат лежит ТОЛЬКО в верхнем
 *       треугольнике (LAPACK convention) — для эрмитовости нужно скопировать
 *       upper → lower с сопряжением. Часть pipeline CholeskyInverterROCm в
 *       SymmetrizeMode::GpuKernel (in-place GPU, без round-trip на CPU).
 *
 * История:
 *   - Создан:  2026-02-26
 *   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
 */

namespace vector_algebra {
namespace kernels {

inline const char* GetSymmetrizeKernelSource() {
  return R"HIP(

// ============================================================================
// symmetrize_upper_to_full — заполнить нижний треугольник (conjugate)
// ============================================================================
// float2 = complex<float>: .x = real, .y = imag
// conjugate: {v.x, -v.y}
//
// Запускать с grid (ceil(n/16), ceil(n/16)), block (16, 16).
// Каждый поток обрабатывает один элемент (row, col).
// Если col > row → записать conj в (col, row).

extern "C" __global__ void symmetrize_upper_to_full(
    float2* __restrict__ data,
    unsigned int n)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= n || col >= n) return;

    // Только верхний треугольник: col > row
    if (col > row) {
        float2 v = data[row * n + col];
        data[col * n + row] = {v.x, -v.y};  // conjugate
    }
}

)HIP";
}

}  // namespace kernels
}  // namespace vector_algebra

#endif  // ENABLE_ROCM
