#pragma once

/**
 * @brief HIP kernel-source для CaponProcessor (компилируется через hiprtc).
 *
 * @note Тип B (technical header): R"HIP(...)HIP" source для GpuContext::CompileModule().
 *       GEMM-операции (R=Y·Y^H, W=R^{-1}·U, Y_out=W^H·Y) — через rocBLAS, не здесь.
 *       Здесь только custom kernel'ы, для которых rocBLAS неэффективен/невозможен.
 *
 * Кернел `compute_capon_relief`:
 *   z[m] = 1 / Re(Σ_{p=0..P-1} conj(U[p,m]) · W[p,m])
 *   - один thread = одно направление m
 *   - launch: grid = (M+255)/256, block = 256
 *   - column-major: [p, m] → m·P + p
 *   - почему свой kernel, а не rocBLAS gemm + diag: gemm даёт U^H·W [M×M],
 *     но нужна только диагональ — 99% работы выкидывается.
 *
 * Соглашения:
 *   - complex<float> представлен как float2 (x=re, y=im)
 *   - все матрицы column-major (LAPACK/rocBLAS convention)
 *
 * История:
 *   - Создан:  2026-03-16
 *   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
 */

#if ENABLE_ROCM

namespace capon {
namespace kernels {

inline const char* GetCaponKernelSource() {
  return R"HIP(

// ============================================================================
// Kernel: compute_capon_relief
// ============================================================================
// z[m] = 1 / Re( Σ_{p=0}^{P-1} conj(U[p,m]) * W[p,m] )
//
// W = R^{-1} * U уже вычислена через rocBLAS CGEMM в CaponReliefOp::Execute().
// Каждый thread обрабатывает одно направление m.
//
// Launch: grid = (M + 255) / 256, block = 256
// ============================================================================

extern "C" __global__ void compute_capon_relief(
    const float2* __restrict__ U,   // [P × M], column-major
    const float2* __restrict__ W,   // W = R^{-1}*U [P × M], column-major
    float*                     z,   // выход: рельеф [M], float
    unsigned int               P,   // число каналов
    unsigned int               M)   // число направлений
{
  unsigned int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= M) return;

  // Суммируем Re(conj(U[p,m]) * W[p,m]) по всем каналам p
  float acc = 0.0f;

  // column-major: элемент [p, m] = m*P + p
  const float2* U_col = U + m * P;
  const float2* W_col = W + m * P;

  for (unsigned int p = 0; p < P; ++p) {
    float2 u = U_col[p];
    float2 w = W_col[p];
    // Re(conj(u) * w) = u.x*w.x + u.y*w.y
    acc += u.x * w.x + u.y * w.y;
  }

  // z[m] = 1 / Re(u^H * R^{-1} * u)
  // Защита от нуля: при корректной регуляризации acc > 0 всегда
  z[m] = (acc > 0.0f) ? (1.0f / acc) : 0.0f;
}

)HIP";
}

}  // namespace kernels
}  // namespace capon

#endif  // ENABLE_ROCM
