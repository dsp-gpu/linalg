#pragma once

// ============================================================================
// CovarianceMatrixOp — ковариация R = (1/N)·Y·Y^H (Layer 5 Ref03)
//
// ЧТО:    Concrete Op (наследник GpuKernelOp): первый шаг Capon-pipeline.
//         Делегирует в MatrixOpsROCm::CovarianceMatrix:
//           R[P × P] = (1/N) · Y[P × N] · Y^H[N × P]   (rocBLAS CHERK или CGEMM)
//         где Y — матрица сигнала из shared kSignal, R записывается в
//         shared kCovMatrix (lazy alloc через RequireShared).
//
// ЗАЧЕМ:  Ковариационная матрица — основа любого MVDR/Capon. Отделение в
//         собственный Op (а не «всё в одном CaponProcessor») даёт SRP:
//         замена реализации (CHERK vs CGEMM, batched vs single, MUSIC-style
//         с выбором размера окна) — изолирована в этом классе. Регуляризация
//         (R += μ·I) сюда НЕ входит — она через Strategy IMatrixRegularizer
//         в CaponProcessor (DIP).
//
// ПОЧЕМУ: - Layer 5 Ref03: один Op = один логический шаг (rocBLAS gemm/herk).
//         - Stateless (нет приватных GPU-буферов) — kCovMatrix в shared ctx_.
//         - rocBLAS column-major: Y[P×N] (P каналов × N снимков) → Y·Y^H даёт
//           R[P×P] (channel-channel ковариация). Для P=4..32, N=1024..16384 —
//           ~1 ms на gfx1201 (memory-bound для больших N).
//         - SRP: регуляризация R += μ·I — ответственность отдельного класса
//           (DiagonalLoadRegularizer) через IMatrixRegularizer Strategy.
//           Здесь — только «чистая» ковариация. Можно тестировать раздельно.
//         - Нормировка 1/N — внутри MatrixOpsROCm::CovarianceMatrix (alpha
//           параметр CHERK/CGEMM), не делает отдельного pass через память.
//
// Использование:
//   // Внутри CaponProcessor::RunCovAndInvert:
//   cov_op_.Execute(P, N, mat_ops_);
//   regularizer_->Apply(ctx_, P, params.mu);   // R += μ·I (Strategy)
//   inv_op_->Execute(R, P);                     // R^{-1} → last_inv_
//
// История:
//   - Создан:  2026-03-16 (Ref03 Layer 5, Capon pipeline)
//   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
// ============================================================================

#if ENABLE_ROCM

#include <core/services/gpu_kernel_op.hpp>
#include <core/interface/gpu_context.hpp>
#include <linalg/capon_types.hpp>
#include <linalg/matrix_ops_rocm.hpp>

#include <rocblas/rocblas.h>

namespace capon {

/**
 * @class CovarianceMatrixOp
 * @brief Layer 5 Ref03 Op: R = (1/N)·Y·Y^H через rocBLAS, результат в kCovMatrix.
 *
 * @note Stateless (нет приватных буферов). kCovMatrix — в shared ctx_.
 * @note Требует #if ENABLE_ROCM. Зависит от rocBLAS (через MatrixOpsROCm).
 * @note Регуляризация (R += μ·I) — НЕ здесь, в IMatrixRegularizer (Strategy).
 * @see vector_algebra::MatrixOpsROCm::CovarianceMatrix
 * @see vector_algebra::IMatrixRegularizer — Strategy для μ·I
 * @see CaponInvertOp — следующий шаг pipeline
 */
class CovarianceMatrixOp : public drv_gpu_lib::GpuKernelOp {
public:
  /**
   * @brief Возвращает имя Op'а для логирования и профилирования.
   *
   * @return C-строка "CovarianceMatrix" (статический литерал).
   *   @test_check std::string(result) == "CovarianceMatrix"
   */
  const char* Name() const override { return "CovarianceMatrix"; }

  /**
   * @brief Вычислить R = (1/N) * Y * Y^H
   * @param n_channels P — число каналов
   *   @test { range=[1..50000], value=128, unit="лучей/каналов" }
   * @param n_samples  N — число отсчётов
   *   @test { range=[100..1300000], value=6000 }
   * @param mat        MatrixOpsROCm из CaponProcessor (stream привязан к ctx_)
   *
   * Читает: ctx_->GetShared(kSignal)  [P × N]
   * Пишет:  ctx_->RequireShared(kCovMatrix, P*P*sizeof(complex<float>))  [P × P]
   */
  void Execute(uint32_t n_channels, uint32_t n_samples,
               vector_algebra::MatrixOpsROCm& mat) {
    const int P = static_cast<int>(n_channels);
    const int N = static_cast<int>(n_samples);

    void* Y = ctx_->GetShared(shared_buf::kSignal);
    void* R = ctx_->RequireShared(
        shared_buf::kCovMatrix,
        static_cast<size_t>(P) * P * sizeof(rocblas_float_complex));

    mat.CovarianceMatrix(Y, P, N, R);
  }

protected:
  void OnRelease() override {}  // нет приватных GPU буферов
};

}  // namespace capon

#endif  // ENABLE_ROCM
