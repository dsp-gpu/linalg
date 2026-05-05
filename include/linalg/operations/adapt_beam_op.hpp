#pragma once

// ============================================================================
// AdaptBeamformOp — адаптивное диаграммообразование Y_out = W^H·Y (Layer 5 Ref03)
//
// ЧТO:    Concrete Op (наследник GpuKernelOp): финальный шаг Capon-pipeline для
//         режима адаптивного ДО (beamforming). Один rocBLAS CGEMM:
//           Y_out[M × N] = W^H[M × P] · Y[P × N]   (ConjTransA × NoTrans)
//         где W = R^{-1}·U уже вычислена ComputeWeightsOp в kWeight.
//
// ЗАЧЕМ:  Разделение Capon-pipeline на atomic Op'ы (SRP): этот Op знает только
//         «как умножить W^H на Y», не лезет в инверсию или ковариацию. Если
//         в будущем нужен другой beamformer (MUSIC, ESPRIT) — добавляется
//         параллельный Op без правки этого. Также позволяет переиспользовать
//         kWeight для CaponReliefOp без повторного CGEMM.
//
// ПОЧЕМУ: - Layer 5 Ref03: один Op = один логический шаг (CGEMM W^H·Y).
//         - Stateless (нет приватных GPU-буферов, OnRelease пустой) — все
//           буферы в shared kSignal/kWeight/kOutput через ctx_.
//         - rocBLAS column-major: W [P × M], W^H трактуется как [M × P]
//           через rocblas_operation_conjugate_transpose — без явного transpose
//           kernel'а, экономит pass через память.
//         - Read kSignal + kWeight, write kOutput — нет hipMalloc внутри
//           Execute (RequireShared lazy alloc один раз).
//
// Использование:
//   // Внутри CaponProcessor::AdaptiveBeamform после ComputeWeightsOp:
//   beam_op_.Execute(P, N, M, mat_ops_);
//   // → kOutput содержит Y_out[M×N], читается ReadBeamResult().
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
 * @class AdaptBeamformOp
 * @brief Layer 5 Ref03 Op: финальный CGEMM Y_out = W^H·Y для адаптивного ДО.
 *
 * @note Stateless (нет приватных буферов). Все данные в shared-буферах ctx_.
 * @note Требует #if ENABLE_ROCM. Зависит от rocBLAS (через MatrixOpsROCm).
 * @note Предусловие: ComputeWeightsOp::Execute() уже записал W в kWeight.
 * @see ComputeWeightsOp — поставщик kWeight
 * @see CaponReliefOp — параллельный финальный Op (для рельефа вместо ДО)
 * @see vector_algebra::MatrixOpsROCm
 */
class AdaptBeamformOp : public drv_gpu_lib::GpuKernelOp {
public:
  /**
   * @brief Возвращает имя Op'а для логирования и профилирования.
   *
   * @return C-строка "AdaptBeamform" (статический литерал).
   *   @test_check std::string(result) == "AdaptBeamform"
   */
  const char* Name() const override { return "AdaptBeamform"; }

  /**
   * @brief Адаптивное ДО: Y_out = W^H * Y
   * @param n_channels   P — число каналов
   *   @test { range=[1..50000], value=128, unit="лучей/каналов", error_values=[-1, 100000, 3.14] }
   * @param n_samples    N — число отсчётов
   *   @test { range=[100..1300000], value=6000, error_values=[-1, 3000000, 3.14] }
   * @param n_directions M — число направлений (лучей)
   * @param mat          MatrixOpsROCm из CaponProcessor (stream привязан к ctx_)
   *
   * Читает: kSignal (Y) [P × N],  kWeight (W = R^{-1}*U) [P × M]
   * Пишет:  kOutput (complex<float>[M × N])
   *
   * Предусловие: ComputeWeightsOp::Execute() уже запущен и записал W в kWeight.
   */
  void Execute(uint32_t n_channels, uint32_t n_samples, uint32_t n_directions,
               vector_algebra::MatrixOpsROCm& mat) {
    const int P = static_cast<int>(n_channels);
    const int N = static_cast<int>(n_samples);
    const int M = static_cast<int>(n_directions);

    void* Y   = ctx_->GetShared(shared_buf::kSignal);
    void* W   = ctx_->GetShared(shared_buf::kWeight);
    void* out = ctx_->RequireShared(shared_buf::kOutput,
                                    static_cast<size_t>(M) * N * sizeof(rocblas_float_complex));

    // Y_out[M×N] = W^H[M×P] * Y[P×N]  (ConjTransA × NoTrans)
    mat.MultiplyConjTransA(W, Y, out, M, N, P);
  }

protected:
  void OnRelease() override {}  // нет приватных GPU буферов
};

}  // namespace capon

#endif  // ENABLE_ROCM
