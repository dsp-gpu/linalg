#pragma once

// ============================================================================
// ComputeWeightsOp — весовая матрица W = R^{-1}·U (Layer 5 Ref03)
//
// ЧТО:    Concrete Op (наследник GpuKernelOp): средний шаг Capon-pipeline.
//         Один rocBLAS CGEMM:
//           W[P × M] = R^{-1}[P × P] · U[P × M]   (NoTrans × NoTrans)
//         Источник R^{-1} — CholeskyResult из CaponInvertOp (передаётся как
//         R_inv_ptr). U — управляющие векторы из shared kSteering. W —
//         в shared kWeight (lazy alloc через RequireShared).
//
// ЗАЧЕМ:  W используется ОБОИМИ финальными Op'ами (CaponReliefOp для рельефа,
//         AdaptBeamformOp для ДО). Если бы каждый из них делал свой CGEMM —
//         R^{-1}·U считался бы дважды для пайплайна, который вычисляет и
//         рельеф, и beam output (типичный сценарий RadarPipeline). Вынос в
//         отдельный Op + shared kWeight = вычисляем W один раз, переиспользуем.
//
// ПОЧЕМУ: - Layer 5 Ref03: один Op = один логический шаг (CGEMM R^{-1}·U).
//         - Stateless (нет приватных GPU-буферов) — kWeight в shared ctx_.
//         - DRY: устранили дублирование CGEMM между CaponReliefOp и
//           AdaptBeamformOp (см. историю — раньше каждый делал свой gemm).
//         - rocBLAS NoTrans × NoTrans: R^{-1} симметрична эрмитова после
//           POTRI+симметризации, прямое умножение корректно.
//         - column-major: согласованно со всем pipeline (Y, U, R, W, Y_out).
//         - RequireShared lazy alloc: первый вызов выделит kWeight, повторные
//           переиспользуют (если M не изменилось).
//
// Использование:
//   // Внутри CaponProcessor::RunComputeWeights:
//   void* R_inv = last_inv_.AsHipPtr();
//   weights_op_.Execute(P, M, R_inv, mat_ops_);
//   // → kWeight содержит W[P×M], готово для CaponReliefOp / AdaptBeamformOp.
//
// История:
//   - Создан:  2026-03-16 (Ref03 Layer 5, Capon pipeline; устранение
//                          дублирования CGEMM в Relief и Beam Op'ах)
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
 * @class ComputeWeightsOp
 * @brief Layer 5 Ref03 Op: W = R^{-1}·U через rocBLAS CGEMM, результат в kWeight.
 *
 * @note Stateless (нет приватных буферов). kWeight — в shared ctx_.
 * @note Требует #if ENABLE_ROCM. Зависит от rocBLAS (через MatrixOpsROCm).
 * @note Предусловие: CaponInvertOp::Execute() вернул валидный R_inv_ptr.
 * @note Постусловие: kWeight содержит W[P×M] для CaponReliefOp И AdaptBeamformOp.
 * @see CaponInvertOp — поставщик R^{-1}
 * @see CaponReliefOp, AdaptBeamformOp — потребители kWeight
 */
class ComputeWeightsOp : public drv_gpu_lib::GpuKernelOp {
public:
  /**
   * @brief Возвращает имя Op'а для логирования и профилирования.
   *
   * @return C-строка "ComputeWeights" (статический литерал).
   *   @test_check std::string(result) == "ComputeWeights"
   */
  const char* Name() const override { return "ComputeWeights"; }

  /**
   * @brief Вычислить W = R^{-1} * U  (MatrixOpsROCm::Multiply)
   * @param n_channels   P — число каналов
   *   @test { range=[1..50000], value=128, unit="лучей/каналов", error_values=[-1, 100000, 3.14] }
   * @param n_directions M — число направлений
   * @param R_inv_ptr    R^{-1} на GPU — из CholeskyResult::AsHipPtr()
   *   @test { pattern=gpu_pointer, values=["valid_alloc", nullptr], error_values=[0xDEADBEEF, null] }
   * @param mat          MatrixOpsROCm из CaponProcessor (stream привязан к ctx_)
   *
   * Читает: kSteering (U) [P × M]
   * Пишет:  kWeight   (W) [P × M]  (аллоцирует при необходимости)
   */
  void Execute(uint32_t n_channels, uint32_t n_directions, void* R_inv_ptr,
               vector_algebra::MatrixOpsROCm& mat) {
    const int P = static_cast<int>(n_channels);
    const int M = static_cast<int>(n_directions);

    void* U = ctx_->GetShared(shared_buf::kSteering);
    void* W = ctx_->RequireShared(
        shared_buf::kWeight,
        static_cast<size_t>(P) * M * sizeof(rocblas_float_complex));

    // W[P×M] = R^{-1}[P×P] * U[P×M]  (NoTrans × NoTrans)
    mat.Multiply(R_inv_ptr, U, W, P, M, P);
  }

protected:
  void OnRelease() override {}  // нет приватных GPU буферов
};

}  // namespace capon

#endif  // ENABLE_ROCM
