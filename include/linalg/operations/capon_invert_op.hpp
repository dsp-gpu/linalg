#pragma once

// ============================================================================
// CaponInvertOp — обёртка над CholeskyInverterROCm: R → R^{-1} (Layer 5 Ref03)
//
// ЧТО:    Тонкий adapter-Op для шага инверсии в Capon-pipeline. Делегирует
//         реальную работу в vector_algebra::CholeskyInverterROCm:
//           - rocSOLVER cpotrf  → нижний треугольный множитель Холецкого L
//           - rocSOLVER cpotri  → инверсия из L через L^{-T}·L^{-1}
//           - симметризация     → заполнение верхнего треугольника (HIP kernel)
//         Не наследует GpuKernelOp (CholeskyInverterROCm — самостоятельный
//         модуль с собственным жизненным циклом и хэндлами rocSOLVER).
//
// ЗАЧЕМ:  Унификация шага в Ref03 pipeline: CovarianceMatrixOp → CaponInvertOp
//         → ComputeWeightsOp → ... — все шаги имеют одинаковый «вид» (Op с
//         Execute), даже если внутри разные реализации (rocBLAS / rocSOLVER /
//         HIP kernel). DRY: не дублировать логику CholeskyInverterROCm в
//         CaponProcessor — фасад остаётся тонким, инверсия инкапсулирована.
//
// ПОЧЕМУ: - Adapter Pattern (GoF): приводит интерфейс CholeskyInverterROCm
//           (Invert(InputData<void*>, n)) к стилю Capon Op'ов (Execute(gpu_R, P)).
//         - Не наследует GpuKernelOp: у CholeskyInverterROCm свой backend и
//           свой набор kernel'ов (для симметризации) — попытка наследовать =
//           конфликт двух владельцев hiprtc-модуля.
//         - Возвращает CholeskyResult (RAII-владелец GPU R^{-1}) — caller
//           (CaponProcessor::last_inv_) хранит его пока R^{-1} нужен. R^{-1}
//           НЕ кладётся в shared_buf, потому что rocSOLVER выделяет workspace
//           с собственным lifetime.
//         - SymmetrizeMode::GpuKernel — единственный быстрый путь на GPU
//           (CPU-вариант = D2H + симметризация + H2D, ×100 медленнее).
//
// Использование:
//   capon::CaponInvertOp inv_op(rocm_backend);
//   inv_op.CompileKernels();                // warmup hiprtc (опционально)
//   void* R_gpu = ctx.GetShared(shared_buf::kCovMatrix);
//   auto result = inv_op.Execute(R_gpu, P);  // result owns GPU R^{-1}
//   void* R_inv = result.AsHipPtr();         // → ComputeWeightsOp
//
// История:
//   - Создан:  2026-03-16 (Ref03 Layer 5, Capon pipeline)
//   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
// ============================================================================

#if ENABLE_ROCM

#include <core/services/gpu_kernel_op.hpp>
#include <core/interface/gpu_context.hpp>
#include <core/interface/input_data.hpp>
#include <linalg/capon_types.hpp>

#include <linalg/cholesky_inverter_rocm.hpp>   // vector_algebra::CholeskyInverterROCm
#include <linalg/vector_algebra_types.hpp>     // vector_algebra::CholeskyResult

#include <hip/hip_runtime.h>
#include <stdexcept>

namespace capon {

/**
 * @class CaponInvertOp
 * @brief Layer 5 Ref03 Op-adapter: R → R^{-1} через CholeskyInverterROCm.
 *
 * @note Non-copyable (CholeskyInverterROCm не копируется).
 * @note НЕ наследник GpuKernelOp — у inverter'а собственный backend и kernels.
 * @note Требует #if ENABLE_ROCM. Зависит от rocSOLVER.
 * @note Возвращает CholeskyResult (RAII) — caller хранит, пока R^{-1} нужен.
 * @see vector_algebra::CholeskyInverterROCm — реальная реализация
 * @see CaponProcessor::last_inv_ — место хранения CholeskyResult
 */
class CaponInvertOp {
public:
  /**
   * @param backend IBackend (ROCm). Должен жить дольше объекта.
   */
  explicit CaponInvertOp(drv_gpu_lib::IBackend* backend)
      : inverter_(backend, vector_algebra::SymmetrizeMode::GpuKernel) {}

  ~CaponInvertOp() = default;

  // Не копируемый (CholeskyInverterROCm не копируемый)
  CaponInvertOp(const CaponInvertOp&) = delete;
  CaponInvertOp& operator=(const CaponInvertOp&) = delete;

  /**
   * @brief Обратить матрицу R → R^{-1} через CholeskyInverterROCm
   * @param gpu_R      Указатель на R на GPU (complex<float>[P × P], column-major)
   * @param n_channels P — размер матрицы
   * @return CholeskyResult — владеет GPU-памятью R^{-1}
   *         (caller должен хранить result пока R^{-1} нужен)
   */
  vector_algebra::CholeskyResult Execute(void* gpu_R, uint32_t n_channels) {
    drv_gpu_lib::InputData<void*> input;
    input.data = gpu_R;

    return inverter_.Invert(input, static_cast<int>(n_channels));
  }

  /// Включить/выключить проверку POTRF/POTRI info (для benchmark — false)
  void SetCheckInfo(bool enabled) { inverter_.SetCheckInfo(enabled); }

  /// Явная прекомпиляция hiprtc kernel (warmup)
  void CompileKernels() { inverter_.CompileKernels(); }

private:
  vector_algebra::CholeskyInverterROCm inverter_;
};

}  // namespace capon

#endif  // ENABLE_ROCM
