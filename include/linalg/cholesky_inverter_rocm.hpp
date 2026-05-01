#pragma once
#if ENABLE_ROCM

// ============================================================================
// CholeskyInverterROCm — инверсия эрмитовой HPD-матрицы (POTRF + POTRI)
//
// ЧТО:    RAII-обёртка над rocSOLVER POTRF (Cholesky факторизация A = U^H·U)
//         и POTRI (вычисление A^{-1} из U). Работает по комплексным эрмитовым
//         положительно определённым матрицам. Поддерживает 3 формата входа:
//         CPU vector, ROCm device pointer, OpenCL cl_mem (ZeroCopy). Batched-
//         версии InvertBatch проходят все матрицы одним handle. После POTRI
//         результат — только в верхнем треугольнике, симметризация в полную
//         форму через 2 режима (Roundtrip / GpuKernel).
//
// ЗАЧЕМ:  Capon (MVDR beamformer) и любой adaptive-pipeline требуют R^{-1}
//         от ковариационной матрицы. rocSOLVER POTRF/POTRI — самый быстрый
//         путь для HPD на GPU (быстрее общего GETRF + GETRI). Прямой вызов
//         rocSOLVER из CaponProcessor вынуждал бы тянуть rocsolver.h и
//         управлять workspace/info вручную — здесь это инкапсулировано.
//
// ПОЧЕМУ: - RAII + non-copy/non-move: класс владеет rocBLAS handle, hipModule
//           компилированного symmetrize-kernel'а и предаллоцированным d_info_
//           (rocblas_int[2]: slot 0=POTRF info, 1=POTRI info). Копирование
//           или перемещение этих ресурсов = двойное освобождение / chaos.
//         - Предаллоцированный d_info_ (Task_12) — убираем hipMalloc/hipFree
//           на каждый вызов Invert (горячий путь Capon: сотни вызовов в сек).
//         - SymmetrizeMode runtime-параметр: Roundtrip (Download → CPU sym →
//           Upload) — отладочный, без HIP kernel. GpuKernel (in-place HIP) —
//           production. Переключаемо через SetSymmetrizeMode для бенчмарков.
//         - GpuContext (unique_ptr) владеет hipModule с symmetrize_kernel'ом
//           через disk cache v2 (CompileKey). Повторный запуск процесса не
//           перекомпилирует — кэш на диске.
//         - CheckInfo отложенная — info с GPU читается одной синхронизацией
//           после всего pipeline (а не после каждого rocSOLVER-вызова), что
//           критично для batched-режима. Можно отключить (benchmark с
//           гарантированно HPD матрицами).
//
// Использование:
//   CholeskyInverterROCm inv(backend, SymmetrizeMode::GpuKernel);
//   auto result = inv.Invert(InputData<vector<complex<float>>>{matrix}, n);
//   void* d_inv = result.AsHipPtr();   // GPU pointer (caller НЕ Free!)
//   // batch:
//   auto batch_result = inv.InvertBatch(InputData<void*>{d_batch}, n);
//
// История:
//   - Создан:  2026-02-26 (Task_11: void* d_data вместо template<T> result)
//   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
// ============================================================================

#include <complex>
#include <memory>
#include <vector>
#include <core/interface/i_backend.hpp>
#include <core/interface/input_data.hpp>
#include <linalg/vector_algebra_types.hpp>

namespace drv_gpu_lib { class GpuContext; }

namespace vector_algebra {

/**
 * @class CholeskyInverterROCm
 * @brief Инверсия эрмитовой HPD-матрицы через rocSOLVER POTRF + POTRI.
 *
 * @note Не копируемый и не перемещаемый — owns rocBLAS handle, hipModule,
 *       d_info_, GpuContext.
 * @note Требует #if ENABLE_ROCM. На non-ROCm — stub бросает runtime_error.
 * @note Lifecycle: ctor(backend, mode) → Invert*/InvertBatch* → dtor.
 * @note Не thread-safe (один экземпляр = одна последовательность операций).
 * @see CholeskyResult — RAII-владелец GPU-памяти результата
 * @see SymmetrizeMode — выбор GpuKernel vs Roundtrip
 * @see DiagonalLoadRegularizer — типичный pre-step перед Invert
 * @ingroup grp_vector_algebra
 */
class CholeskyInverterROCm {
public:
  /**
   * @brief Конструктор.
   * @param backend  ROCmBackend (или HybridBackend). Должен жить дольше объекта.
   * @param mode     Режим симметризации (по умолчанию GpuKernel).
   */
  explicit CholeskyInverterROCm(
      drv_gpu_lib::IBackend* backend,
      SymmetrizeMode mode = SymmetrizeMode::GpuKernel);

  ~CholeskyInverterROCm();

  CholeskyInverterROCm(const CholeskyInverterROCm&) = delete;
  CholeskyInverterROCm& operator=(const CholeskyInverterROCm&) = delete;

  /// Изменить режим симметризации
  void SetSymmetrizeMode(SymmetrizeMode mode);

  /// Текущий режим симметризации
  SymmetrizeMode GetSymmetrizeMode() const { return mode_; }

  /// Явная прекомпиляция hiprtc kernel (для warmup/benchmark).
  /// Вызывается автоматически в конструкторе при GpuKernel mode.
  void CompileKernels();

  /// Включить/выключить проверку info (POTRF/POTRI). Default: true.
  /// Для benchmark/production с гарантированно HPD матрицами — false.
  void SetCheckInfo(bool enabled) { check_info_ = enabled; }

  // ─── Одна матрица ─────────────────────────────────────────────────────

  /// CPU вектор → GPU → CholeskyResult
  CholeskyResult Invert(
      const drv_gpu_lib::InputData<std::vector<std::complex<float>>>& input,
      int n = 0);

  /// ROCm device pointer → CholeskyResult
  CholeskyResult Invert(
      const drv_gpu_lib::InputData<void*>& input,
      int n = 0);

#ifdef CL_VERSION_1_0
  /// OpenCL cl_mem (ZeroCopy) → CholeskyResult
  CholeskyResult Invert(
      const drv_gpu_lib::InputData<cl_mem>& input,
      int n = 0);
#endif

  // ─── Batched ──────────────────────────────────────────────────────────

  /// CPU batched → CholeskyResult
  CholeskyResult InvertBatch(
      const drv_gpu_lib::InputData<std::vector<std::complex<float>>>& input,
      int n);

  /// GPU batched → CholeskyResult
  CholeskyResult InvertBatch(
      const drv_gpu_lib::InputData<void*>& input,
      int n);

#ifdef CL_VERSION_1_0
  /// cl_mem batched (ZeroCopy) → CholeskyResult
  CholeskyResult InvertBatch(
      const drv_gpu_lib::InputData<cl_mem>& input,
      int n);
#endif

private:
  drv_gpu_lib::IBackend* backend_;
  void* handle_ = nullptr;    ///< rocblas_handle (opaque)
  SymmetrizeMode mode_;

  // ─── Предаллоцированный dev_info (Task_12: убираем hipMalloc/hipFree) ──
  void* d_info_ = nullptr;    ///< rocblas_int[2] на GPU (slot 0=potrf, 1=potri)

  // ─── GpuContext (owns hipModule + disk cache v2) ─────────────────────
  std::unique_ptr<drv_gpu_lib::GpuContext> ctx_;
  void* sym_kernel_ = nullptr;   ///< hipFunction_t cached after ctx_->GetKernel()

  // ─── Core GPU ops ─────────────────────────────────────────────────────

  /// POTRF: Cholesky decomposition A = U^H * U
  void CorePotrf(void* d_matrix, int n, void* stream);

  /// POTRI: Compute A^{-1} from U
  void CorePotri(void* d_matrix, int n, void* stream);

  /// POTRF batched (sequential per matrix, uses same handle)
  void CorePotrfBatched(void* d_contiguous, int n, int batch, void* stream);

  /// POTRI batched
  void CorePotriBatched(void* d_contiguous, int n, int batch, void* stream);

  // ─── Symmetrize: Roundtrip ────────────────────────────────────────────

  /// Download → CPU symmetrize → Upload (одна матрица)
  void SymmetrizeRoundtrip(void* d_matrix, int n);

  /// Download → CPU symmetrize → Upload (batched)
  void SymmetrizeRoundtripBatched(void* d_contiguous, int n, int batch);

  // ─── Symmetrize: GPU Kernel (в symmetrize_gpu_rocm.cpp) ──────────────

  /// HIP kernel in-place (одна матрица)
  void SymmetrizeGpuKernel(void* d_matrix, int n, void* stream);

  /// HIP kernel (batched — цикл по матрицам)
  void SymmetrizeGpuKernelBatched(void* d_contiguous, int n, int batch,
                                   void* stream);

  // ─── Утилиты ──────────────────────────────────────────────────────────

  /// Диспетчер симметризации: выбирает Roundtrip или GpuKernel
  void Symmetrize(void* d_matrix, int n, void* stream);

  /// Диспетчер batched
  void SymmetrizeBatched(void* d_contiguous, int n, int batch, void* stream);

  /// Проверить d_info_ после pipeline (отложенная проверка, одна синхронизация).
  /// Вызывается автоматически если check_info_ == true.
  void CheckInfo(const char* context);

  bool check_info_ = true;  ///< true=проверять info (safe), false=пропустить (benchmark)

  /// Вычислить n из n_point (sqrt) или вернуть n_hint
  int ResolveMatrixSize(uint32_t n_point, int n_hint) const;

  /// CPU-side symmetrize (используется в Roundtrip)
  static void SymmetrizeUpperToFull(std::complex<float>* data, int n);
};

}  // namespace vector_algebra

#else  // !ENABLE_ROCM — Windows stub

#include <complex>
#include <vector>
#include <stdexcept>
#include <core/interface/i_backend.hpp>
#include <core/interface/input_data.hpp>
#include <linalg/vector_algebra_types.hpp>

namespace vector_algebra {

class CholeskyInverterROCm {
public:
  explicit CholeskyInverterROCm(drv_gpu_lib::IBackend*, SymmetrizeMode = SymmetrizeMode::GpuKernel) {}
  ~CholeskyInverterROCm() = default;

  CholeskyInverterROCm(const CholeskyInverterROCm&) = delete;
  CholeskyInverterROCm& operator=(const CholeskyInverterROCm&) = delete;

  void SetSymmetrizeMode(SymmetrizeMode) {}
  SymmetrizeMode GetSymmetrizeMode() const { return SymmetrizeMode::GpuKernel; }
  void CompileKernels() {}
  void SetCheckInfo(bool) {}

  CholeskyResult Invert(const drv_gpu_lib::InputData<std::vector<std::complex<float>>>&, int = 0) {
    throw std::runtime_error("CholeskyInverterROCm: ROCm not enabled");
  }
  CholeskyResult Invert(const drv_gpu_lib::InputData<void*>&, int = 0) {
    throw std::runtime_error("CholeskyInverterROCm: ROCm not enabled");
  }
  CholeskyResult InvertBatch(const drv_gpu_lib::InputData<std::vector<std::complex<float>>>&, int) {
    throw std::runtime_error("CholeskyInverterROCm: ROCm not enabled");
  }
  CholeskyResult InvertBatch(const drv_gpu_lib::InputData<void*>&, int) {
    throw std::runtime_error("CholeskyInverterROCm: ROCm not enabled");
  }
};

}  // namespace vector_algebra

#endif  // ENABLE_ROCM
