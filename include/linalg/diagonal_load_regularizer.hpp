#pragma once
#if ENABLE_ROCM

// ============================================================================
// DiagonalLoadRegularizer — A += mu·I на GPU (Concrete Strategy GoF)
//
// ЧТО:    Реализация IMatrixRegularizer: добавляет вещественный коэффициент mu
//         к диагонали комплексной квадратной матрицы (A[i,i].re += mu).
//         Запуск HIP kernel diagonal_load через hipModuleLaunchKernel,
//         kernel компилируется в конструкторе через GpuContext (disk cache v2).
//         Матрица A: n×n, complex<float>, column-major. In-place на GPU.
//
// ЗАЧЕМ:  Diagonal loading (Тихоновская регуляризация / Ridge) — стандартный
//         приём защиты ковариационной матрицы R от ill-conditioning перед
//         инверсией в Capon (MVDR), shrinkage estimators, adaptive filtering.
//         Без регуляризации малые eigenvalue → ε → R^{-1} взрывается. mu·I
//         сдвигает спектр, делает матрицу гарантированно invertible. Самая
//         простая и быстрая регуляризация (один kernel-launch, n threads).
//
// ПОЧЕМУ: - Concrete Strategy: реализует IMatrixRegularizer, подключается в
//           CaponProcessor через unique_ptr<IMatrixRegularizer>. Альтернатива
//           — NoOpRegularizer (отключение). Через DIP фасад не знает про
//           DiagonalLoadRegularizer напрямую.
//         - Move-only: copy запрещён (owns GpuContext + hipFunction_t cache).
//           Move нужен для возврата из factory / хранения в контейнере.
//         - GpuContext (unique_ptr) — компилирует diagonal_load.hpp source
//           один раз (disk cache v2, CompileKey keyed by source hash). При
//           повторном запуске процесса perekompiilaатсiя НЕ нужна.
//         - function_ кэшируется после ctx_->GetKernel — избегаем lookup
//           по имени на каждый Apply (горячий путь Capon).
//         - mu == 0.0f → kernel не запускается (no-op). Защита от пустого
//           вызова при динамическом выборе регуляризатора.
//         - stream параметр обязателен (default nullptr → backend stream).
//           Передавайте ctx.stream() для гарантии порядка после CGEMM.
//
// Использование:
//   DiagonalLoadRegularizer reg(backend);
//   reg.Apply(d_cov_matrix, P, 0.01f, ctx.stream());   // R += 0.01·I
//   inverter.Invert(InputData<void*>{d_cov_matrix}, P);
//
// История:
//   - Создан:  2026-03-16 (Concrete Strategy для Capon)
//   - Изменён: 2026-04-22 (миграция на GpuContext v2 disk cache,
//                          вместо ручного hiprtc + hipModule)
//   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
// ============================================================================

#include <linalg/i_matrix_regularizer.hpp>
#include <core/interface/i_backend.hpp>

#include <hip/hip_runtime.h>
#include <memory>

namespace drv_gpu_lib { class GpuContext; }

namespace vector_algebra {

/**
 * @class DiagonalLoadRegularizer
 * @brief Concrete Strategy: A += mu·I через скомпилированный HIP kernel.
 *
 * @note Move-only. Owns GpuContext (hipModule + disk cache v2).
 * @note Требует #if ENABLE_ROCM. На non-ROCm — класс не компилируется.
 * @note Не thread-safe (один kernel handle = один владелец).
 * @note mu == 0 → no-op (kernel не запускается).
 * @see IMatrixRegularizer (родительский интерфейс)
 * @see NoOpRegularizer (альтернатива при mu = 0)
 * @see CaponProcessor (главный потребитель)
 */
class DiagonalLoadRegularizer : public IMatrixRegularizer {
public:
  /**
   * @brief Конструктор — компилирует HIP kernel через GpuContext (cached).
   * @param backend  ROCm backend (должен быть инициализирован)
   * @throws std::runtime_error при ошибке компиляции или backend не ROCm
   */
  explicit DiagonalLoadRegularizer(drv_gpu_lib::IBackend* backend);

  ~DiagonalLoadRegularizer();

  DiagonalLoadRegularizer(const DiagonalLoadRegularizer&)            = delete;
  DiagonalLoadRegularizer& operator=(const DiagonalLoadRegularizer&) = delete;

  DiagonalLoadRegularizer(DiagonalLoadRegularizer&& other) noexcept;
  DiagonalLoadRegularizer& operator=(DiagonalLoadRegularizer&& other) noexcept;

  /**
   * @brief Применить диагональную загрузку: A[i,i] += mu (in-place на GPU).
   * @param d_matrix  GPU-указатель на матрицу n×n (complex<float>, column-major)
   * @param n         Размер матрицы
   * @param mu        Коэффициент регуляризации (mu > 0 рекомендуется)
   * @param stream    HIP stream (nullptr → использовать stream_ из backend).
   *                  Передавайте ctx.stream() для гарантии порядка исполнения
   *                  после предшествующего rocBLAS CGEMM.
   *
   * Если mu == 0.0f — kernel не запускается (no-op).
   */
  void Apply(void* d_matrix, int n, float mu,
             hipStream_t stream = nullptr) override;

private:
  hipStream_t                              stream_ = nullptr;  ///< Stream из backend (non-owning)
  std::unique_ptr<drv_gpu_lib::GpuContext> ctx_;                ///< Owns hipModule + cache v2
  void*                                    function_ = nullptr; ///< hipFunction_t (non-owning, из ctx_->GetKernel)
};

}  // namespace vector_algebra

#endif  // ENABLE_ROCM
