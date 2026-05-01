#pragma once

// ============================================================================
// CaponProcessor — фасад алгоритма Кейпона (MVDR) на GPU (Layer 6 Ref03)
//
// ЧТО:    Фасад для adaptive beamforming по методу Кейпона (Minimum Variance
//         Distortionless Response). Координирует pipeline из 5 Layer-5 Op'ов:
//           1. CovarianceMatrixOp   → R = (1/N)·Y·Y^H            (rocBLAS CGEMM)
//           2. DiagonalLoadRegularizer → R += μ·I                (HIP kernel)
//           3. CaponInvertOp        → R^{-1}                     (rocSOLVER POTRF+POTRI)
//           4. ComputeWeightsOp     → W = R^{-1}·U               (rocBLAS CGEMM → kWeight)
//           5a. CaponReliefOp       → z[m] = 1/Re(u_m^H·W[:,m])  (HIP kernel)  — для рельефа
//           5b. AdaptBeamformOp     → Y_out = W^H·Y              (rocBLAS CGEMM) — для ДО
//         Поддерживает CPU- и GPU-входы (upload или D2D-копия в kSignal/kSteering).
//
// ЗАЧЕМ:  Это публичный API модуля linalg для Capon. Python-биндинги,
//         RadarPipeline и пользовательские тесты создают один CaponProcessor
//         и вызывают ComputeRelief/AdaptiveBeamform — без знания о rocBLAS,
//         rocSOLVER и порядке Op'ов внутри. SRP: фасад ТОЛЬКО координирует
//         (DI: Op'ы инжектируются как value-члены, IMatrixRegularizer —
//         через Strategy unique_ptr).
//
// ПОЧЕМУ: - Layer 6 Ref03 (Facade): не делает kernel launch'ей сам, делегирует
//           Op'ам через ctx_ (GpuContext — Layer 1, единая точка для compiled
//           kernels и shared buffers).
//         - Op'ы хранятся как value-члены (cov_op_, weights_op_, relief_op_,
//           beam_op_) — zero-overhead, инициализируются один раз через ctx_.
//           CaponInvertOp — в unique_ptr (CholeskyInverterROCm non-copyable,
//           unique_ptr даёт корректный move без boilerplate).
//         - Strategy IMatrixRegularizer (DiagonalLoadRegularizer по умолчанию)
//           через unique_ptr → DIP: фасад зависит от абстракции, можно
//           подменять (LoadingFactor, Tikhonov, MUSIC-like, ...) без правки.
//         - last_inv_ (CholeskyResult) хранит R^{-1} между шагами pipeline —
//           ComputeRelief и AdaptiveBeamform не пересчитывают инверсию,
//           если R не изменилось.
//         - GpuContext per-module → thread-safe by instance: каждый
//           CaponProcessor держит свой stream, параллельные вызовы из разных
//           потоков на разных экземплярах не конфликтуют.
//         - Move-only (=delete copy) — owns GPU buffers через ctx_; копировать
//           = chaos с lifetime hipMalloc'ов.
//         - rocBLAS column-major (как BLAS): Y/U/R/W/Y_out — все column-major,
//           rocblas_set_stream привязывает handle к ctx_.stream() лениво
//           внутри MatrixOpsROCm.
//
// Использование:
//   capon::CaponProcessor proc(rocm_backend);
//   capon::CaponParams params{
//       .n_channels   = 16,    // P — антенны
//       .n_samples    = 1024,  // N — снимки
//       .n_directions = 181,   // M — углы θ ∈ [-90°, +90°]
//       .mu           = 1e-3f, // диагональная загрузка
//   };
//   auto relief = proc.ComputeRelief(signal_iq, steering_vectors, params);
//   // relief.relief[m] = P(θ_m), m=0..M-1 — angular power spectrum
//
// История:
//   - Создан:  2026-03-16 (миграция Capon из vector_algebra/Doc_Addition)
//   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
// ============================================================================

#if ENABLE_ROCM

#include <linalg/capon_types.hpp>
#include <core/interface/gpu_context.hpp>

// Op классы (Layer 5)
#include <linalg/operations/covariance_matrix_op.hpp>
#include <linalg/operations/capon_invert_op.hpp>      // обёртка над CholeskyInverterROCm
#include <linalg/operations/compute_weights_op.hpp>   // W = R^{-1}*U  (единый, без дублирования)
#include <linalg/operations/capon_relief_op.hpp>
#include <linalg/operations/adapt_beam_op.hpp>

// Тип результата инверсии из vector_algebra
#include <linalg/vector_algebra_types.hpp>

// Регуляризация через Strategy (DIP — зависим от интерфейсе)
#include <linalg/i_matrix_regularizer.hpp>
#include <linalg/diagonal_load_regularizer.hpp>

// GEMM операции через vector_algebra (единый фасад для rocBLAS)
#include <linalg/matrix_ops_rocm.hpp>

#include <core/interface/i_backend.hpp>

#include <complex>
#include <memory>
#include <vector>
#include <cstdint>

namespace capon {

/**
 * @class CaponProcessor
 * @brief Layer 6 Ref03 фасад: pipeline алгоритма Кейпона (MVDR) на ROCm.
 *
 * @note Move-only (copy запрещён) — owns GPU buffers через GpuContext.
 * @note Не thread-safe per-instance. Параллельные вызовы — на разных экземплярах.
 * @note Требует #if ENABLE_ROCM. Без ROCm — stub с runtime_error.
 * @note Lifecycle: ctor(backend) → ComputeRelief / AdaptiveBeamform → dtor.
 * @see CovarianceMatrixOp, CaponInvertOp, ComputeWeightsOp, CaponReliefOp, AdaptBeamformOp
 * @see vector_algebra::CholeskyInverterROCm — реальная инверсия R^{-1}
 * @see vector_algebra::IMatrixRegularizer — Strategy для регуляризации
 * @ingroup grp_capon
 */
class CaponProcessor {
public:
  // =========================================================================
  // Constructor / Destructor
  // =========================================================================

  /**
   * @brief Конструктор
   * @param backend Указатель на IBackend (не владеющий, должен быть ROCm backend)
   */
  explicit CaponProcessor(drv_gpu_lib::IBackend* backend);

  ~CaponProcessor();

  // No copying
  CaponProcessor(const CaponProcessor&) = delete;
  CaponProcessor& operator=(const CaponProcessor&) = delete;

  // Move semantics
  CaponProcessor(CaponProcessor&& other) noexcept;
  CaponProcessor& operator=(CaponProcessor&& other) noexcept;

  // =========================================================================
  // Public API — CPU data (upload → compute → download)
  // =========================================================================

  /**
   * @brief Вычислить рельеф Кейпона
   * @param signal    Y: матрица сигнала [n_channels × n_samples], column-major
   * @param steering  U: управляющие векторы [n_channels × n_directions], column-major
   * @param params    Параметры (n_channels, n_samples, n_directions, mu)
   * @return CaponReliefResult — M вещественных значений пространственного спектра
   */
  CaponReliefResult ComputeRelief(
      const std::vector<std::complex<float>>& signal,
      const std::vector<std::complex<float>>& steering,
      const CaponParams& params);

  /**
   * @brief Адаптивное диаграммообразование
   * @param signal    Y: матрица сигнала [n_channels × n_samples], column-major
   * @param steering  U: управляющие векторы [n_channels × n_directions], column-major
   * @param params    Параметры
   * @return CaponBeamResult — матрица [n_directions × n_samples]
   */
  CaponBeamResult AdaptiveBeamform(
      const std::vector<std::complex<float>>& signal,
      const std::vector<std::complex<float>>& steering,
      const CaponParams& params);

  // =========================================================================
  // Public API — GPU data (данные уже на устройстве)
  // =========================================================================

  /**
   * @brief Рельеф Кейпона (GPU входы)
   * @param gpu_signal   Y на GPU: complex<float>[n_channels × n_samples], column-major
   * @param gpu_steering U на GPU: complex<float>[n_channels × n_directions], column-major
   */
  CaponReliefResult ComputeRelief(
      void* gpu_signal,
      void* gpu_steering,
      const CaponParams& params);

  /**
   * @brief Адаптивное ДО (GPU входы)
   */
  CaponBeamResult AdaptiveBeamform(
      void* gpu_signal,
      void* gpu_steering,
      const CaponParams& params);

private:
  // =========================================================================
  // Internal pipeline
  // =========================================================================

  /// Скомпилировать kernels (ленивая одноразовая инициализация)
  void EnsureCompiled();

  /// Загрузить CPU данные сигнала в shared kSignal буфер
  void UploadSignal(const std::complex<float>* data, size_t count);

  /// Загрузить CPU управляющие векторы в shared kSteering буфер
  void UploadSteering(const std::complex<float>* data, size_t count);

  /// Скопировать GPU данные в shared kSignal (D2D)
  void CopySignalGpu(void* src, size_t count);

  /// Скопировать GPU управляющие векторы в shared kSteering (D2D)
  void CopySteeringGpu(void* src, size_t count);

  /**
   * @brief Шаги 1-3 общего пайплайна:
   *   CovarianceMatrixOp → регуляризация → CaponInvertOp → last_inv_
   */
  void RunCovAndInvert(const CaponParams& params);

  /**
   * @brief Шаг 4: W = R^{-1} * U  (ComputeWeightsOp → kWeight)
   * Вызывается после RunCovAndInvert, до Relief/Beam Op.
   */
  void RunComputeWeights(const CaponParams& params);

  /// Прочитать рельеф из kOutput (float[M])
  CaponReliefResult ReadReliefResult(uint32_t n_directions);

  /// Прочитать адаптивный выход из kOutput (complex<float>[M×N])
  CaponBeamResult ReadBeamResult(uint32_t n_directions, uint32_t n_samples);

  // ── Members ──────────────────────────────────────────────────────────────

  drv_gpu_lib::IBackend* backend_;  ///< Сохраняем для CaponInvertOp

  drv_gpu_lib::GpuContext ctx_;     ///< Per-module контекст (stream, kernels, shared bufs)

  // Op instances (Layer 5)
  CovarianceMatrixOp  cov_op_;     ///< R = (1/N)*Y*Y^H  (rocBLAS CGEMM)
  /// CaponInvertOp хранится в unique_ptr: CholeskyInverterROCm non-copyable,
  /// unique_ptr даёт корректный move constructor И move assignment без дополнительного кода.
  std::unique_ptr<CaponInvertOp> inv_op_;  ///< R^{-1} via CholeskyInverterROCm
  ComputeWeightsOp    weights_op_; ///< W = R^{-1}*U (→ kWeight shared buf)
  CaponReliefOp       relief_op_;  ///< z[m] = 1/Re(u^H * W[m])
  AdaptBeamformOp     beam_op_;    ///< Y_out = W^H * Y

  /// GEMM операции через vector_algebra::MatrixOpsROCm.
  /// Инициализируется с &ctx_ — handle привязан к ctx_.stream() (lazy init).
  vector_algebra::MatrixOpsROCm mat_ops_;

  /// Регуляризатор — Strategy (GoF): DiagonalLoadRegularizer по умолчанию.
  /// DIP: зависим от IMatrixRegularizer, не от конкретной реализации.
  std::unique_ptr<vector_algebra::IMatrixRegularizer> regularizer_;

  /// Результат инверсии — хранит R^{-1} на GPU между шагами пайплайна.
  /// Обновляется каждый раз в RunCovAndInvert().
  vector_algebra::CholeskyResult last_inv_;

  bool compiled_ = false;
};

}  // namespace capon

#else  // !ENABLE_ROCM — Windows stub

#include <linalg/capon_types.hpp>
#include <core/interface/i_backend.hpp>
#include <stdexcept>
#include <complex>
#include <vector>

namespace capon {

class CaponProcessor {
public:
  explicit CaponProcessor(drv_gpu_lib::IBackend*) {}
  ~CaponProcessor() = default;

  CaponProcessor(const CaponProcessor&) = delete;
  CaponProcessor& operator=(const CaponProcessor&) = delete;
  CaponProcessor(CaponProcessor&&) noexcept = default;
  CaponProcessor& operator=(CaponProcessor&&) noexcept = default;

  CaponReliefResult ComputeRelief(const std::vector<std::complex<float>>&,
      const std::vector<std::complex<float>>&, const CaponParams&) {
    throw std::runtime_error("CaponProcessor: ROCm not enabled");
  }
  CaponBeamResult AdaptiveBeamform(const std::vector<std::complex<float>>&,
      const std::vector<std::complex<float>>&, const CaponParams&) {
    throw std::runtime_error("CaponProcessor: ROCm not enabled");
  }
  CaponReliefResult ComputeRelief(void*, void*, const CaponParams&) {
    throw std::runtime_error("CaponProcessor: ROCm not enabled");
  }
  CaponBeamResult AdaptiveBeamform(void*, void*, const CaponParams&) {
    throw std::runtime_error("CaponProcessor: ROCm not enabled");
  }
};

}  // namespace capon

#endif  // ENABLE_ROCM
