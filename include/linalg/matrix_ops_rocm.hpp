#pragma once

// ============================================================================
// MatrixOpsROCm — фасад rocBLAS CGEMM для complex<float> матриц (Layer 6 Ref03)
//
// ЧТО:    Тонкая обёртка над rocBLAS CGEMM (комплексное матричное умножение).
//         Предоставляет именованные shortcut'ы под типовые паттерны сигнальной
//         обработки: CovarianceMatrix (R = Y·Y^H/N), Multiply (C = A·B),
//         MultiplyConjTransA (C = A^H·B) — плюс общий CGEMM для произвольных
//         транспозиций. Все матрицы — complex<float>, column-major.
//
// ЗАЧЕМ:  Capon, statistics и другие модули верхнего уровня не должны тянуть
//         rocblas.h в свои public headers (зависимость на конкретный BLAS-
//         бэкенд). MatrixOpsROCm централизует все CGEMM-вызовы за единым API,
//         оставляя возможность подменить реализацию (HybridBackend, future
//         CPU-fallback) без правки потребителей.
//
// ПОЧЕМУ: - Layer 6 Facade Ref03 — координирует rocBLAS, не делает kernel-
//           launch'и. Делегирует rocblas_cgemm через handle из GpuContext.
//         - rocBLAS handle берём из GpuContext (lazy init, уже привязан к
//           stream_) → не нужно вручную rocblas_set_stream, синхронизация
//           гарантирована Layer 1.
//         - Non-owning ctx_ — MatrixOpsROCm не владеет GpuContext, только
//           использует. Move разрешён (передача указателя), copy запрещён
//           (логика владения должна быть явной у вызывающего).
//         - Именованные методы (CovarianceMatrix / MultiplyConjTransA) над
//           generic CGEMM — самодокументируемый код в capon без магических
//           rocblas_operation констант на каждом вызове.
//
// Использование:
//   drv_gpu_lib::GpuContext ctx{backend};
//   MatrixOpsROCm mat_ops(&ctx);
//   mat_ops.CovarianceMatrix(d_Y, P, N, d_R);   // R = (1/N) Y Y^H
//   mat_ops.Multiply(d_R_inv, d_U, d_W, P, M, P);
//
// История:
//   - Создан:  2026-03-16 (вынос rocBLAS из capon/statistics в общий фасад)
//   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
// ============================================================================

#if ENABLE_ROCM

#include <core/interface/gpu_context.hpp>

#include <rocblas/rocblas.h>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace vector_algebra {

/**
 * @class MatrixOpsROCm
 * @brief Фасад rocBLAS CGEMM, привязанный к GpuContext (stream + handle).
 *
 * @note Move-only: copy запрещён, move передаёт non-owning указатель ctx_.
 * @note Не владеет GpuContext — ctx должен жить дольше MatrixOpsROCm.
 * @note Требует #if ENABLE_ROCM. На non-ROCm — stub с пустыми методами.
 * @see drv_gpu_lib::GpuContext (Layer 1 — lazy rocBLAS handle)
 * @see CaponProcessor (главный потребитель именованных shortcut'ов)
 */
class MatrixOpsROCm {
public:
  // =========================================================================
  // Constructor
  // =========================================================================

  MatrixOpsROCm() = default;

  /**
   * @param ctx GpuContext модуля (non-owning). Должен жить дольше объекта.
   *            ctx->GetRocblasHandleRaw() возвращает handle, привязанный
   *            к ctx->stream() — синхронизация гарантирована GpuContext.
   */
  explicit MatrixOpsROCm(drv_gpu_lib::GpuContext* ctx) : ctx_(ctx) {}

  // No copy
  MatrixOpsROCm(const MatrixOpsROCm&)            = delete;
  MatrixOpsROCm& operator=(const MatrixOpsROCm&) = delete;

  // Move
  MatrixOpsROCm(MatrixOpsROCm&& o) noexcept : ctx_(o.ctx_) { o.ctx_ = nullptr; }
  MatrixOpsROCm& operator=(MatrixOpsROCm&& o) noexcept {
    ctx_ = o.ctx_; o.ctx_ = nullptr; return *this;
  }

  // =========================================================================
  // Named shortcuts — конкретные паттерны для сигнальной обработки
  // =========================================================================

  /**
   * @brief Ковариационная матрица: R = (1/N) * Y * Y^H
   *
   * @param Y   [P × N] complex<float>, column-major
   * @param P   число строк (каналы)
   * @param N   число столбцов (отсчёты)
   * @param R   [P × P] выход, complex<float>, column-major
   *
   * rocBLAS: C[P×P] = alpha * Y[P×N] * Y^H[N×P]
   *   opA=None, opB=ConjTrans, alpha=1/N, beta=0
   */
  void CovarianceMatrix(const void* Y, int P, int N, void* R);

  /**
   * @brief Матричное умножение: C = A * B  (NoTrans × NoTrans)
   *
   * @param A   [m × k] complex<float>, column-major
   * @param B   [k × n] complex<float>, column-major
   * @param C   [m × n] выход, complex<float>, column-major
   * @param m   строки A и C
   * @param n   столбцы B и C
   * @param k   столбцы A = строки B
   *
   * Применения в capon:
   *   W = R^{-1} * U  →  Multiply(R_inv, U, W, P, M, P)
   */
  void Multiply(const void* A, const void* B, void* C, int m, int n, int k);

  /**
   * @brief Умножение с эрмитовым транспонированием первого аргумента: C = A^H * B
   *
   * A хранится как [k × m] column-major (до транспонирования).
   * После ConjTrans: A^H = [m × k].
   *
   * @param A   [k × m] complex<float>, column-major (хранится транспонированным)
   * @param B   [k × n] complex<float>, column-major
   * @param C   [m × n] выход, complex<float>, column-major
   * @param m   строки C  (= столбцы A до транспонирования)
   * @param n   столбцы C (= столбцы B)
   * @param k   строки A и B (= contraction dimension)
   *
   * Применения в capon:
   *   Y_out = W^H * Y  →  MultiplyConjTransA(W, Y, Y_out, M, N, P)
   *     W[P×M] хранится как [k=P × m=M] → lda=P
   */
  void MultiplyConjTransA(const void* A, const void* B, void* C,
                           int m, int n, int k);

  // =========================================================================
  // General CGEMM — для произвольных транспозиций
  // =========================================================================

  /**
   * @brief Общий CGEMM: C = alpha * op(A) * op(B) + beta * C
   *
   * @param transA  rocblas_operation_none / rocblas_operation_conjugate_transpose
   * @param transB  rocblas_operation_none / rocblas_operation_conjugate_transpose
   * @param m       строки op(A) и C
   * @param n       столбцы op(B) и C
   * @param k       столбцы op(A) = строки op(B)
   * @param alpha   скаляр alpha (complex<float>)
   * @param A       матрица A на GPU
   * @param lda     leading dimension A
   * @param B       матрица B на GPU
   * @param ldb     leading dimension B
   * @param beta    скаляр beta (complex<float>)
   * @param C       матрица C на GPU (вход/выход)
   * @param ldc     leading dimension C
   *
   * @throws std::runtime_error при ошибке rocBLAS
   */
  void CGEMM(rocblas_operation transA, rocblas_operation transB,
             int m, int n, int k,
             const rocblas_float_complex* alpha,
             const void* A, int lda,
             const void* B, int ldb,
             const rocblas_float_complex* beta,
             void* C, int ldc);

private:
  drv_gpu_lib::GpuContext* ctx_ = nullptr;

  /// Получить rocBLAS handle (lazy init в GpuContext, уже привязан к stream)
  rocblas_handle blas() const {
    return static_cast<rocblas_handle>(ctx_->GetRocblasHandleRaw());
  }
};

}  // namespace vector_algebra

#else  // !ENABLE_ROCM — Windows stub

namespace vector_algebra {
class MatrixOpsROCm {
public:
  MatrixOpsROCm() = default;
  void CovarianceMatrix(const void*, int, int, void*) {}
  void Multiply(const void*, const void*, void*, int, int, int) {}
  void MultiplyConjTransA(const void*, const void*, void*, int, int, int) {}
};
}  // namespace vector_algebra

#endif  // ENABLE_ROCM
