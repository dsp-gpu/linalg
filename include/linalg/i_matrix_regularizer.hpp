#pragma once

// ============================================================================
// IMatrixRegularizer — интерфейс матричной регуляризации (Strategy GoF)
//
// ЧТО:    Pure-virtual интерфейс с единственным методом Apply(d_matrix, n,
//         mu, stream). Применяется in-place к квадратной комплексной матрице
//         на GPU перед инверсией. Конкретный смысл коэффициента mu задаёт
//         реализация: DiagonalLoadRegularizer добавляет mu·I, NoOpRegularizer
//         не делает ничего (Null Object).
//
// ЗАЧЕМ:  Capon (MVDR) и любой adaptive-pipeline сталкивается с ill-
//         conditioned ковариационной матрицей (близкая к сингулярной).
//         Регуляризация перед Cholesky/POTRI критична для устойчивости
//         результата. Через интерфейс CaponProcessor хранит
//         std::unique_ptr<IMatrixRegularizer> и не зависит от конкретного
//         алгоритма — можно подключить tapering, shrinkage, diagonal load
//         без правки фасада.
//
// ПОЧЕМУ: - ISP: один метод Apply — минимальный интерфейс. Никаких
//           Initialize/Release: регуляризатор сам управляет state в ctor/dtor.
//         - DIP: потребители (CholeskyInverterROCm wrappers, CaponProcessor)
//           зависят от IMatrixRegularizer*, не от concrete-типа.
//         - OCP: новые алгоритмы (TaperingRegularizer, ShrinkageRegularizer)
//           — отдельные классы, существующий код не меняется.
//         - hipStream_t явный параметр (с default nullptr) — чтобы
//           гарантировать порядок исполнения после rocBLAS CGEMM на том же
//           stream'е (без явного hipStreamSynchronize).
//         - На non-ROCm платформах hipStream_t заменяется на void* — для
//           NoOpRegularizer (CPU-тесты pipeline без GPU).
//
// Использование:
//   std::unique_ptr<IMatrixRegularizer> reg =
//       std::make_unique<DiagonalLoadRegularizer>(backend);
//   reg->Apply(d_R, n, 0.01f, ctx.stream());
//   inverter.Invert(InputData<void*>{d_R}, n);
//
// История:
//   - Создан:  2026-03-16 (Strategy для Capon: NoOp + DiagonalLoad)
//   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
// ============================================================================

#if ENABLE_ROCM
#include <hip/hip_runtime.h>
#else
using hipStream_t = void*;  ///< Stub для non-ROCm платформ (NoOpRegularizer)
#endif

namespace vector_algebra {

/**
 * @class IMatrixRegularizer
 * @brief Pure-virtual Strategy: регуляризация квадратной complex-матрицы на GPU.
 *
 * @note Pure interface — нельзя инстанцировать. Метод Apply обязателен.
 * @note Применяется in-place перед инверсией (Cholesky/POTRI).
 * @note Семантика mu — в реализации (mu·I для DiagonalLoad, n/a для NoOp).
 * @see NoOpRegularizer (Null Object)
 * @see DiagonalLoadRegularizer (concrete strategy: A += mu·I)
 * @see CholeskyInverterROCm (типичный consumer после Apply)
 */
class IMatrixRegularizer {
public:
  virtual ~IMatrixRegularizer() = default;

  /**
   * @brief Применить регуляризацию к матрице (GPU, in-place).
   *
   * @param d_matrix  GPU-указатель на квадратную матрицу n×n
   *   @test { pattern=gpu_pointer, values=["valid_alloc", nullptr] }
   *                  (complex<float>, column-major)
   * @param n         Размер матрицы (строки = столбцы = n)
   *   @test { range=[512..1300000], value=8192 }
   * @param mu        Коэффициент регуляризации (семантика — в реализации)
   * @param stream    HIP stream для запуска kernel (nullptr → использовать
   *                  внутренний stream регуляризатора из backend).
   *                  Передавайте ctx.stream() чтобы гарантировать порядок
   *                  выполнения после предшествующих CGEMM/kernel операций
   *                  на том же stream.
   */
  virtual void Apply(void* d_matrix, int n, float mu,
                     hipStream_t stream = nullptr) = 0;
};

}  // namespace vector_algebra
