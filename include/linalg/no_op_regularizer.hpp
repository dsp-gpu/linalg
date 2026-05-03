#pragma once

// ============================================================================
// NoOpRegularizer — регуляризатор-заглушка (Null Object Pattern, GoF)
//
// ЧТО:    Реализация IMatrixRegularizer, чей Apply() ничего не делает.
//         Матрица не изменяется, kernel не запускается. Безопасная замена
//         nullptr там, где регуляризатор опционален (CaponProcessor хранит
//         unique_ptr<IMatrixRegularizer>, проверка на nullptr была бы шумом).
//
// ЗАЧЕМ:  - mu = 0 / матрица гарантированно HPD (искусственные тесты,
//           эталонные signal generators) — регуляризация не нужна.
//         - Тестирование pipeline без реального GPU (на non-ROCm сборках
//           hipStream_t заменён на void*, метод тривиально пуст).
//         - Default-значение в фасаде: всегда есть валидный регуляризатор,
//           вызовы reg->Apply(...) не требуют if-чека.
//
// ПОЧЕМУ: - Null Object Pattern: вместо «if reg → reg->Apply()» —
//           «reg->Apply()» всегда. Меньше веток в горячем пути.
//         - Header-only inline impl — не нужен .cpp, no-op обнуляется
//           компилятором.
//         - Не наследует от RAII-классов: нет ресурсов → нет деструктора.
//
// Использование:
//   // Default в фасаде, mu = 0 в тестах:
//   std::unique_ptr<IMatrixRegularizer> reg =
//       std::make_unique<NoOpRegularizer>();
//   reg->Apply(d_R, n, 0.0f);   // ничего не происходит
//
// История:
//   - Создан:  2026-03-16 (Null Object для Strategy chain Capon)
//   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
// ============================================================================

#include <linalg/i_matrix_regularizer.hpp>

namespace vector_algebra {

/**
 * @class NoOpRegularizer
 * @brief Null Object — безопасный no-op вместо nullptr-регуляризатора.
 *
 * @note Не имеет state, deterministically inline — zero overhead.
 * @note Работает на всех платформах (non-ROCm hipStream_t = void*).
 * @see IMatrixRegularizer
 * @see DiagonalLoadRegularizer (полноценная альтернатива)
 */
class NoOpRegularizer : public IMatrixRegularizer {
public:
  /**
   * @brief Null Object: ничего не делает (zero overhead, kernel не запускается).
   *
   */
  void Apply(void* /*d_matrix*/, int /*n*/, float /*mu*/,
             hipStream_t /*stream*/ = nullptr) override {}
};

}  // namespace vector_algebra
