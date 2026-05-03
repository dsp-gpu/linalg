#pragma once

// ============================================================================
// CaponReliefOp — рельеф Кейпона z[m] = 1/Re(u_m^H·W[:,m]) (Layer 5 Ref03)
//
// ЧТО:    Concrete Op (наследник GpuKernelOp): финальный шаг pipeline для
//         режима «angular power spectrum» Capon-MVDR. Запускает один HIP
//         kernel `compute_capon_relief`, который для каждого направления m
//         вычисляет:
//           acc[m] = Σ_{p=0..P-1} Re(conj(U[p,m]) · W[p,m])
//           z[m]   = 1 / acc[m]   (защита от ноля → 0)
//         где W = R^{-1}·U уже посчитана ComputeWeightsOp в kWeight.
//         Результат z[M] — пространственный спектр P(θ_m).
//
// ЗАЧЕМ:  В отличие от AdaptBeamformOp (CGEMM Y_out=W^H·Y) здесь нужен НЕ
//         полный CGEMM, а только диагональ Re(U^H·W) — поэтому свой HIP
//         kernel вместо rocBLAS gemm + extract_diag (gemm × diag = памяти и
//         лишний pass). Один kernel, один thread на направление m, цикл по
//         P каналам — оптимально для типичных P=4..32, M=181.
//
// ПОЧЕМУ: - Layer 5 Ref03: один Op = один логический шаг (custom HIP kernel).
//         - Stateless (нет приватных GPU-буферов, OnRelease пустой) — все
//           буферы в shared kSteering/kWeight/kOutput через ctx_.
//         - Custom kernel вместо rocBLAS — экономия: gemm даёт U^H·W [M×M],
//           но нужна только диагональ [M], то есть 99% работы выкидывается.
//         - column-major (как rocBLAS): U[P×M], W[P×M], индекс [p,m] = m·P+p.
//         - Block 256, grid (M+255)/256 — стандарт для линейных задач,
//           wavefront=64 на RDNA даёт 4 wavefront/block (хорошая occupancy).
//         - Защита (acc>0)?1/acc:0 — при корректной регуляризации μ>0
//           матрица R положительно определена, acc>0 всегда; защита для
//           случая μ=0 + сингулярная R (нерегулярный вход).
//
// Использование:
//   // Внутри CaponProcessor::ComputeRelief после ComputeWeightsOp:
//   relief_op_.Execute(P, M);
//   // → kOutput содержит float[M], читается ReadReliefResult().
//
// История:
//   - Создан:  2026-03-16 (Ref03 Layer 5, Capon pipeline)
//   - Изменён: 2026-05-01 (унификация формата шапки под dsp-asst RAG-индексер)
// ============================================================================

#if ENABLE_ROCM

#include <core/services/gpu_kernel_op.hpp>
#include <core/interface/gpu_context.hpp>
#include <linalg/capon_types.hpp>

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>

namespace capon {

/**
 * @class CaponReliefOp
 * @brief Layer 5 Ref03 Op: HIP kernel z[m] = 1/Re(u_m^H·W[:,m]) — рельеф Кейпона.
 *
 * @note Stateless (нет приватных буферов). Все данные в shared-буферах ctx_.
 * @note Требует #if ENABLE_ROCM. Кернел compute_capon_relief — в capon_kernels_rocm.hpp.
 * @note Предусловие: ComputeWeightsOp::Execute() уже записал W в kWeight.
 * @see ComputeWeightsOp — поставщик kWeight
 * @see AdaptBeamformOp — параллельный финальный Op (для ДО вместо рельефа)
 * @see capon::kernels::GetCaponKernelSource
 */
class CaponReliefOp : public drv_gpu_lib::GpuKernelOp {
public:
  /**
   * @brief Возвращает имя Op'а для логирования и профилирования.
   *
   * @return C-строка "CaponRelief" (статический литерал).
   *   @test_check std::string(result) == "CaponRelief"
   */
  const char* Name() const override { return "CaponRelief"; }

  /**
   * @brief Вычислить рельеф Кейпона
   * @param n_channels   P — число каналов
   *   @test { range=[1..50000], value=128, unit="лучей/каналов" }
   * @param n_directions M — число направлений
   *
   * Читает: kSteering (U) [P × M],  kWeight (W = R^{-1}*U) [P × M]
   * Пишет:  kOutput (float[M])
   *
   * Предусловие: ComputeWeightsOp::Execute() уже запущен и записал W в kWeight.
   * @throws std::runtime_error при сбое hipModuleLaunchKernel("compute_capon_relief").
   *   @test_check throws on hipModuleLaunchKernel != hipSuccess
   */
  void Execute(uint32_t n_channels, uint32_t n_directions) {
    uint32_t P = n_channels;
    uint32_t M = n_directions;

    void* U   = ctx_->GetShared(shared_buf::kSteering);
    void* W   = ctx_->GetShared(shared_buf::kWeight);
    void* out = ctx_->RequireShared(shared_buf::kOutput,
                                    static_cast<size_t>(M) * sizeof(float));

    // z[m] = 1 / Re(Σ_p conj(U[p,m]) * W[p,m])
    // W на том же stream (ComputeWeightsOp вызывает rocblas_set_stream(blas, stream()))
    void* args[] = { &U, &W, &out, &P, &M };
    hipError_t err = hipModuleLaunchKernel(
        kernel("compute_capon_relief"),
        (M + 255) / 256, 1, 1,
        256, 1, 1,
        0, stream(),
        args, nullptr);
    if (err != hipSuccess) {
      throw std::runtime_error("CaponReliefOp compute_capon_relief: " +
                               std::string(hipGetErrorString(err)));
    }
  }

protected:
  void OnRelease() override {}  // нет приватных GPU буферов
};

}  // namespace capon

#endif  // ENABLE_ROCM
