/**
 * @file symmetrize_gpu_rocm.cpp
 * @brief CompileKernels + SymmetrizeGpuKernel via GpuContext (v2 disk cache)
 *
 * Реализация GPU-пути симметризации: компиляция HIP kernel через
 * GpuContext::CompileModule (clean-slate v2 cache), запуск через
 * hipModuleLaunchKernel.
 *
 * Phase C2 of kernel_cache_v2: removed manual hiprtc, own cache path.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-02-26  (migrated 2026-04-22 to GpuContext)
 */

#if ENABLE_ROCM

#include <linalg/cholesky_inverter_rocm.hpp>
#include <linalg/kernels/symmetrize_kernel_sources_rocm.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>

#include <hip/hip_runtime.h>

#include <core/interface/gpu_context.hpp>
#include <core/services/console_output.hpp>

namespace vector_algebra {

// ════════════════════════════════════════════════════════════════════════════
// CompileKernels — via GpuContext (idempotent, disk-cached v2)
// ════════════════════════════════════════════════════════════════════════════

void CholeskyInverterROCm::CompileKernels() {
  if (sym_kernel_ != nullptr) return;  // idempotent — kernel already cached
  if (!ctx_) {
    throw std::runtime_error(
        "CompileKernels: GpuContext not initialised (ctor failed?)");
  }

  const char* src = kernels::GetSymmetrizeKernelSource();
  ctx_->CompileModule(src,
                      {"symmetrize_upper_to_full"},
                      /*extra_defines=*/{});
  sym_kernel_ = static_cast<void*>(ctx_->GetKernel("symmetrize_upper_to_full"));

  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();
  con.Print(0, "VecAlg", "symmetrize kernel ready (GpuContext v2 cache)");
}

// ════════════════════════════════════════════════════════════════════════════
// SymmetrizeGpuKernel — single matrix
// ════════════════════════════════════════════════════════════════════════════

void CholeskyInverterROCm::SymmetrizeGpuKernel(void* d_matrix, int n,
                                                 void* stream) {
  CompileKernels();

  unsigned int un = static_cast<unsigned int>(n);
  dim3 block(16, 16);
  dim3 grid((un + 15) / 16, (un + 15) / 16);

  void* args[] = {&d_matrix, &un};

  hipError_t err = hipModuleLaunchKernel(
      static_cast<hipFunction_t>(sym_kernel_),
      grid.x, grid.y, 1,    // grid dimensions
      block.x, block.y, 1,  // block dimensions
      0,                     // shared memory
      static_cast<hipStream_t>(stream),
      args, nullptr);

  if (err != hipSuccess) {
    throw std::runtime_error(
        "SymmetrizeGpuKernel: launch failed: " +
        std::string(hipGetErrorString(err)));
  }
}

// ════════════════════════════════════════════════════════════════════════════
// SymmetrizeGpuKernelBatched — цикл по матрицам
// ════════════════════════════════════════════════════════════════════════════

void CholeskyInverterROCm::SymmetrizeGpuKernelBatched(void* d_contiguous,
                                                        int n, int batch,
                                                        void* stream) {
  const size_t one_bytes =
      static_cast<size_t>(n) * n * sizeof(std::complex<float>);
  auto* base = static_cast<char*>(d_contiguous);

  for (int k = 0; k < batch; ++k) {
    void* ptr_k = base + static_cast<size_t>(k) * one_bytes;
    SymmetrizeGpuKernel(ptr_k, n, stream);
  }
}

}  // namespace vector_algebra

#endif  // ENABLE_ROCM
