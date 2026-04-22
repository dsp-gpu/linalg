/**
 * @file diagonal_load_regularizer.cpp
 * @brief DiagonalLoadRegularizer — реализация через GpuContext (v2 disk cache)
 *
 * Phase C3 of kernel_cache_v2: manual hiprtc removed.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-03-16  (migrated 2026-04-22 to GpuContext)
 */

#if ENABLE_ROCM

#include <linalg/diagonal_load_regularizer.hpp>
#include <linalg/kernels/diagonal_load_kernel_rocm.hpp>
#include <core/interface/gpu_context.hpp>
#include <core/services/cache_dir_resolver.hpp>

#include <stdexcept>
#include <string>
#include <utility>

namespace vector_algebra {

// ════════════════════════════════════════════════════════════════════════════
// Constructor / Destructor / Move
// ════════════════════════════════════════════════════════════════════════════

DiagonalLoadRegularizer::DiagonalLoadRegularizer(drv_gpu_lib::IBackend* backend) {
  if (!backend || !backend->IsInitialized()) {
    throw std::runtime_error(
        "DiagonalLoadRegularizer: backend is null or not initialized");
  }
  stream_ = static_cast<hipStream_t>(backend->GetNativeQueue());

  // GpuContext v2 — кеш по CompileKey (source + defines + arch + hiprtc_ver).
  // cache_dir shared с другими vector_algebra модулями.
  ctx_ = std::make_unique<drv_gpu_lib::GpuContext>(
      backend, "DiagLoad",
      drv_gpu_lib::ResolveCacheDir("vector_algebra"));

  const char* source = kernels::GetDiagonalLoadKernelSource();
  ctx_->CompileModule(source, {"diagonal_load"}, /*extra_defines=*/{});
  function_ = static_cast<void*>(ctx_->GetKernel("diagonal_load"));
}

DiagonalLoadRegularizer::~DiagonalLoadRegularizer() = default;

DiagonalLoadRegularizer::DiagonalLoadRegularizer(
    DiagonalLoadRegularizer&& other) noexcept
    : stream_(other.stream_)
    , ctx_(std::move(other.ctx_))
    , function_(other.function_) {
  other.stream_   = nullptr;
  other.function_ = nullptr;
}

DiagonalLoadRegularizer& DiagonalLoadRegularizer::operator=(
    DiagonalLoadRegularizer&& other) noexcept {
  if (this != &other) {
    stream_   = other.stream_;
    ctx_      = std::move(other.ctx_);
    function_ = other.function_;
    other.stream_   = nullptr;
    other.function_ = nullptr;
  }
  return *this;
}

// ════════════════════════════════════════════════════════════════════════════
// Apply — запуск kernel: A[i,i] += mu
// ════════════════════════════════════════════════════════════════════════════

void DiagonalLoadRegularizer::Apply(void* d_matrix, int n, float mu,
                                    hipStream_t stream) {
  if (mu == 0.0f) return;  // no-op: нечего прибавлять

  hipStream_t target_stream = (stream != nullptr) ? stream : stream_;

  auto func = static_cast<hipFunction_t>(function_);
  auto un   = static_cast<unsigned int>(n);

  void* args[] = { &d_matrix, &mu, &un };
  hipError_t err = hipModuleLaunchKernel(
      func,
      (un + 255u) / 256u, 1, 1,  // grid
      256, 1, 1,                  // block
      0, target_stream,
      args, nullptr);

  if (err != hipSuccess) {
    throw std::runtime_error(
        "DiagonalLoadRegularizer::Apply: kernel launch failed: " +
        std::string(hipGetErrorString(err)));
  }
}

}  // namespace vector_algebra

#endif  // ENABLE_ROCM
