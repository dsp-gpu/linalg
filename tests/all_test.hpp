#pragma once

/**
 * @file all_test.hpp
 * @brief Индексный файл всех тестов модуля linalg (capon + vector_algebra)
 *
 * main.cpp вызывает linalg_all_test::run() — НЕ отдельные тесты напрямую.
 * Включать/выключать тесты здесь.
 *
 * NOTE: linalg — ROCm-only модуль. Все тесты под #if ENABLE_ROCM.
 *
 * @author Кодо (AI Assistant)
 * @date 2026-04-15
 */

#if ENABLE_ROCM
// ─── Capon tests ───────────────────────────────────────────────────────────
#include "test_capon_rocm.hpp"
#include "test_capon_reference_data.hpp"
#include "test_capon_opencl_to_rocm.hpp"
#include "test_capon_hip_opencl_to_rocm.hpp"
#include "capon_benchmark.hpp"
#include "test_capon_benchmark_rocm.hpp"

// ─── vector_algebra tests (регрессия миграции: ранее не запускались) ──────
#include "test_cholesky_inverter_rocm.hpp"
#include "test_cross_backend_conversion.hpp"
#include "test_benchmark_symmetrize.hpp"
#include "test_stage_profiling.hpp"

// vector_algebra создаёт свой ROCmBackend внутри run() — так же как statistics.
#include <core/backends/rocm/rocm_core.hpp>
#include <core/backends/rocm/rocm_backend.hpp>
#include <core/services/console_output.hpp>
#endif

// ─── Capon suite (существовал ранее в этом же файле) ───────────────────────
namespace capon_all_test {

inline void run() {
#if ENABLE_ROCM
  test_capon_rocm::run();
  test_capon_reference_data::run();
  test_capon_opencl_to_rocm::run();         // OpenCL cl_mem → ZeroCopy → ROCm Capon
  test_capon_hip_opencl_to_rocm::run();     // hipMalloc → OpenCL writes → ROCm Capon
  // Benchmark (запускается только при is_prof=true в configGPU.json):
  // test_capon_benchmark_rocm::run();
#endif
}

}  // namespace capon_all_test

// ─── vector_algebra suite (ранее жил в src/vector_algebra/tests/all_test.hpp) ─
namespace vector_algebra_all_test {

inline void run() {
#if ENABLE_ROCM
  using namespace vector_algebra;
  using namespace vector_algebra::tests;

  auto& con = drv_gpu_lib::ConsoleOutput::GetInstance();

  int device_count = drv_gpu_lib::ROCmCore::GetAvailableDeviceCount();
  if (device_count == 0) {
    con.Print(0, "VecAlg", "[!] No ROCm devices found -- skipping vector_algebra tests");
    return;
  }

  // Create ROCm backend on device 0
  drv_gpu_lib::ROCmBackend rocm;
  rocm.Initialize(0);
  drv_gpu_lib::IBackend* backend = &rocm;

  con.Print(0, "VecAlg", "════════════════════════════════════════════════════════");
  con.Print(0, "VecAlg", "  vector_algebra tests (Task_11 v2)");
  con.Print(0, "VecAlg", "════════════════════════════════════════════════════════");

  // --- Утилиты (без GPU) ---
  TestResolveMatrixSize();

  // --- Core: Roundtrip mode ---
  con.Print(0, "VecAlg", "─── Roundtrip mode ───");
  TestCpuIdentity(backend, SymmetrizeMode::Roundtrip);
  TestCpu341(backend, SymmetrizeMode::Roundtrip);
  TestGpuVoidPtr341(backend, SymmetrizeMode::Roundtrip);
  TestZeroCopyClMem(backend, SymmetrizeMode::Roundtrip);
  TestBatchCpu_4x64(backend, SymmetrizeMode::Roundtrip);
  TestBatchGpu_4x64(backend, SymmetrizeMode::Roundtrip);
  TestBatchSizes(backend, SymmetrizeMode::Roundtrip);
  TestMatrixSizes(backend, SymmetrizeMode::Roundtrip);
  TestResultAccess(backend, SymmetrizeMode::Roundtrip);

  // --- Core: GpuKernel mode ---
  con.Print(0, "VecAlg", "─── GpuKernel mode ───");
  TestCpuIdentity(backend, SymmetrizeMode::GpuKernel);
  TestCpu341(backend, SymmetrizeMode::GpuKernel);
  TestGpuVoidPtr341(backend, SymmetrizeMode::GpuKernel);
  TestZeroCopyClMem(backend, SymmetrizeMode::GpuKernel);
  TestBatchCpu_4x64(backend, SymmetrizeMode::GpuKernel);
  TestBatchGpu_4x64(backend, SymmetrizeMode::GpuKernel);
  TestBatchSizes(backend, SymmetrizeMode::GpuKernel);
  TestMatrixSizes(backend, SymmetrizeMode::GpuKernel);
  TestResultAccess(backend, SymmetrizeMode::GpuKernel);

  // --- Cross-backend 85×85 ---
  con.Print(0, "VecAlg", "─── Cross-backend 85×85 ───");
  TestConvert_VectorInput(backend, SymmetrizeMode::Roundtrip);
  TestConvert_VectorInput(backend, SymmetrizeMode::GpuKernel);
  TestConvert_HipInput(backend, SymmetrizeMode::Roundtrip);
  TestConvert_HipInput(backend, SymmetrizeMode::GpuKernel);
  TestConvert_ClMemInput(backend, SymmetrizeMode::Roundtrip);
  TestConvert_ClMemInput(backend, SymmetrizeMode::GpuKernel);
  TestConvert_OutputFormats(backend, SymmetrizeMode::Roundtrip);
  TestConvert_OutputFormats(backend, SymmetrizeMode::GpuKernel);

  // --- Stage Profiling (Task_12) ---
  TestStageProfiling(backend);

  // --- Profiler ---
  TestProfilerIntegration(backend);

  // --- Benchmark (hipEvent GPU timing, MD report) ---
  RunComprehensiveBenchmark(backend);

  rocm.Cleanup();

  con.Print(0, "VecAlg", "════════════════════════════════════════════════════════");
  con.Print(0, "VecAlg", "  vector_algebra: ALL TESTS PASSED");
  con.Print(0, "VecAlg", "════════════════════════════════════════════════════════");
#endif  // ENABLE_ROCM
}

}  // namespace vector_algebra_all_test

// ─── Единая точка входа ────────────────────────────────────────────────────
namespace linalg_all_test {

inline void run() {
#if ENABLE_ROCM
  // 1. Capon suite
  capon_all_test::run();
  // 2. vector_algebra suite (создаёт свой ROCmBackend)
  vector_algebra_all_test::run();
#endif
}

}  // namespace linalg_all_test
