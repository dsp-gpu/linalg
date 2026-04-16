#pragma once

/**
 * @file py_capon_rocm.hpp
 * @brief Python wrapper for CaponProcessor (MVDR beamformer, ROCm).
 *
 * Экспорт в Python:
 *   import dsp_linalg
 *   params = dsp_linalg.CaponParams(n_channels=85, n_samples=1000, n_directions=181, mu=1e-3)
 *   cap = dsp_linalg.CaponProcessor(backend)
 *   relief = cap.compute_relief(signal, steering, params)        # np.float32[M]
 *   beam   = cap.adaptive_beamform(signal, steering, params)     # np.complex64[M, N]
 *
 * Include AFTER py_helpers.hpp (py, vector_to_numpy, vector_to_numpy_2d).
 *
 * @author Кодо (AI Assistant)
 * @date 2026-04-15
 */

#if ENABLE_ROCM

#include <linalg/capon_processor.hpp>
#include <linalg/capon_types.hpp>
#include <core/interface/i_backend.hpp>

// ════════════════════════════════════════════════════════════════════════════
// PyCaponProcessor — Python wrapper
// ════════════════════════════════════════════════════════════════════════════

// CaponProcessor владеет GPU ресурсами (ctx_, rocblas handle, CholeskyResult).
// В Python передаём ROCmGPUContext& (py_gpu_context.hpp) — backend берётся
// из ctx.backend(). Единый паттерн для всех модулей DSP-GPU.
class PyCaponProcessor {
public:
  explicit PyCaponProcessor(ROCmGPUContext& ctx)
      : ctx_(ctx), processor_(ctx.backend()) {}
private:
  ROCmGPUContext& ctx_;  // prevent GC from collecting context before processor
public:

  // ── CPU API (H2D → GPU → D2H) ─────────────────────────────────────────────
  py::array_t<float> compute_relief(
      py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> signal_flat,
      py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> steering_flat,
      const capon::CaponParams& params) {
    auto sig_buf = signal_flat.request();
    auto st_buf  = steering_flat.request();

    std::vector<std::complex<float>> signal(
        static_cast<std::complex<float>*>(sig_buf.ptr),
        static_cast<std::complex<float>*>(sig_buf.ptr) + sig_buf.size);
    std::vector<std::complex<float>> steering(
        static_cast<std::complex<float>*>(st_buf.ptr),
        static_cast<std::complex<float>*>(st_buf.ptr) + st_buf.size);

    capon::CaponReliefResult result;
    {
      py::gil_scoped_release release;
      result = processor_.ComputeRelief(signal, steering, params);
    }
    return vector_to_numpy<float>(std::move(result.relief));
  }

  py::array_t<std::complex<float>> adaptive_beamform(
      py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> signal_flat,
      py::array_t<std::complex<float>, py::array::c_style | py::array::forcecast> steering_flat,
      const capon::CaponParams& params) {
    auto sig_buf = signal_flat.request();
    auto st_buf  = steering_flat.request();

    std::vector<std::complex<float>> signal(
        static_cast<std::complex<float>*>(sig_buf.ptr),
        static_cast<std::complex<float>*>(sig_buf.ptr) + sig_buf.size);
    std::vector<std::complex<float>> steering(
        static_cast<std::complex<float>*>(st_buf.ptr),
        static_cast<std::complex<float>*>(st_buf.ptr) + st_buf.size);

    capon::CaponBeamResult result;
    {
      py::gil_scoped_release release;
      result = processor_.AdaptiveBeamform(signal, steering, params);
    }
    return vector_to_numpy_2d<std::complex<float>>(
        std::move(result.output),
        static_cast<size_t>(result.n_directions),
        static_cast<size_t>(result.n_samples));
  }

  // ── GPU-pointer API (ZeroCopy из OpenCL/ROCm без H2D) ─────────────────────
  // signal_ptr / steering_ptr — int64 (uintptr_t) от Python, интерпретируется
  // как void* на GPU memory. Используется для real-time pipeline (сопровождение цели).
  py::array_t<float> compute_relief_gpu(
      uintptr_t signal_ptr, uintptr_t steering_ptr,
      const capon::CaponParams& params) {
    capon::CaponReliefResult result;
    {
      py::gil_scoped_release release;
      result = processor_.ComputeRelief(
          reinterpret_cast<void*>(signal_ptr),
          reinterpret_cast<void*>(steering_ptr),
          params);
    }
    return vector_to_numpy<float>(std::move(result.relief));
  }

  py::array_t<std::complex<float>> adaptive_beamform_gpu(
      uintptr_t signal_ptr, uintptr_t steering_ptr,
      const capon::CaponParams& params) {
    capon::CaponBeamResult result;
    {
      py::gil_scoped_release release;
      result = processor_.AdaptiveBeamform(
          reinterpret_cast<void*>(signal_ptr),
          reinterpret_cast<void*>(steering_ptr),
          params);
    }
    return vector_to_numpy_2d<std::complex<float>>(
        std::move(result.output),
        static_cast<size_t>(result.n_directions),
        static_cast<size_t>(result.n_samples));
  }

private:
  capon::CaponProcessor processor_;
};

// ════════════════════════════════════════════════════════════════════════════
// Binding registration
// ════════════════════════════════════════════════════════════════════════════

inline void register_capon_processor(py::module& m) {
  // CaponParams — POD struct
  py::class_<capon::CaponParams>(m, "CaponParams")
      .def(py::init<>())
      .def(py::init([](uint32_t p, uint32_t n, uint32_t dir, float mu) {
             capon::CaponParams x;
             x.n_channels = p; x.n_samples = n; x.n_directions = dir; x.mu = mu;
             return x;
           }),
           py::arg("n_channels"), py::arg("n_samples"),
           py::arg("n_directions"), py::arg("mu") = 0.0f)
      .def_readwrite("n_channels",   &capon::CaponParams::n_channels)
      .def_readwrite("n_samples",    &capon::CaponParams::n_samples)
      .def_readwrite("n_directions", &capon::CaponParams::n_directions)
      .def_readwrite("mu",           &capon::CaponParams::mu);

  // CaponProcessor wrapper
  py::class_<PyCaponProcessor>(m, "CaponProcessor",
      "MVDR (Capon) beamformer — full GPU pipeline.\n\n"
      "Usage:\n"
      "  ctx = dsp_linalg.ROCmGPUContext(0)\n"
      "  cap = dsp_linalg.CaponProcessor(ctx)\n"
      "  relief = cap.compute_relief(signal, steering, params)\n")
      .def(py::init<ROCmGPUContext&>(),
           py::arg("ctx"),
           py::keep_alive<1, 2>())  // processor держит ссылку на context
      .def("compute_relief",      &PyCaponProcessor::compute_relief,
           py::arg("signal"), py::arg("steering"), py::arg("params"))
      .def("adaptive_beamform",   &PyCaponProcessor::adaptive_beamform,
           py::arg("signal"), py::arg("steering"), py::arg("params"))
      .def("compute_relief_gpu",  &PyCaponProcessor::compute_relief_gpu,
           py::arg("signal_ptr"), py::arg("steering_ptr"), py::arg("params"))
      .def("adaptive_beamform_gpu", &PyCaponProcessor::adaptive_beamform_gpu,
           py::arg("signal_ptr"), py::arg("steering_ptr"), py::arg("params"));
}

#endif  // ENABLE_ROCM
