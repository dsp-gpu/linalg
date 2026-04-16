/**
 * @file dsp_linalg_module.cpp
 * @brief pybind11 bindings for dsp::linalg (vector_algebra + capon)
 *
 * Экспортируемые классы:
 *   CholeskyInverterROCm — инверсия матриц (rocSOLVER POTRF+POTRI)
 *   CaponProcessor       — MVDR beamformer (полный GPU pipeline)
 *   CaponParams          — параметры алгоритма Capon
 *   SymmetrizeMode       — режим симметризации (Roundtrip/GpuKernel)
 */

#include "py_helpers.hpp"

#if ENABLE_ROCM
#include "py_gpu_context.hpp"
#include "py_vector_algebra_rocm.hpp"
#include "py_capon_rocm.hpp"
#endif

PYBIND11_MODULE(dsp_linalg, m) {
    m.doc() = "dsp::linalg — linear algebra on GPU (ROCm)\n\n"
              "Classes:\n"
              "  ROCmGPUContext           - GPU context (AMD ROCm)\n"
              "  SymmetrizeMode           - Roundtrip / GpuKernel\n"
              "  CholeskyInverterROCm     - matrix inversion (Cholesky, ROCm)\n"
              "  CaponParams              - MVDR beamformer parameters\n"
              "  CaponProcessor           - MVDR beamformer (full GPU pipeline)\n";

#if ENABLE_ROCM
    py::class_<ROCmGPUContext>(m, "ROCmGPUContext",
        "ROCm GPU context (creates HIP backend for AMD GPU).")
        .def(py::init<int>(), py::arg("device_index") = 0)
        .def_property_readonly("device_name", &ROCmGPUContext::device_name)
        .def_property_readonly("device_index", &ROCmGPUContext::device_index);

    register_cholesky_inverter_rocm(m);
    register_capon_processor(m);
#endif
}
