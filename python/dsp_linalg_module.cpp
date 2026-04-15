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
#include "py_vector_algebra_rocm.hpp"
#include "py_capon_rocm.hpp"
#endif

PYBIND11_MODULE(dsp_linalg, m) {
    m.doc() = "dsp::linalg — linear algebra on GPU (ROCm)";

#if ENABLE_ROCM
    register_cholesky_inverter_rocm(m);
    register_capon_processor(m);
#endif
}
