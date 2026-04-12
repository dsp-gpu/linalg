/**
 * @file dsp_linalg_module.cpp
 * @brief pybind11 bindings for dsp::linalg
 *
 * Python API:
 *   import dsp_linalg
 *   inv = dsp_linalg.CholeskyInverterROCm(ctx)
 *   R_inv = inv.invert(R_matrix)
 *
 * Экспортируемые классы:
 *   CholeskyInverterROCm — Cholesky inversion via rocSOLVER (ROCm)
 *
 * Note: Capon beamformer реализован поверх CholeskyInverterROCm
 *   в Python-тестах, не как отдельный C++ класс в биндингах.
 */

#include "py_helpers.hpp"

#if ENABLE_ROCM
#include "py_vector_algebra_rocm.hpp"
#endif

PYBIND11_MODULE(dsp_linalg, m) {
    m.doc() = "dsp::linalg — linear algebra on GPU (ROCm)\n\n"
              "Classes:\n"
              "  CholeskyInverterROCm - matrix inversion via rocSOLVER\n";

#if ENABLE_ROCM
    register_cholesky_inverter_rocm(m);
#endif
}
