# ── DspLinalg Python bindings ─────────────────────────────────────
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
execute_process(
  COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(dsp_linalg dsp_linalg_module.cpp)
target_link_libraries(dsp_linalg PRIVATE DspLinalg::DspLinalg)
target_include_directories(dsp_linalg PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_SOURCE_DIR}/src/vector_algebra/include
  ${CMAKE_SOURCE_DIR}/src/capon/include)
target_compile_definitions(dsp_linalg PRIVATE
  $<$<BOOL:${ENABLE_ROCM}>:ENABLE_ROCM=1>)
install(TARGETS dsp_linalg DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
