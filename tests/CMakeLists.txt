# ── DspLinalg tests ────────────────────────────────────────────────
add_executable(test_linalg_main main.cpp)
target_link_libraries(test_linalg_main PRIVATE DspLinalg::DspLinalg DspCore::TestUtils)
target_include_directories(test_linalg_main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(test_linalg_main PRIVATE
  $<$<BOOL:${ENABLE_ROCM}>:ENABLE_ROCM=1>)
