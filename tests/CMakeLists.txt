# ── DspLinalg tests ────────────────────────────────────────────────

# HSA runtime: нужен для ZeroCopy (hsa_interop.cpp через DspCore)
find_library(HSA_RUNTIME_LIB_LINALG
  NAMES hsa-runtime64
  PATH_SUFFIXES lib lib64
  REQUIRED
)

add_executable(test_linalg_main main.cpp)
target_link_libraries(test_linalg_main PRIVATE
  DspLinalg::DspLinalg
  DspCore::TestUtils
)
# HSA runtime нужен ПОСЛЕ DspCore.a (ld ищет символы слева направо).
# --no-as-needed гарантирует что .so не будет выброшен до обработки .a
target_link_options(test_linalg_main PRIVATE "LINKER:--no-as-needed")
target_link_libraries(test_linalg_main PRIVATE ${HSA_RUNTIME_LIB_LINALG})
target_include_directories(test_linalg_main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(test_linalg_main PRIVATE
  $<$<BOOL:${ENABLE_ROCM}>:ENABLE_ROCM=1>)
