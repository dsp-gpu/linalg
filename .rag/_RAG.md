---
schema_version: 1
repo: linalg
version: 0.1.0
layer: compute
maturity: alpha
purpose: "TODO: AI-fill — назначение репо linalg"

modules:
  public:                               # auto: include/<repo>/*
    - kernels
    - operations
  internal:                             # auto: src/* кроме include
    - capon
    - vector_algebra

key_classes:                            # auto: top по test_params
  - fqn: dsp::linalg::MatrixOpsROCm
    brief: "rocBLAS CGEMM операции, привязанные к GpuContext (stream + handle)."
    maturity: alpha
    methods: 13
    test_params_rows: 29
    test_params: test_params/dsp_linalg_MatrixOpsROCm.md
  - fqn: dsp::linalg::CaponProcessor
    brief: "@ingroup grp_capon"
    maturity: alpha
    methods: 28
    test_params_rows: 10
    test_params: test_params/dsp_linalg_CaponProcessor.md
  - fqn: dsp::linalg::CholeskyInverterROCm
    brief: "Инверсия эрмитовой положительно определённой матрицы (POTRF + POTRI)."
    maturity: alpha
    methods: 39
    test_params_rows: 4
    test_params: test_params/dsp_linalg_CholeskyInverterROCm.md
  - fqn: drv_gpu_lib::GpuContext
    brief: "TODO: AI-fill"
    maturity: alpha
    methods: 14
    test_params_rows: 4
    test_params: test_params/drv_gpu_lib_GpuContext.md
  - fqn: dsp::linalg::DiagonalLoadRegularizer
    brief: "Диагональная загрузка: A += mu * I (GPU, compiled via GpuContext)."
    maturity: alpha
    methods: 8
    test_params_rows: 4
    test_params: test_params/dsp_linalg_DiagonalLoadRegularizer.md
  - fqn: dsp::linalg::AdaptBeamformOp
    brief: "TODO: AI-fill"
    maturity: alpha
    methods: 2
    test_params_rows: 4
    test_params: test_params/dsp_linalg_AdaptBeamformOp.md
  - fqn: dsp::linalg::ComputeWeightsOp
    brief: "TODO: AI-fill"
    maturity: alpha
    methods: 2
    test_params_rows: 4
    test_params: test_params/dsp_linalg_ComputeWeightsOp.md
  - fqn: dsp::linalg::CovarianceMatrixOp
    brief: "TODO: AI-fill"
    maturity: alpha
    methods: 2
    test_params_rows: 3
    test_params: test_params/dsp_linalg_CovarianceMatrixOp.md
  - fqn: dsp::linalg::CaponInvertOp
    brief: "Обёртка инверсии ковариационной матрицы."
    maturity: alpha
    methods: 6
    test_params_rows: 2
    test_params: test_params/dsp_linalg_CaponInvertOp.md
  - fqn: dsp::linalg::CaponReliefOp
    brief: "TODO: AI-fill"
    maturity: alpha
    methods: 2
    test_params_rows: 2
    test_params: test_params/dsp_linalg_CaponReliefOp.md
  - fqn: dsp::linalg::CholeskyResult
    brief: "Результат инверсии матрицы — владеет GPU памятью."
    maturity: alpha
    methods: 13
    test_params_rows: 0
    test_params: test_params/dsp_linalg_CholeskyResult.md
  - fqn: PyCaponProcessor
    brief: "TODO: AI-fill"
    maturity: alpha
    methods: 5
    test_params_rows: 0
    test_params: test_params/PyCaponProcessor.md
  - fqn: PyCholeskyInverterROCm
    brief: "TODO: AI-fill"
    maturity: alpha
    methods: 5
    test_params_rows: 0
    test_params: test_params/PyCholeskyInverterROCm.md
  - fqn: test_capon_rocm_bench::CaponBeamformBenchmarkROCm
    brief: "TODO: AI-fill"
    maturity: alpha
    methods: 3
    test_params_rows: 0
    test_params: test_params/test_capon_rocm_bench_CaponBeamformBenchmarkROCm.md
  - fqn: test_capon_rocm_bench::CaponReliefBenchmarkROCm
    brief: "TODO: AI-fill"
    maturity: alpha
    methods: 3
    test_params_rows: 0
    test_params: test_params/test_capon_rocm_bench_CaponReliefBenchmarkROCm.md
  - fqn: dsp::linalg::IMatrixRegularizer
    brief: "Применить регуляризацию к квадратной комплексной матрице на GPU."
    maturity: alpha
    methods: 1
    test_params_rows: 0
    test_params: test_params/dsp_linalg_IMatrixRegularizer.md
  - fqn: dsp::linalg::NoOpRegularizer
    brief: "Ничего не делает. Безопасный заменитель nullptr (Null Object)."
    maturity: alpha
    methods: 1
    test_params_rows: 0
    test_params: test_params/dsp_linalg_NoOpRegularizer.md

test_params_summary:
  classes_with_params: 9
  methods_with_params: 14
  ready_for_autotest:  39
  partial_coverage:    23
  no_status:           0
  total_rows:          62

repo_stats:
  total_symbols: 332
  public_classes: 28
  total_files: 65

depends_on:                              # TODO: ручная разметка после deps таблицы
  internal: []
  external: []

used_by: []                              # TODO: AI-fill из других _RAG.md

python_modules:                          # TODO: auto from pybind_bindings
  - TODO

architecture_files:                       # auto: arch_files generator
  - .rag/arch/C2_container.md
  - .rag/arch/C3_component.md
  - .rag/arch/C4_code.md
tags:                                    # auto-inferred (RAG_CLAUDE_C4)
  - "#layer:compute"
  - "#repo:linalg"
  - "#namespace:dsp_linalg"
  - "#namespace:dsp_linalg"
  - "#namespace:drv_gpu_lib"
  - "#pattern:Pipeline:CaponProcessor"
  - "#pattern:Facade:CaponProcessor"
  - "#pattern:Facade:MatrixOpsROCm"
  - "#pattern:Strategy:IMatrixRegularizer"
  - "#pattern:Operation:AdaptBeamformOp"
  - "#pattern:Operation:CaponInvertOp"
  - "#pattern:Operation:CaponReliefOp"
  - "#pattern:Operation:ComputeWeightsOp"
  - "#pattern:Operation:CovarianceMatrixOp"
  - "#pattern:Adapter:PyCaponProcessor"

notes: []                                # TODO: AI-fill из ai_summary

ai_generated_at: 2026-05-09T05:27:59Z
ai_model: TODO (auto-fields only, AI-brief pending)
ai_sections: []
parser_version: 1
---

# linalg

## Назначение
*(TODO: AI-fill через ollama qwen3:8b)*

## Ключевые классы
*(автогенерируется из YAML key_classes выше)*

## Дополнительная документация
- [../Doc/](../Doc/)

<!-- ⚙️ Auto-generated by generate_rag_manifest.py — отредактируй и закоммить. -->
