# 🤖 CLAUDE — `linalg`

> Линейная алгебра на GPU: matrix ops, SVD, eig, Capon beamformer.
> Зависит от: `core` + `rocblas` + `rocsolver`. Глобальные правила → `../CLAUDE.md` + `.claude/rules/*.md`.

## 🎯 Что здесь

| Класс | Что делает |
|-------|-----------|
| `MatrixOpsROCm` | gemm / gemv / transpose / conjugate / outer product |
| `SVDOp` | SVD через rocSOLVER |
| `EigOp` | Собственные значения/векторы через rocSOLVER |
| `CaponProcessor` | Capon (MVDR) beamformer — полный pipeline |

## 📁 Структура

```
linalg/
├── include/dsp/linalg/
│   ├── matrix_ops_rocm.hpp
│   ├── capon_processor.hpp          # facade
│   ├── gpu_context.hpp
│   ├── operations/                  # GemmOp, GemvOp, SVDOp, EigOp, TransposeOp
│   └── strategies/
├── src/
├── kernels/rocm/                    # custom_conjugate.hip, outer_product.hip
├── tests/                           # эталонный набор (vector_algebra, capon)
└── python/dsp_linalg_module.cpp
```

## ⚠️ Специфика

- **rocBLAS** — для gemm/gemv. **rocSOLVER** — для SVD/eig. Свои реализации запрещены.
- **Row-major vs Column-major**: rocBLAS использует **column-major** (как BLAS). Транспонировать при интеграции.
- **Capon** — регуляризация ковариационной матрицы перед обращением (diagonal loading).
- **LDA/LDB/LDC**: leading dimension = физический stride, не логический размер.

## 🚫 Запреты

- Свой gemm / gemv — только rocBLAS.
- Свой SVD / eig — только rocSOLVER.
- Не смешивать row-major и column-major без явной транспонировки.

## 🔗 Эталон

`linalg/tests/` — **эталонный набор** для остальных модулей (стиль test_*.hpp, GRASP, GoF).

<!-- BEGIN: RAG_CLAUDE_C4 (auto) -->
## 🏗️ Архитектура (C4 — компактно)

- **C1 System Context:** репо `linalg` (layer=compute). Полный C4 → `MemoryBank/.architecture/DSP-GPU_Design_C4_Full.md` §`linalg`
- **C2 Container:** namespace из top key_classes (см. `.rag/_RAG.md`)
- **C3 Component:** `key_classes` в `.rag/_RAG.md` (top по test_params)
- **C4 Code:** MatrixOpsROCm · CaponProcessor · CholeskyInverterROCm · GpuContext

## 🏷️ RAG теги



`#layer:compute` `#repo:linalg` `#namespace:vector_algebra` `#namespace:capon` `#namespace:drv_gpu_lib` `#pattern:Pipeline:CaponProcessor` `#pattern:Facade:CaponProcessor` `#pattern:Facade:MatrixOpsROCm` `#pattern:Strategy:IMatrixRegularizer` `#pattern:Operation:AdaptBeamformOp` `#pattern:Operation:CaponInvertOp` `#pattern:Operation:CaponReliefOp` `#pattern:Operation:ComputeWeightsOp` `#pattern:Operation:CovarianceMatrixOp` `#pattern:Adapter:PyCaponProcessor`

## 🔗 Правила (path-scoped автоматически)

- `09-rocm-only.md` — rocBLAS / rocSOLVER
- `05-architecture-ref03.md`
- `14-cpp-style.md` + `15-cpp-testing.md`
- `11-python-bindings.md`
