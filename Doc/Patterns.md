# Архитектурные паттерны репо `linalg`

> **Источник истины:** `linalg/.rag/_RAG.md` (теги `#pattern:Type:Class`, auto-inferred RAG_CLAUDE_C4 от 9.05).
> Brief'ы — из `key_classes:` того же манифеста (fallback из `rag_dsp.symbols`).
>
> Используется как источник для `dataset_v4` (collect_doc_deep подхватит Doc/Patterns.md).
> Alex: проверить + добавить руками то что не размечено в `_RAG.md tags:`.

## Facade

> Тонкий публичный API над набором операций. Стабильный → Python-биндинги не ломаются.


- **`capon::CaponProcessor`** — `linalg/include/linalg/capon_processor.hpp:66`
  - Facade полного MVDR (Capon) pipeline: Cov → diagonal-load → Invert (Cholesky) → ComputeWeights → AdaptiveBeamform / Relief. Использует rocBLAS CGEMM + rocSOLVER. Регуляризация ковариационной матрицы перед обращением.
- **`vector_algebra::MatrixOpsROCm`** — `linalg/include/linalg/matrix_ops_rocm.hpp:54`
  - rocBLAS CGEMM операции, привязанные к GpuContext (stream + handle).

## Pipeline

> Композиция операций в цепочку. Конфиг → Pipeline объект.


- **`capon::CaponProcessor`** — `linalg/include/linalg/capon_processor.hpp:66`
  - Facade полного MVDR (Capon) pipeline: Cov → diagonal-load → Invert (Cholesky) → ComputeWeights → AdaptiveBeamform / Relief. Использует rocBLAS CGEMM + rocSOLVER. Регуляризация ковариационной матрицы перед обращением.

## Adapter

> Тонкая pybind-обёртка над C++ Facade: адаптирует API под Python (numpy↔GPU, GIL release).


- **`PyCaponProcessor`** — `linalg/python/py_capon_rocm.hpp:33`
  - Pybind-Adapter над `capon::CaponProcessor`: numpy↔GPU, GIL-release в `Compute*`. Принимает `ROCmGPUContext&` (`py_gpu_context.hpp`), backend берётся из `ctx.backend()` — единый паттерн для всех модулей DSP-GPU.

## Strategy

> Семейство взаимозаменяемых алгоритмов за общим интерфейсом (`IPipelineStep`).


- **`vector_algebra::IMatrixRegularizer`** — `linalg/include/linalg/i_matrix_regularizer.hpp:45`
  - Применить регуляризацию к квадратной комплексной матрице на GPU.

## Operation

> Атомарная GPU-операция Ref03 Layer 5: `Initialize() / IsReady() / Release()`.


- **`capon::AdaptBeamformOp`** — `linalg/include/linalg/operations/adapt_beam_op.hpp:33`
  - AdaptBeamformOp — адаптивное диаграммообразование Y_out = W^H·Y (Layer 5 Ref03) ЧТO: Concrete Op (наследник GpuKernelOp): финальный шаг Capon-pipeline для режима адаптивного ДО (beamforming). Один rocBLAS CGEMM: Y_out[M × N] = W^H[M × P] · 
- **`capon::CaponInvertOp`** — `linalg/include/linalg/operations/capon_invert_op.hpp:48`
  - Обёртка инверсии ковариационной матрицы.
- **`capon::CaponReliefOp`** — `linalg/include/linalg/operations/capon_relief_op.hpp:35`
  - Concrete Op (наследник GpuKernelOp): финальный шаг pipeline для режима «angular power spectrum» Capon-MVDR. Запускает один HIP kernel `compute_capon_relief`, который для каждого направления m вычисляет: acc[m] = Σ_{p=0..P-1} Re(conj(U[p,m])
- **`capon::ComputeWeightsOp`** — `linalg/include/linalg/operations/compute_weights_op.hpp:33`
  - Concrete Op (наследник GpuKernelOp): средний шаг Capon-pipeline. Один rocBLAS CGEMM: W[P × M] = R^{-1}[P × P] · U[P × M] (NoTrans × NoTrans) Источник R^{-1} — CholeskyResult из CaponInvertOp (передаётся как R_inv_ptr). U — управляющие векто
- **`capon::CovarianceMatrixOp`** — `linalg/include/linalg/operations/covariance_matrix_op.hpp:33`
  - Concrete Op (наследник GpuKernelOp): первый шаг Capon-pipeline. Делегирует в MatrixOpsROCm::CovarianceMatrix: R[P × P] = (1/N) · Y[P × N] · Y^H[N × P] (rocBLAS CHERK или CGEMM) где Y — матрица сигнала из shared kSignal, R записывается в sha


## См. также

- `linalg/.rag/arch/C2_container.md`
- `linalg/.rag/arch/C3_component.md`
- `linalg/.rag/arch/C4_code.md`
- `MemoryBank/.architecture/DSP-GPU_Design_C4_Full.md`

---

*Сгенерировано из `_RAG.md` тегов. Alex редактирует руками + коммитит.*
