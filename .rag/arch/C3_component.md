---
schema_version: 1
repo: linalg
arch_level: c3
tags:
  - "#level:c3"
  - "#repo:linalg"
  - "#layer:compute"
  - "#namespace:dsp_linalg"
  - "#namespace:dsp_linalg"
  - "#namespace:drv_gpu_lib"
description: "C3 Component — key classes и интерфейсы репо linalg."
---

# C3 Component — `linalg`

## Key classes (top-10 по test_params)

### `dsp::linalg::MatrixOpsROCm`

- **Namespace:** `vector_algebra`
- **Методы:** 13, **test_params rows:** 41
- **Brief:** CGEMM операции, привязанные к GpuContext (stream + handle). Создаётся в конструкторе модуля: @code MatrixOpsROCm mat_ops(&ctx_);

### `dsp::linalg::CaponProcessor`

- **Namespace:** `capon`
- **Методы:** 28, **test_params rows:** 19
- **Brief:** *(описание не задано)*

### `drv_gpu_lib::GpuContext`

- **Namespace:** `drv_gpu_lib`
- **Методы:** 14, **test_params rows:** 7
- **Brief:** *(описание не задано)*

### `dsp::linalg::CholeskyInverterROCm`

- **Namespace:** `vector_algebra`
- **Методы:** 39, **test_params rows:** 4
- **Brief:** эрмитовой положительно определённой матрицы (POTRF + POTRI). Два режима симметризации: - Roundtrip: Download → CPU sym → Upload

### `dsp::linalg::DiagonalLoadRegularizer`

- **Namespace:** `vector_algebra`
- **Методы:** 8, **test_params rows:** 4
- **Brief:** загрузка: A += mu I (GPU, compiled via GpuContext). Не копируемый (владеет GpuContext). Перемещаемый. @code Diagona

### `dsp::linalg::AdaptBeamformOp`

- **Namespace:** `capon`
- **Методы:** 2, **test_params rows:** 4
- **Brief:** *(описание не задано)*

### `dsp::linalg::ComputeWeightsOp`

- **Namespace:** `capon`
- **Методы:** 2, **test_params rows:** 4
- **Brief:** *(описание не задано)*

### `dsp::linalg::CholeskyResult`

- **Namespace:** `vector_algebra`
- **Методы:** 13, **test_params rows:** 3
- **Brief:** инверсии матрицы — владеет GPU памятью. Базовый формат — void* d_data (HIP device pointer). Методы AsVector() / matrix() / matrices() скачивают на CPU. AsHipPt

### `dsp::linalg::CaponInvertOp`

- **Namespace:** `capon`
- **Методы:** 6, **test_params rows:** 3
- **Brief:** инверсии ковариационной матрицы. Держит экземпляр CholeskyInverterROCm (он не наследует GpuKernelOp, поэтому CaponInvertOp также не наследует — это обычный класс).

### `dsp::linalg::CovarianceMatrixOp`

- **Namespace:** `capon`
- **Методы:** 2, **test_params rows:** 3
- **Brief:** *(описание не задано)*

## Интерфейсы (наследуемые)

- `dsp::linalg::IMatrixRegularizer` (потенциальных реализаций: 0)

