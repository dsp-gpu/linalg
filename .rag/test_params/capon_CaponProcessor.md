---
schema_version: 1
repo: linalg
class_fqn: capon::CaponProcessor
file: E:/DSP-GPU/linalg/include/linalg/capon_processor.hpp
line: 66
brief: "Реализует алгоритм Capon (MVDR) для формирования луча и вычисления рельефа на GPU с использованием ROCm/hip."
methods_total: 8
methods_with_doxygen: 8
ai_generated: true
human_verified: false
parser_version: 2
synonyms_ru: ['CaponBeamformer', 'MVDRProcessor', 'Радиолокационный фильтр', 'GPUОбработкаСигналов']
synonyms_en: ['CaponBeamformer', 'MVDRProcessor', 'RadarFilter', 'GPUSignalProcessing']
tags: ['ROCm', 'hip', 'Capon', 'beamforming', 'GPU']
---

# `capon::CaponProcessor` — карточка класса

> **Этот файл генерируется автоматически** командой `dsp-asst rag cards build --repo linalg --class CaponProcessor`.
> Не править руками — правки потеряются при следующем refresh.
> Источник правды — Doxygen-теги в `.hpp` + секции в `Doc/*.md`.

---

## Описание класса

<!-- rag-block: id=linalg__capon_processor__class_overview__v1 -->

**ЧТО**: Реализует алгоритм Capon (MVDR) для формирования луча и вычисления рельефа на GPU с использованием ROCm/hip.

**ЗАЧЕМ**: Решает задачи радиолокационной обработки сигналов с высокой производительностью через оптимизированные rocBLAS операции и кастомные GPU-ядра.

**КАК**: Использует ROCm/hip для GPU-вычислений, rocBLAS для матричных операций, кастомные ядра для рельефа, поддержку CPU/GPU через разные сигнатуры методов.

**Пример**:
```cpp
#include <linalg/capon_processor.hpp>
using namespace capon;

CaponProcessor proc(backend);
CaponParams params = {8, 128, 32, 0.01f};

// GPU-версия
void* gpu_signal = ...;
void* gpu_steering = ...;

CaponBeamResult result = proc.AdaptiveBeamform(gpu_signal, gpu_steering, params);
```

<!-- /rag-block -->

## Связанные секции из Doc/

- `linalg__dsp__capon_beamforming_005__v1` (capon_beamforming): ```c // grid=(M+255)/256, block=256, каждый thread m → одно направление extern "C" __global__ void compute_capon_relief(     const float2* U, const float2* W, float* z, unsigned int P, unsigned int M)…
- `linalg__quick__capon_beamforming_002__v1` (capon_beamforming): ``` Y [P×N],  U [P×M]   1. R = Y·Y^H/N + μI       ← rocBLAS CGEMM + HIP kernel add_regularization   2. R⁻¹                    ← vector_algebra::CholeskyInverterROCm (POTRF+POTRI)   3a. Relief: z[m] = …
- `linalg__meta__claude_card__v1` (meta_claude): <!-- type:meta_claude repo:linalg source:linalg/CLAUDE.md -->  # linalg — Repository Card  _Источник: `linalg/CLAUDE.md`_  # 🤖 CLAUDE — `linalg`  > Линейная алгебра на GPU: matrix ops, SVD, eig, Capon…
- `linalg__api__capon_beamforming_002__v1` (capon_beamforming): // Не копируемый, перемещаемый // NOTE: move assignment не переприсваивает inv_op_ (CholeskyInverterROCm не перемещаемый) ```  ---  #### Op-классы (Layer 5)  ##### CovarianceMatrixOp (`operations/cova…
- `linalg__api__capon_beamforming_001__v1` (capon_beamforming): ## Компонент: Capon Beamforming  **Namespace**: `capon` | **Backend**: ROCm only (`ENABLE_ROCM=1`)  ---  #### Типы (`capon_types.hpp`)  ```cpp struct CaponParams {   uint32_t n_channels;    // P — чис…

## Public-методы (8)

## Method 1: `ComputeRelief`

**Сигнатура** (`capon_processor.hpp:140`):
```cpp
CaponReliefResult ComputeRelief( const std::vector<std::complex<float>>& signal, const std::vector<std::complex<float>>& steering, const CaponParams& params)
```

**Параметры**:
- `signal` — `const std::vector<std::complex<float>>&`
- `steering` — `const std::vector<std::complex<float>>&`
- `params` — `const CaponParams&`

**Возвращает**: `CaponReliefResult`

**Doxygen-источник**:
```cpp
/**

   * @brief Вычислить рельеф Кейпона

   * @param signal    Y: матрица сигнала [n_channels × n_samples], column-major

   * @param steering  U: управляющие векторы [n_channels × n_directions], column-major

   * @param params    Параметры (n_channels, n_samples, n_directions, mu)

   *   @test_ref CaponParams

   * @return CaponReliefResult — M вещественных значений пространственного спектра

   *   @test_check result.relief.size() == params.n_directions

   */
```

## Method 2: `AdaptiveBeamform`

**Сигнатура** (`capon_processor.hpp:154`):
```cpp
CaponBeamResult AdaptiveBeamform( const std::vector<std::complex<float>>& signal, const std::vector<std::complex<float>>& steering, const CaponParams& params)
```

**Параметры**:
- `signal` — `const std::vector<std::complex<float>>&`
- `steering` — `const std::vector<std::complex<float>>&`
- `params` — `const CaponParams&`

**Возвращает**: `CaponBeamResult`

**Doxygen-источник**:
```cpp
/**

   * @brief Адаптивное диаграммообразование

   * @param signal    Y: матрица сигнала [n_channels × n_samples], column-major

   * @param steering  U: управляющие векторы [n_channels × n_directions], column-major

   * @param params    Параметры

   *   @test_ref CaponParams

   * @return CaponBeamResult — матрица [n_directions × n_samples]

   *   @test_check result.output.size() == params.n_directions * params.n_samples

   */
```

## Method 3: `ComputeRelief`

**Сигнатура** (`capon_processor.hpp:174`):
```cpp
CaponReliefResult ComputeRelief( void* gpu_signal, void* gpu_steering, const CaponParams& params)
```

**Параметры**:
- `gpu_signal` — `void*` *(pointer)* *(void\*)*
- `gpu_steering` — `void*` *(pointer)* *(void\*)*
- `params` — `const CaponParams&`

**Возвращает**: `CaponReliefResult`

**Doxygen-источник**:
```cpp
/**

   * @brief Рельеф Кейпона (GPU входы)

   * @param gpu_signal   Y на GPU: complex<float>[n_channels × n_samples], column-major

   *   @test { pattern=gpu_pointer, values=["valid_alloc", nullptr], error_values=[0xDEADBEEF, null] }

   * @param gpu_steering U на GPU: complex<float>[n_channels × n_directions], column-major

   * @param params Параметры (n_channels, n_samples, n_directions, mu).

   *   @test_ref CaponParams

   * @return CaponReliefResult — M вещественных значений пространственного спектра.

   *   @test_check result.relief.size() == params.n_directions

   *   @test { pattern=gpu_pointer, values=["valid_alloc", nullptr], error_values=[0xDEADBEEF, null] }

   */
```

## Method 4: `AdaptiveBeamform`

**Сигнатура** (`capon_processor.hpp:190`):
```cpp
CaponBeamResult AdaptiveBeamform( void* gpu_signal, void* gpu_steering, const CaponParams& params)
```

**Параметры**:
- `gpu_signal` — `void*` *(pointer)* *(void\*)*
- `gpu_steering` — `void*` *(pointer)* *(void\*)*
- `params` — `const CaponParams&`

**Возвращает**: `CaponBeamResult`

**Doxygen-источник**:
```cpp
/**

   * @brief Адаптивное ДО (GPU входы)

   * @param gpu_signal Y на GPU: complex<float>[n_channels × n_samples], column-major.

   *   @test { pattern=gpu_pointer, values=["valid_alloc", nullptr], error_values=[0xDEADBEEF, null] }

   * @param gpu_steering U на GPU: complex<float>[n_channels × n_directions], column-major.

   *   @test { pattern=gpu_pointer, values=["valid_alloc", nullptr], error_values=[0xDEADBEEF, null] }

   * @param params Параметры (n_channels, n_samples, n_directions, mu).

   *   @test_ref CaponParams

   * @return CaponBeamResult — матрица [n_directions × n_samples].

   *   @test_check result.output.size() == params.n_directions * params.n_samples

   */
```

## Method 5: `ComputeRelief`

**Сигнатура** (`capon_processor.hpp:295`):
```cpp
CaponReliefResult ComputeRelief(const std::vector<std::complex<float>>&, const std::vector<std::complex<float>>&, const CaponParams&) { throw std::runtime_error("CaponProcessor: ROCm not enabled");
```

**Параметры**:
- `_unnamed_` — `const std::vector<std::complex<float>>&`
- `_unnamed_` — `const std::vector<std::complex<float>>&`
- `_unnamed_` — `const CaponParams&`

**Возвращает**: `CaponReliefResult`

**Doxygen-источник**:
```cpp
/**

   * @brief Stub: бросает runtime_error — ComputeRelief доступен только в ROCm-сборке.

   *

   *

   * @return Никогда не возвращает (всегда throw).

   *   @test_check throws std::runtime_error

   *

   * @throws std::runtime_error всегда: "ROCm not enabled".

   *   @test_check throws std::runtime_error

   */
```

## Method 6: `AdaptiveBeamform`

**Сигнатура** (`capon_processor.hpp:309`):
```cpp
CaponBeamResult AdaptiveBeamform(const std::vector<std::complex<float>>&, const std::vector<std::complex<float>>&, const CaponParams&) { throw std::runtime_error("CaponProcessor: ROCm not enabled");
```

**Параметры**:
- `_unnamed_` — `const std::vector<std::complex<float>>&`
- `_unnamed_` — `const std::vector<std::complex<float>>&`
- `_unnamed_` — `const CaponParams&`

**Возвращает**: `CaponBeamResult`

**Doxygen-источник**:
```cpp
/**

   * @brief Stub: бросает runtime_error — AdaptiveBeamform доступен только в ROCm-сборке.

   *

   *

   * @return Никогда не возвращает (всегда throw).

   *   @test_check throws std::runtime_error

   *

   * @throws std::runtime_error всегда: "ROCm not enabled".

   *   @test_check throws std::runtime_error

   */
```

## Method 7: `ComputeRelief`

**Сигнатура** (`capon_processor.hpp:323`):
```cpp
CaponReliefResult ComputeRelief(void*, void*, const CaponParams&) { throw std::runtime_error("CaponProcessor: ROCm not enabled");
```

**Параметры**:
- `_unnamed_` — `void*` *(pointer)* *(void\*)*
- `_unnamed_` — `void*` *(pointer)* *(void\*)*
- `_unnamed_` — `const CaponParams&`

**Возвращает**: `CaponReliefResult`

**Doxygen-источник**:
```cpp
/**

   * @brief Stub: бросает runtime_error — ComputeRelief (GPU) доступен только в ROCm-сборке.

   *

   *

   * @return Никогда не возвращает (всегда throw).

   *   @test_check throws std::runtime_error

   *

   * @throws std::runtime_error всегда: "ROCm not enabled".

   *   @test_check throws std::runtime_error

   */
```

## Method 8: `AdaptiveBeamform`

**Сигнатура** (`capon_processor.hpp:336`):
```cpp
CaponBeamResult AdaptiveBeamform(void*, void*, const CaponParams&) { throw std::runtime_error("CaponProcessor: ROCm not enabled");
```

**Параметры**:
- `_unnamed_` — `void*` *(pointer)* *(void\*)*
- `_unnamed_` — `void*` *(pointer)* *(void\*)*
- `_unnamed_` — `const CaponParams&`

**Возвращает**: `CaponBeamResult`

**Doxygen-источник**:
```cpp
/**

   * @brief Stub: бросает runtime_error — AdaptiveBeamform (GPU) доступен только в ROCm-сборке.

   *

   *

   * @return Никогда не возвращает (всегда throw).

   *   @test_check throws std::runtime_error

   *

   * @throws std::runtime_error всегда: "ROCm not enabled".

   *   @test_check throws std::runtime_error

   */
```


## Python API

**Pybind модуль**: `dsp_linalg` · **Класс Python**: `CaponProcessor` · **Wrapper C++**: `PyCaponProcessor`

_Источник биндинга_: `linalg/python/py_capon_rocm.hpp`

**Конструктор**: `py::init<ROCmGPUContext&>()`

| Python | C++ | Overload |
|---|---|---|
| `compute_relief` | `PyCaponProcessor::compute_relief` | — |
| `adaptive_beamform` | `PyCaponProcessor::adaptive_beamform` | — |
| `compute_relief_gpu` | `PyCaponProcessor::compute_relief_gpu` | — |
| `adaptive_beamform_gpu` | `PyCaponProcessor::adaptive_beamform_gpu` | — |
