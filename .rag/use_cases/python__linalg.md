---
id: dsp__linalg_linalg__python_test_usecase__v1
type: python_test_usecase
source_path: linalg/python/t_linalg.py
primary_repo: linalg
module: linalg
uses_repos: ['linalg']
uses_external: ['numpy']
has_test_runner: false
is_opencl: false
line_count: 214
title: Тесты биндингов linalg ROCm
tags: ['linalg', 'rocm', 'gpu', 'python', 'signal_processing', 'matrix_ops', 'cross_repo']
uses_pybind:
  - dsp_linalg.ROCmGPUContext
  - dsp_linalg.SymmetrizeMode.Roundtrip
  - dsp_linalg.SymmetrizeMode
  - dsp_linalg.SymmetrizeMode.GpuKernel
  - dsp_linalg.CholeskyInverterROCm
  - dsp_linalg.CaponParams
  - dsp_linalg.CaponProcessor
top_functions:
  - check
  - make_hpd_matrix
  - make_steering_ula
  - make_noise
synonyms_ru:
  - тесты биндингов
  - проверка linalg
  - ROCm тесты
  - Python биндинги
  - dsp_gpu тесты
synonyms_en:
  - linalg tests
  - ROCm bindings
  - Python bindings
  - dsp_gpu tests
  - python_test suite
inherits_block_id: linalg__symmetrize_mode__class_overview__v1
block_refs:
  - linalg__symmetrize_mode__class_overview__v1
ai_generated: false
human_verified: false
---

<!-- rag-block: id=dsp__linalg_linalg__python_test_usecase__v1 -->

# Python use-case: Тесты биндингов linalg ROCm

## Цель

Проверка корректности Python-биндингов linalg на ROCm

## Когда применять

Запускать после изменений в linalg или ROCmGPUContext

## Используемые pybind-классы

| Класс / символ | Репо |
|---|---|
| `dsp_linalg.ROCmGPUContext` | linalg |
| `dsp_linalg.SymmetrizeMode.Roundtrip` | linalg |
| `dsp_linalg.SymmetrizeMode` | linalg |
| `dsp_linalg.SymmetrizeMode.GpuKernel` | linalg |
| `dsp_linalg.CholeskyInverterROCm` | linalg |
| `dsp_linalg.CaponParams` | linalg |
| `dsp_linalg.CaponProcessor` | linalg |

## Внешние зависимости

numpy

## Solution (фрагмент кода)

```python
def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}" + (f"  ({detail})" if detail else ""))
    else:
        failed += 1
        print(f"  [FAIL] {name}" + (f"  ({detail})" if detail else ""))


def make_hpd_matrix(n, seed=42):
    """Create Hermitian Positive Definite matrix: A = B*B^H + n*I"""
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    B = B.astype(np.complex64)
    A = B @ B.conj().T + n * np.eye(n, dtype=np.complex64)
    return A


def make_steering_ula(P, M, theta_min=-np.pi/3, theta_max=np.pi/3):
    """ULA steering: u[p,m] = exp(j*2pi*p*0.5*sin(theta_m)), column-major"""
    U = np.zeros((P, M), dtype=np.complex64)
    for m in range(M):
        theta = theta_min + (theta_max - theta_min) * m / max(M - 1, 1)
        d_sin = np.sin(theta) * 0.5
```

## Connection (C++ ↔ Python)

- C++ class-card: `linalg__symmetrize_mode__class_overview__v1`

## Метаданные

- **Source**: `linalg/python/t_linalg.py`
- **Строк кода**: 214
- **Top-функций**: 4
- **Test runner**: standalone (без runner)

<!-- /rag-block -->
