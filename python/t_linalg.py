#!/usr/bin/env python3
"""
Standalone test for dsp_linalg Python bindings (ROCm).
Запуск: python test_linalg.py

Tests:
  1. Import + ROCmGPUContext creation
  2. CholeskyInverterROCm — identity matrix inversion
  3. CholeskyInverterROCm — HPD matrix inversion (A * A^{-1} ≈ I)
  4. CholeskyInverterROCm — batch inversion (4×64×64)
  5. SymmetrizeMode enum
  6. CaponParams creation
  7. CaponProcessor — relief (noise only, all > 0)
  8. CaponProcessor — interference peak detection
  9. CaponProcessor — adaptive beamform dimensions
"""

import sys
import os
import numpy as np

# .so лежит рядом — добавляем в sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Build dir — Python .so может быть там
build_python = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', 'build', 'python')
if os.path.isdir(build_python):
    sys.path.insert(0, build_python)

passed = 0
failed = 0

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
        for p in range(P):
            phase = 2.0 * np.pi * p * d_sin
            U[p, m] = np.exp(1j * phase)
    # column-major flat: [p + m*P]
    return U.flatten(order='F')


def make_noise(count, sigma=1.0, seed=42):
    """Complex Gaussian noise"""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(count) + 1j * rng.standard_normal(count)).astype(np.complex64) * sigma


print("=" * 60)
print("  dsp_linalg Python bindings test")
print("=" * 60)

# ── Test 1: Import ──────────────────────────────────────────────────
try:
    import dsp_linalg
    check("import dsp_linalg", True)
except ImportError as e:
    check("import dsp_linalg", False, str(e))
    sys.exit(1)

# ── Test 2: ROCmGPUContext ──────────────────────────────────────────
try:
    ctx = dsp_linalg.ROCmGPUContext(0)
    check("ROCmGPUContext(0)", True, f"device={ctx.device_name}")
except Exception as e:
    check("ROCmGPUContext(0)", False, str(e))
    sys.exit(1)

# ── Test 3: SymmetrizeMode enum ─────────────────────────────────────
try:
    rt = dsp_linalg.SymmetrizeMode.Roundtrip
    gk = dsp_linalg.SymmetrizeMode.GpuKernel
    check("SymmetrizeMode enum", rt != gk)
except Exception as e:
    check("SymmetrizeMode enum", False, str(e))

# ── Test 4: CholeskyInverterROCm — Identity ─────────────────────────
try:
    inv = dsp_linalg.CholeskyInverterROCm(ctx, dsp_linalg.SymmetrizeMode.GpuKernel)
    I5 = np.eye(5, dtype=np.complex64)
    I5_inv = inv.invert_cpu(I5.flatten(), n=5)
    err = np.linalg.norm(I5_inv - I5)
    check("Cholesky identity 5x5", err < 1e-5, f"err={err:.2e}")
except Exception as e:
    check("Cholesky identity 5x5", False, str(e))

# ── Test 5: CholeskyInverterROCm — HPD matrix ───────────────────────
try:
    n = 64
    A = make_hpd_matrix(n, seed=123)
    inv2 = dsp_linalg.CholeskyInverterROCm(ctx, dsp_linalg.SymmetrizeMode.GpuKernel)
    A_inv = inv2.invert_cpu(A.flatten(), n=n)
    product = A @ A_inv
    err = np.linalg.norm(product - np.eye(n, dtype=np.complex64))
    check("Cholesky HPD 64x64", err < 0.1, f"||A*A^-1 - I||={err:.4e}")
except Exception as e:
    check("Cholesky HPD 64x64", False, str(e))

# ── Test 6: CholeskyInverterROCm — batch ────────────────────────────
try:
    n = 32
    batch = 4
    matrices = np.stack([make_hpd_matrix(n, seed=k+10) for k in range(batch)])
    flat = matrices.flatten()
    inv3 = dsp_linalg.CholeskyInverterROCm(ctx, dsp_linalg.SymmetrizeMode.GpuKernel)
    result = inv3.invert_batch_cpu(flat, n=n, batch_count=batch)
    check("Cholesky batch 4x32x32", result.shape == (batch, n, n),
          f"shape={result.shape}")
except Exception as e:
    check("Cholesky batch 4x32x32", False, str(e))

# ── Test 7: CaponParams ─────────────────────────────────────────────
try:
    params = dsp_linalg.CaponParams(n_channels=8, n_samples=64,
                                     n_directions=16, mu=0.01)
    check("CaponParams", params.n_channels == 8 and abs(params.mu - 0.01) < 1e-6,
          f"n_ch={params.n_channels}, mu={params.mu}")
except Exception as e:
    check("CaponParams", False, str(e))

# ── Test 8: CaponProcessor — relief (noise only) ────────────────────
try:
    P, N, M = 8, 64, 16
    signal = make_noise(P * N, seed=42)
    steering = make_steering_ula(P, M)
    params = dsp_linalg.CaponParams(n_channels=P, n_samples=N,
                                     n_directions=M, mu=0.01)
    cap = dsp_linalg.CaponProcessor(ctx)
    relief = cap.compute_relief(signal, steering, params)
    all_pos = np.all(relief > 0) and np.all(np.isfinite(relief))
    check("Capon relief (noise)", all_pos and len(relief) == M,
          f"len={len(relief)}, min={relief.min():.4f}, max={relief.max():.4f}")
except Exception as e:
    check("Capon relief (noise)", False, str(e))

# ── Test 9: CaponProcessor — interference peak ──────────────────────
try:
    P, N, M = 8, 128, 32
    signal = make_noise(P * N, seed=77)
    # Add interference at 0 degrees
    theta_int = 0.0
    for n_idx in range(N):
        for p in range(P):
            phase = 2.0 * np.pi * p * 0.5 * np.sin(theta_int) + 0.37 * n_idx
            signal[n_idx * P + p] += 10.0 * np.exp(1j * phase)
    steering = make_steering_ula(P, M)
    params = dsp_linalg.CaponParams(n_channels=P, n_samples=N,
                                     n_directions=M, mu=0.001)
    cap2 = dsp_linalg.CaponProcessor(ctx)
    relief = cap2.compute_relief(signal, steering, params)
    # Find peak direction
    m_peak = np.argmax(relief)
    mean_r = np.mean(relief)
    # Interference at 0 deg should be near center (m≈15-16 for M=32)
    is_peak = relief[m_peak] > mean_r * 2.0
    check("Capon interference peak", is_peak,
          f"m_peak={m_peak}, ratio={relief[m_peak]/mean_r:.1f}")
except Exception as e:
    check("Capon interference peak", False, str(e))

# ── Test 10: CaponProcessor — beamform dimensions ───────────────────
try:
    P, N, M = 4, 32, 6
    signal = make_noise(P * N, seed=13)
    steering = make_steering_ula(P, M, -np.pi/6, np.pi/6)
    params = dsp_linalg.CaponParams(n_channels=P, n_samples=N,
                                     n_directions=M, mu=0.01)
    cap3 = dsp_linalg.CaponProcessor(ctx)
    beam = cap3.adaptive_beamform(signal, steering, params)
    check("Capon beamform dims", beam.shape == (M, N),
          f"shape={beam.shape}")
except Exception as e:
    check("Capon beamform dims", False, str(e))

# ── Test 11: repr ───────────────────────────────────────────────────
try:
    inv_r = dsp_linalg.CholeskyInverterROCm(ctx)
    r = repr(inv_r)
    check("CholeskyInverterROCm repr", "POTRF" in r, r)
except Exception as e:
    check("repr", False, str(e))

# ── Summary ─────────────────────────────────────────────────────────
print("=" * 60)
total = passed + failed
print(f"  Results: {passed}/{total} passed" +
      (f", {failed} FAILED" if failed else " -- ALL PASSED"))
print("=" * 60)

sys.exit(1 if failed else 0)
