#!/usr/bin/env python3
"""Run the CZ gate optimizer on smooth JP and JP bang-bang protocols."""

from src.qpu_simulator.micro_physics.neutral_atoms.rydberg_gates.optimize_cz_gate import (
    optimize_cz_gate, ApparatusConstraints
)
import time

apparatus = ApparatusConstraints()
print(f"Apparatus: spacing={apparatus.spacing_factor}, n_rydberg={apparatus.n_rydberg}")
print(f"Delta_e/2pi = {apparatus.Delta_e/(2*3.14159*1e9):.2f} GHz")
print()

# Smooth JP optimization - 4 continuous params: A, omega_mod_ratio, phi_offset, omega_tau
print("=" * 60)
print("OPTIMIZING SMOOTH JP PROTOCOL")
print("=" * 60)
t0 = time.time()
result_smooth = optimize_cz_gate(
    protocol="smooth_jp",
    apparatus=apparatus,
    include_noise=False,
    maxiter=80,
    verbose=True,
)
dt = time.time() - t0
print(f"\nSmooth JP optimization took {dt:.1f}s")
print(result_smooth)

# JP bang-bang optimization - tries both 5 and 7 segments
print()
print("=" * 60)
print("OPTIMIZING JP BANG-BANG PROTOCOL")
print("=" * 60)
t0 = time.time()
result_jp = optimize_cz_gate(
    protocol="jp_bangbang",
    apparatus=apparatus,
    include_noise=False,
    maxiter=80,
    verbose=True,
)
dt = time.time() - t0
print(f"\nJP bang-bang optimization took {dt:.1f}s")
print(result_jp)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Smooth JP: F={result_smooth.best_metrics['avg_fidelity']*100:.2f}%, "
      f"phase_err={result_smooth.best_metrics['phase_error_deg']:.1f}deg")
print(f"JP BB:     F={result_jp.best_metrics['avg_fidelity']*100:.2f}%, "
      f"phase_err={result_jp.best_metrics['phase_error_deg']:.1f}deg")
