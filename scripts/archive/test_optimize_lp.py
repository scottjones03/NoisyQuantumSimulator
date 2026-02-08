#!/usr/bin/env python3
"""Run the CZ gate optimizer on LP protocol first (fastest to converge)."""

from src.qpu_simulator.micro_physics.neutral_atoms.rydberg_gates.optimize_cz_gate import (
    optimize_cz_gate, ApparatusConstraints
)
import time

apparatus = ApparatusConstraints()
print(f"Apparatus: spacing={apparatus.spacing_factor}, n_rydberg={apparatus.n_rydberg}")
print(f"Delta_e/2pi = {apparatus.Delta_e/(2*3.14159*1e9):.2f} GHz")
print()

# LP optimization - 2 continuous params: delta_over_omega, omega_tau
print("=" * 60)
print("OPTIMIZING LP PROTOCOL")
print("=" * 60)
t0 = time.time()
result = optimize_cz_gate(
    protocol="lp",
    apparatus=apparatus,
    include_noise=False,
    maxiter=80,
    verbose=True,
)
dt = time.time() - t0
print(f"\nLP optimization took {dt:.1f}s")
print(result)
