#!/usr/bin/env python3
"""Quick test: run baselines for all 3 CZ protocols with fixed parameter injection."""

from src.qpu_simulator.micro_physics.neutral_atoms.rydberg_gates.optimize_cz_gate import (
    run_baseline, ApparatusConstraints
)
import time

apparatus = ApparatusConstraints()
print("Running baselines with default ApparatusConstraints...")
print(f"  spacing_factor={apparatus.spacing_factor}, n_rydberg={apparatus.n_rydberg}")
print()

for proto in ["lp", "jp_bangbang", "smooth_jp"]:
    t0 = time.time()
    print(f"--- {proto.upper()} ---")
    metrics = run_baseline(proto, apparatus, include_noise=False, verbose=True)
    dt = time.time() - t0
    print(f"  Time: {dt:.1f}s")
    print(f"  Avg fidelity:  {metrics['avg_fidelity']*100:.2f}%")
    print(f"  Phase error:   {metrics['phase_error_deg']:.1f}deg")
    print(f"  F(|11>):       {metrics['f11']*100:.2f}%")
    if "V_over_Omega" in metrics:
        print(f"  V/Omega:       {metrics['V_over_Omega']:.1f}")
    print()

print("=== BASELINE SUMMARY ===")
print("Expected: LP ~99.65%, JP/smooth should now respond to correct params")
