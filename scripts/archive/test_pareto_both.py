#!/usr/bin/env python3
"""Run Pareto sweep for both protocols with realistic parameters."""

import sys
sys.path.insert(0, '/Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/NoisyQuantumSimulator/src')

import numpy as np
from qpu_simulator.micro_physics.neutral_atoms.rydberg_gates.optimization import (
    explore_parameter_space
)

print("=" * 60)
print("PARETO SWEEP FOR BOTH PROTOCOLS")
print("=" * 60)

# Run LP protocol
print("\n" + "="*60)
print("LEVINE-PICHLER PROTOCOL")
print("="*60)
lp_result = explore_parameter_space(
    protocol="levine_pichler",
    species="Rb87",
    n_runs=1,
    maxiter=20,
    popsize=10,
    verbose=True
)
print(f"\nLP Results:")
print(f"  Evaluations: {lp_result.n_evaluations}")
print(f"  Pareto points: {len(lp_result.pareto_front)}")
if lp_result.pareto_front:
    for i, pt in enumerate(lp_result.pareto_front[:5]):
        print(f"    {i+1}. F={pt.fidelity:.4f}, τ={pt.gate_time_ns:.1f} ns")

# Run JP protocol
print("\n" + "="*60)
print("JANDURA-PUPILLO PROTOCOL")
print("="*60)
jp_result = explore_parameter_space(
    protocol="jandura_pupillo",
    species="Rb87",
    n_runs=1,
    maxiter=20,
    popsize=10,
    verbose=True
)
print(f"\nJP Results:")
print(f"  Evaluations: {jp_result.n_evaluations}")
print(f"  Pareto points: {len(jp_result.pareto_front)}")
if jp_result.pareto_front:
    for i, pt in enumerate(jp_result.pareto_front[:5]):
        print(f"    {i+1}. F={pt.fidelity:.4f}, τ={pt.gate_time_ns:.1f} ns")

# Compare best points
print("\n" + "="*60)
print("COMPARISON: BEST FIDELITY POINTS")
print("="*60)
if lp_result.pareto_front and jp_result.pareto_front:
    lp_best = max(lp_result.pareto_front, key=lambda x: x.fidelity)
    jp_best = max(jp_result.pareto_front, key=lambda x: x.fidelity)
    print(f"LP Best: F={lp_best.fidelity:.4f}, τ={lp_best.gate_time_ns:.1f} ns")
    print(f"JP Best: F={jp_best.fidelity:.4f}, τ={jp_best.gate_time_ns:.1f} ns")
