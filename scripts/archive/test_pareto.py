#!/usr/bin/env python3
"""Quick Pareto sweep test."""

from src.qpu_simulator.micro_physics.neutral_atoms.rydberg_gates import (
    explore_parameter_space, AtomicConfiguration
)
import numpy as np

# Quick test of the new method
config = AtomicConfiguration(species='Rb87', n_rydberg=60)
print('Clock transition:', config.is_clock_transition)
print('Zeeman dephasing at 1mG:', config.estimate_zeeman_dephasing(1e-7), 'rad/s')

# Try a minimal Pareto sweep
print('\nRunning mini Pareto sweep...')
result = explore_parameter_space(
    protocol="levine_pichler",
    species="Rb87",
    n_runs=1,
    maxiter=5,  # Very small for quick test
    popsize=5,  # Very small for quick test
    verbose=True
)
print(f'Completed {result.n_evaluations} evaluations')
print(f'Pareto points: {len(result.pareto_front)}')
if result.pareto_front:
    best = result.pareto_front[0]
    print(f'Best fidelity: {best.fidelity:.4f}, gate time: {best.gate_time_ns:.2f} ns')
