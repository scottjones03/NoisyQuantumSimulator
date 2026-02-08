#!/usr/bin/env python3
"""Quick test of LP baseline parameters."""

from src.qpu_simulator.micro_physics.neutral_atoms.rydberg_gates import (
    simulate_CZ_gate,
    LPSimulationInputs,
    TwoPhotonExcitationConfig,
    NoiseSourceConfig,
    LaserParameters,
)
import numpy as np

print("=" * 60)
print("LP Baseline Test")
print("=" * 60)

# Standard configuration from verify_phase_penalty.py
laser_1 = LaserParameters(power=50e-6, waist=50e-6, linewidth_hz=1000)
laser_2 = LaserParameters(power=0.3, waist=50e-6, linewidth_hz=1000)
excitation = TwoPhotonExcitationConfig(
    laser_1=laser_1,
    laser_2=laser_2,
    Delta_e=5e9,  # 5 GHz in Hz (NOT rad/s!)
    counter_propagating=True,
)
noise = NoiseSourceConfig()

lp_inputs = LPSimulationInputs(excitation=excitation, noise=noise)

# Test at spacing_factor=2.8 (typical)
print("\n--- LP at spacing_factor=2.8 ---")
result = simulate_CZ_gate(
    simulation_inputs=lp_inputs,
    spacing_factor=2.8,
    n_rydberg=70,
    temperature=2e-6,
    tweezer_power=0.020,
    tweezer_waist=0.8e-6,
    include_noise=False,
    verbose=False,
)
print(f"V/Ω = {result.V_over_Omega:.1f}")
print(f"Avg fidelity = {result.avg_fidelity*100:.2f}%")
print(f"Controlled phase = {result.phase_info.get('controlled_phase_deg', np.nan):.1f}°")
print(f"Phase error = {result.phase_info.get('phase_error_from_pi_deg', np.nan):.1f}°")
print(f"|11⟩ fidelity = {result.fidelities.get('11', np.nan)*100:.2f}%")
