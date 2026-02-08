#!/usr/bin/env python3
"""
Test: Verify weak blockade still shows low fidelity.
"""
import sys
sys.path.insert(0, '/Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/NoisyQuantumSimulator/src')

from qpu_simulator.micro_physics.neutral_atoms.rydberg_gates import (
    simulate_CZ_gate,
    LPSimulationInputs,
    JPSimulationInputs,
    TwoPhotonExcitationConfig,
    NoiseSourceConfig,
    LaserParameters,
)

# Setup laser configuration
laser_1 = LaserParameters(power=50e-6, waist=50e-6, linewidth_hz=1000, polarization="sigma+", polarization_purity=0.99)
laser_2 = LaserParameters(power=0.3, waist=50e-6, linewidth_hz=1000, polarization="sigma+", polarization_purity=0.99)
excitation = TwoPhotonExcitationConfig(laser_1=laser_1, laser_2=laser_2, Delta_e=5e9)
noise_config = NoiseSourceConfig()

lp_inputs = LPSimulationInputs(excitation=excitation, noise=noise_config)
jp_inputs = JPSimulationInputs(excitation=excitation, noise=noise_config)

print("=" * 70)
print("WEAK BLOCKADE TEST (sf=4.0, should show low fidelity)")
print("=" * 70)

# Weak blockade: sf=4.0 gives V/Ω ~ 5-10
print("\n--- LP Protocol (sf=4.0, no noise) ---")
lp_result = simulate_CZ_gate(
    simulation_inputs=lp_inputs,
    species="Rb87",
    n_rydberg=60,
    temperature=1e-6,
    spacing_factor=4.0,  # Weak blockade
    tweezer_power=0.02,
    tweezer_waist=0.8e-6,
    include_noise=False,
    verbose=True
)
print(f"\n  Fidelity: {lp_result.avg_fidelity*100:.2f}%")
print(f"  V/Ω: {lp_result.V_over_Omega:.1f}")
for state, f in lp_result.fidelities.items():
    print(f"    |{state}⟩: {f*100:.3f}%")

print("\n--- JP Protocol (sf=4.0, no noise) ---")
jp_result = simulate_CZ_gate(
    simulation_inputs=jp_inputs,
    species="Rb87",
    n_rydberg=60,
    temperature=1e-6,
    spacing_factor=4.0,  # Weak blockade
    tweezer_power=0.02,
    tweezer_waist=0.8e-6,
    include_noise=False,
    verbose=True
)
print(f"\n  Fidelity: {jp_result.avg_fidelity*100:.2f}%")
print(f"  V/Ω: {jp_result.V_over_Omega:.1f}")
for state, f in jp_result.fidelities.items():
    print(f"    |{state}⟩: {f*100:.3f}%")

print("\n" + "=" * 70)
print("VERY WEAK BLOCKADE TEST (sf=6.0)")
print("=" * 70)

print("\n--- LP Protocol (sf=6.0, no noise) ---")
lp_result2 = simulate_CZ_gate(
    simulation_inputs=lp_inputs,
    species="Rb87",
    n_rydberg=60,
    temperature=1e-6,
    spacing_factor=6.0,  # Very weak blockade
    tweezer_power=0.02,
    tweezer_waist=0.8e-6,
    include_noise=False,
    verbose=True
)
print(f"\n  Fidelity: {lp_result2.avg_fidelity*100:.2f}%")
print(f"  V/Ω: {lp_result2.V_over_Omega:.1f}")
for state, f in lp_result2.fidelities.items():
    print(f"    |{state}⟩: {f*100:.3f}%")
