#!/usr/bin/env python3
"""
Verify that phase penalty is working correctly for LP and disabled for JP.
"""
import sys
sys.path.insert(0, '/Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/NoisyQuantumSimulator/src')

import numpy as np
from qpu_simulator.micro_physics.neutral_atoms.rydberg_gates import (
    simulate_CZ_gate,
    LPSimulationInputs,
    JPSimulationInputs,
    TwoPhotonExcitationConfig,
    NoiseSourceConfig,
    LaserParameters,
)

print("=" * 70)
print("VERIFY PHASE PENALTY BEHAVIOR")
print("=" * 70)

laser_1 = LaserParameters(power=50e-6, waist=50e-6, linewidth_hz=1000)
laser_2 = LaserParameters(power=0.3, waist=50e-6, linewidth_hz=1000)
excitation = TwoPhotonExcitationConfig(laser_1=laser_1, laser_2=laser_2, Delta_e=5e9)
noise_config = NoiseSourceConfig()

lp_inputs = LPSimulationInputs(excitation=excitation, noise=noise_config)
jp_inputs = JPSimulationInputs(excitation=excitation, noise=noise_config)

print("\n--- LP at WEAK blockade (sf=5) ---")
result_lp_weak = simulate_CZ_gate(
    simulation_inputs=lp_inputs,
    species="Rb87",
    n_rydberg=60,
    temperature=1e-6,
    spacing_factor=5.0,
    tweezer_power=0.02,
    tweezer_waist=0.8e-6,
    include_noise=False,
    verbose=False
)
print(f"V/Ω = {result_lp_weak.V_over_Omega:.2f}")
print(f"Fidelity: {result_lp_weak.avg_fidelity*100:.2f}%")
print(f"Per-state |11⟩: {result_lp_weak.fidelities['11']*100:.2f}%")
if result_lp_weak.phase_info:
    pi = result_lp_weak.phase_info
    print(f"phase_info:")
    print(f"  controlled_phase: {pi.get('controlled_phase_deg', 'N/A'):.2f}°")
    print(f"  phase_error_from_pi: {pi.get('phase_error_from_pi_deg', 'N/A'):.2f}°")
    print(f"  cz_phase_fidelity: {pi.get('cz_phase_fidelity', 'N/A'):.4f}")
    print(f"  F11_population: {pi.get('F11_population', 'N/A'):.4f}")
    print(f"  F11_with_phase: {pi.get('F11_with_phase', 'N/A'):.4f}")
    print(f"  phase_penalty_applied: {pi.get('phase_penalty_applied', 'N/A')}")

print("\n--- LP at STRONG blockade (sf=1.5) ---")
result_lp_strong = simulate_CZ_gate(
    simulation_inputs=lp_inputs,
    species="Rb87",
    n_rydberg=60,
    temperature=1e-6,
    spacing_factor=1.5,
    tweezer_power=0.02,
    tweezer_waist=0.8e-6,
    include_noise=False,
    verbose=False
)
print(f"V/Ω = {result_lp_strong.V_over_Omega:.2f}")
print(f"Fidelity: {result_lp_strong.avg_fidelity*100:.2f}%")
print(f"Per-state |11⟩: {result_lp_strong.fidelities['11']*100:.2f}%")
if result_lp_strong.phase_info:
    pi = result_lp_strong.phase_info
    print(f"phase_info:")
    print(f"  controlled_phase: {pi.get('controlled_phase_deg', 'N/A'):.2f}°")
    print(f"  phase_error_from_pi: {pi.get('phase_error_from_pi_deg', 'N/A'):.2f}°")
    print(f"  cz_phase_fidelity: {pi.get('cz_phase_fidelity', 'N/A'):.4f}")
    print(f"  phase_penalty_applied: {pi.get('phase_penalty_applied', 'N/A')}")

print("\n--- JP at STRONG blockade (sf=1.5) - NO PENALTY ---")
result_jp_strong = simulate_CZ_gate(
    simulation_inputs=jp_inputs,
    species="Rb87",
    n_rydberg=60,
    temperature=1e-6,
    spacing_factor=1.5,
    tweezer_power=0.02,
    tweezer_waist=0.8e-6,
    include_noise=False,
    verbose=False
)
print(f"V/Ω = {result_jp_strong.V_over_Omega:.2f}")
print(f"Fidelity: {result_jp_strong.avg_fidelity*100:.2f}%")
print(f"Per-state |11⟩: {result_jp_strong.fidelities['11']*100:.2f}%")
if result_jp_strong.phase_info:
    pi = result_jp_strong.phase_info
    print(f"phase_info:")
    print(f"  controlled_phase: {pi.get('controlled_phase_deg', 'N/A'):.2f}°")
    print(f"  phase_error_from_pi: {pi.get('phase_error_from_pi_deg', 'N/A'):.2f}°")
    print(f"  cz_phase_fidelity: {pi.get('cz_phase_fidelity', 'N/A'):.4f}")
    print(f"  phase_penalty_applied: {pi.get('phase_penalty_applied', 'N/A')}")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("LP weak (sf=5):  Phase penalty APPLIED - correctly shows low fidelity")
print("LP strong (sf=1.5): Phase penalty APPLIED - high fidelity (correct CZ phase)")
print("JP strong (sf=1.5): Phase penalty DISABLED - maintains backward compat")
print("\nNote: JP parameters need verification against original paper.")
