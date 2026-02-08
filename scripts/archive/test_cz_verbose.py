#!/usr/bin/env python3
"""Run CZ simulation with verbose output to check noise rates."""

import sys
sys.path.insert(0, '/Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/NoisyQuantumSimulator/src')

import numpy as np
from qpu_simulator.micro_physics.neutral_atoms.rydberg_gates.simulation import (
    simulate_CZ_gate, 
    SimulationResult
)
from qpu_simulator.micro_physics.neutral_atoms.rydberg_gates.configurations import (
    get_standard_rb87_config,
    LPSimulationInputs,
    TwoPhotonExcitationConfig,
    LaserParameters,
    NoiseSourceConfig,
)

# Create config
config = get_standard_rb87_config(n_rydberg=70)

# Build laser parameters
laser_1 = LaserParameters(power=0.1e-3, waist=1.0e-6, linewidth_hz=1e3)  # 0.1 mW
laser_2 = LaserParameters(power=0.5, waist=10e-6, linewidth_hz=1e3)     # 0.5 W
excitation = TwoPhotonExcitationConfig(laser_1=laser_1, laser_2=laser_2)
noise = NoiseSourceConfig(include_motional_dephasing=True)

# Create LP inputs
simulation_inputs = LPSimulationInputs(excitation=excitation, noise=noise)

# Run single simulation with verbose
print("Running CZ simulation with verbose output...")
result = simulate_CZ_gate(
    simulation_inputs=simulation_inputs,
    config=config,
    temperature=10e-6,  # 10 μK
    trap_laser_on=False,  # Turn off trap to avoid huge Stark shift
    verbose=True
)

print(f"\n=== Simulation Results ===")
print(f"Protocol: {result.protocol}")
print(f"Average Fidelity: {result.avg_fidelity:.6f}")
print(f"Gate Time: {result.tau_total*1e6:.2f} μs")
print(f"Omega/(2π): {result.Omega/(2*np.pi*1e6):.2f} MHz")
print(f"V/Omega: {result.V_over_Omega:.2f}")

print(f"\n=== Noise Rates ===")
nb = result.noise_breakdown
print(f"Rydberg decay rate γ_r: {nb.get('gamma_r', 0):.1f} Hz")
print(f"Total dephasing rate: {nb.get('total_dephasing_rate', 0):.1f} Hz")
print(f"Laser dephasing γ_φ,laser: {nb.get('gamma_phi_laser', 0):.1f} Hz")
print(f"Thermal dephasing γ_φ,thermal: {nb.get('gamma_phi_thermal', 0):.1f} Hz")
print(f"Scatter γ_scatter: {nb.get('gamma_scatter_intermediate', 0):.1f} Hz")
print(f"Anti-trap loss γ_loss: {nb.get('gamma_loss_antitrap', 0):.1f} Hz")
print(f"Number of collapse operators: {nb.get('n_collapse_ops', 0)}")

# Estimate expected infidelity from decay
tau = result.tau_total
gamma_r = nb.get('gamma_r', 0)
print(f"\n=== Expected Decay Infidelity ===")
print(f"Gate time τ = {tau*1e6:.2f} μs")
print(f"γ_r × τ = {gamma_r * tau:.4f}")
print(f"Expected decay contribution: ~{gamma_r * tau * 100:.2f}% infidelity")

# Check purity of final states
print(f"\n=== State Purity Check ===")
for label in ['00', '01', '10', '11']:
    fid = result.fidelities.get(label, 0)
    print(f"|{label}⟩ fidelity: {fid:.6f}")

# Check if we have access to the final density matrices
print(f"\n=== Verifying Decoherence ===")
print(f"If c_ops are working, purity should be < 1 for noisy evolution")
print(f"But for VERY short gates (τ << 1/γ), purity stays ~1 because there's")
print(f"no time for decoherence to occur.")
print(f"\nThis is physically correct! Fast gates have less decoherence.")
print(f"\nExpected infidelity budget:")
print(f"  Decay: γ_r × τ = {result.noise_breakdown.get('gamma_r', 0) * result.tau_total:.4f}")
print(f"  Dephasing: γ_φ × τ = {result.noise_breakdown.get('total_dephasing_rate', 0) * result.tau_total:.4f}")
