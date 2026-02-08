#!/usr/bin/env python3
"""Quick test of simulation."""
import sys
sys.path.insert(0, 'src')

from qpu_simulator.micro_physics.neutral_atoms.rydberg_gates import (
    simulate_CZ_gate, LPSimulationInputs, TwoPhotonExcitationConfig, 
    NoiseSourceConfig, LaserParameters
)

# Use more reasonable laser powers (much lower)
laser_1 = LaserParameters(
    power=100e-6, waist=40e-6, linewidth_hz=1000,  # 100 μW 
    polarization='sigma+', polarization_purity=0.99
)
laser_2 = LaserParameters(
    power=0.5, waist=40e-6, linewidth_hz=1000,  # 500 mW
    polarization='sigma+', polarization_purity=0.99
)
excitation = TwoPhotonExcitationConfig(
    laser_1=laser_1, laser_2=laser_2, Delta_e=2e9
)
noise_config = NoiseSourceConfig()
sim_inputs = LPSimulationInputs(
    excitation=excitation, noise=noise_config, 
    delta_over_omega=0.377, omega_tau=4.29
)

print("Running simulation...")
result = simulate_CZ_gate(
    simulation_inputs=sim_inputs, 
    species='Rb87', 
    n_rydberg=60, 
    temperature=5e-6, 
    spacing_factor=3.0, 
    tweezer_power=0.02, 
    tweezer_waist=0.8e-6, 
    include_noise=True
)

# Access as dataclass attributes
print(f'Fidelity: {result.avg_fidelity*100:.2f}%')
print(f'Gate time: {result.gate_time_ns:.0f} ns')
print(f'V/Omega: {result.V_over_Omega:.1f}')
print(f'Omega_MHz: {result.Omega_MHz:.1f}')
if result.noise_breakdown:
    print(f'Noise breakdown keys: {list(result.noise_breakdown.keys())[:5]}')
