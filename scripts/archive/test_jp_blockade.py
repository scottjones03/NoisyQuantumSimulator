"""Test JP protocol at strong blockade to understand the fidelity issue."""
import sys
sys.path.insert(0, 'src')
from qpu_simulator.micro_physics.neutral_atoms.rydberg_gates import (
    simulate_CZ_gate, LPSimulationInputs, JPSimulationInputs, TwoPhotonExcitationConfig, NoiseSourceConfig, LaserParameters
)

# Test parameters
laser_1 = LaserParameters(power=50e-6, waist=50e-6, linewidth_hz=1000, polarization='sigma+', polarization_purity=0.99)
laser_2 = LaserParameters(power=0.3, waist=50e-6, linewidth_hz=1000, polarization='sigma+', polarization_purity=0.99)
excitation = TwoPhotonExcitationConfig(laser_1=laser_1, laser_2=laser_2, Delta_e=5e9)
noise_config = NoiseSourceConfig()

print("Testing both protocols at strong blockade (sf=1.5, V/Ω~1500):")
print("="*60)

# LP with strong blockade (sf=1.5)
sim_inputs = LPSimulationInputs(excitation=excitation, noise=noise_config, delta_over_omega=None, omega_tau=None)
result = simulate_CZ_gate(simulation_inputs=sim_inputs, species='Rb87', n_rydberg=60,
                          temperature=5e-6, spacing_factor=1.5, tweezer_power=0.02, tweezer_waist=0.8e-6,
                          include_noise=False, verbose=False)
print(f'LP at sf=1.5: F={result.avg_fidelity*100:.2f}%, V/Ω={result.V_over_Omega:.1f}, τ={result.gate_time_us*1000:.0f}ns')

# JP with strong blockade (sf=1.5)
sim_inputs = JPSimulationInputs(excitation=excitation, noise=noise_config, omega_tau=None)
result = simulate_CZ_gate(simulation_inputs=sim_inputs, species='Rb87', n_rydberg=60,
                          temperature=5e-6, spacing_factor=1.5, tweezer_power=0.02, tweezer_waist=0.8e-6,
                          include_noise=False, verbose=False)
print(f'JP at sf=1.5: F={result.avg_fidelity*100:.2f}%, V/Ω={result.V_over_Omega:.1f}, τ={result.gate_time_us*1000:.0f}ns')

print("\nTesting at optimal blockade (sf=2.8, V/Ω~30):")
print("="*60)

# LP with optimal blockade
sim_inputs = LPSimulationInputs(excitation=excitation, noise=noise_config, delta_over_omega=None, omega_tau=None)
result = simulate_CZ_gate(simulation_inputs=sim_inputs, species='Rb87', n_rydberg=60,
                          temperature=5e-6, spacing_factor=2.8, tweezer_power=0.02, tweezer_waist=0.8e-6,
                          include_noise=False, verbose=False)
print(f'LP at sf=2.8: F={result.avg_fidelity*100:.2f}%, V/Ω={result.V_over_Omega:.1f}, τ={result.gate_time_us*1000:.0f}ns')

# JP with optimal blockade
sim_inputs = JPSimulationInputs(excitation=excitation, noise=noise_config, omega_tau=None)
result = simulate_CZ_gate(simulation_inputs=sim_inputs, species='Rb87', n_rydberg=60,
                          temperature=5e-6, spacing_factor=2.8, tweezer_power=0.02, tweezer_waist=0.8e-6,
                          include_noise=False, verbose=False)
print(f'JP at sf=2.8: F={result.avg_fidelity*100:.2f}%, V/Ω={result.V_over_Omega:.1f}, τ={result.gate_time_us*1000:.0f}ns')
