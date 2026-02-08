#!/usr/bin/env python3
"""
Verify simulation results against experimental literature values.

Literature references:
- Levine et al., PRL 123, 170503 (2019): "High-Fidelity Control of Rydberg Gates"
  - Rb87, n=70, LP protocol, F = 97.4(3)% for CZ gate
  - Gate time ~400 ns (two-pulse), Ω/2π ~ 4 MHz, spacing ~6 μm
  
- Evered et al., Nature 622, 268 (2023): "High-fidelity parallel entangling gates"
  - Rb87, n=61, two-pulse protocol, F = 99.5% (postselected)
  - Gate time ~350 ns, spacing ~4 μm
  
- Bluvstein et al., Nature 604, 451 (2022): "A quantum processor based on coherent transport"
  - Rb87, n=70, F ~ 97-99% for CZ
  - Used with atom transport

- Jandura & Pupillo, PRX Quantum 3, 010353 (2022): "Time-Optimal Two- and Three-Qubit Gates"
  - Theoretical optimal control, time-optimal single-pulse CZ
  - Predicts ~50% faster than LP for same fidelity

- Graham et al., Phys. Rev. Lett. 123, 230501 (2019): "Rydberg-Mediated Entanglement"
  - Cs133 Rydberg gates, n=70, F = 89% raw, 97.5% SPAM-corrected
"""

import numpy as np
from src.qpu_simulator.micro_physics.neutral_atoms.rydberg_gates import (
    simulate_CZ_gate,
    LPSimulationInputs,
    JPSimulationInputs,
    TwoPhotonExcitationConfig,
    LaserParameters,
    NoiseSourceConfig,
)

def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def print_comparison(param, simulated, literature, unit=""):
    status = "✓" if abs(simulated - literature) / max(literature, 1e-10) < 0.3 else "✗"
    print(f"  {param:30s}: Sim = {simulated:8.4f} {unit:5s} | Lit = {literature:8.4f} {unit:5s} [{status}]")

# =============================================================================
# LEVINE ET AL. PRL 2019 - LEVINE-PICHLER PROTOCOL
# =============================================================================
print_header("LEVINE ET AL. PRL 2019 - LEVINE-PICHLER PROTOCOL")
print("Reference: PRL 123, 170503 (2019)")
print("Key results: Rb87, n=70, CZ fidelity = 97.4(3)%, gate time ~400 ns")
print()

# Replicate their experimental conditions
laser_1_levine = LaserParameters(power=50e-6, waist=50e-6, linewidth_hz=1000)
laser_2_levine = LaserParameters(power=500e-3, waist=50e-6, linewidth_hz=1000)
excitation_levine = TwoPhotonExcitationConfig(
    laser_1=laser_1_levine,
    laser_2=laser_2_levine,
    Delta_e=2*np.pi*1e9,
)
noise_levine = NoiseSourceConfig(include_motional_dephasing=True)
lp_inputs_levine = LPSimulationInputs(
    excitation=excitation_levine,
    noise=noise_levine,
    pulse_shape="square",
)

levine_result = simulate_CZ_gate(
    simulation_inputs=lp_inputs_levine,
    species="Rb87",
    n_rydberg=70,
    temperature=10e-6,          # ~10 μK typical
    spacing_factor=3.5,         # ~6 μm spacing
    tweezer_power=30e-3,
    tweezer_waist=1.0e-6,
    include_noise=True,
)

print("Simulated vs Literature:")
print_comparison("Gate fidelity", levine_result.avg_fidelity, 0.974, "")
print_comparison("Gate time", levine_result.tau_total * 1e9, 400, "ns")
print_comparison("Ω/2π", levine_result.Omega / (2*np.pi*1e6), 4.0, "MHz")
print_comparison("V/Ω ratio", levine_result.V_over_Omega, 50, "")
print()
print(f"  Per-state fidelities: {levine_result.fidelities}")

# =============================================================================
# EVERED ET AL. NATURE 2023 - HIGH-FIDELITY GATES
# =============================================================================
print_header("EVERED ET AL. NATURE 2023 - HIGH-FIDELITY GATES")
print("Reference: Nature 622, 268 (2023)")
print("Key results: Rb87, n=61, CZ fidelity = 99.5% (postselected), ~350 ns")
print()

# Their setup - very optimized
laser_1_evered = LaserParameters(power=100e-6, waist=50e-6, linewidth_hz=100)
laser_2_evered = LaserParameters(power=800e-3, waist=50e-6, linewidth_hz=100)
excitation_evered = TwoPhotonExcitationConfig(
    laser_1=laser_1_evered,
    laser_2=laser_2_evered,
    Delta_e=2*np.pi*1e9,
)
noise_evered = NoiseSourceConfig(include_motional_dephasing=True)
lp_inputs_evered = LPSimulationInputs(
    excitation=excitation_evered,
    noise=noise_evered,
    pulse_shape="square",
)

evered_result = simulate_CZ_gate(
    simulation_inputs=lp_inputs_evered,
    species="Rb87",
    n_rydberg=61,               # n=61 for better lifetime
    temperature=2e-6,           # Very cold ~2 μK
    spacing_factor=2.5,         # ~4 μm spacing
    tweezer_power=30e-3,
    tweezer_waist=0.8e-6,       # Tighter tweezers
    include_noise=True,
)

print("Simulated vs Literature:")
print_comparison("Gate fidelity", evered_result.avg_fidelity, 0.995, "")
print_comparison("Gate time", evered_result.tau_total * 1e9, 350, "ns")
print()
print(f"  Note: Literature value is postselected (loss-free subspace)")
print(f"  Per-state fidelities: {evered_result.fidelities}")

# =============================================================================
# JANDURA-PUPILLO PRX QUANTUM 2022 - TIME-OPTIMAL PROTOCOL
# =============================================================================
print_header("JANDURA-PUPILLO PRX QUANTUM 2022 - TIME-OPTIMAL PROTOCOL")
print("Reference: PRX Quantum 3, 010353 (2022)")
print("Key results: Single-pulse CZ, ~50% faster than LP for same fidelity")
print()

# Compare LP vs JP with identical parameters
laser_1_base = LaserParameters(power=50e-6, waist=50e-6, linewidth_hz=500)
laser_2_base = LaserParameters(power=500e-3, waist=50e-6, linewidth_hz=500)
excitation_base = TwoPhotonExcitationConfig(
    laser_1=laser_1_base,
    laser_2=laser_2_base,
    Delta_e=2*np.pi*1e9,
)
noise_base = NoiseSourceConfig(include_motional_dephasing=True)

lp_inputs = LPSimulationInputs(
    excitation=excitation_base,
    noise=noise_base,
    pulse_shape="square",
)
jp_inputs = JPSimulationInputs(
    excitation=excitation_base,
    noise=noise_base,
)

lp_result = simulate_CZ_gate(
    simulation_inputs=lp_inputs,
    species="Rb87",
    n_rydberg=70,
    temperature=5e-6,
    spacing_factor=3.0,
    tweezer_power=30e-3,
    tweezer_waist=1.0e-6,
    include_noise=True,
)
jp_result = simulate_CZ_gate(
    simulation_inputs=jp_inputs,
    species="Rb87",
    n_rydberg=70,
    temperature=5e-6,
    spacing_factor=3.0,
    tweezer_power=30e-3,
    tweezer_waist=1.0e-6,
    include_noise=True,
)

print("LP vs JP comparison (same parameters):")
print(f"  {'Protocol':<25s} {'Fidelity':>10s} {'Gate time':>12s} {'V/Ω':>8s}")
print(f"  {'-'*55}")
print(f"  {'Levine-Pichler (2-pulse)':<25s} {lp_result.avg_fidelity:>10.4f} {lp_result.tau_total*1e9:>10.1f} ns {lp_result.V_over_Omega:>8.1f}")
print(f"  {'Jandura-Pupillo (1-pulse)':<25s} {jp_result.avg_fidelity:>10.4f} {jp_result.tau_total*1e9:>10.1f} ns {jp_result.V_over_Omega:>8.1f}")
print()

time_ratio = lp_result.tau_total / jp_result.tau_total
print(f"  Time ratio (LP/JP): {time_ratio:.2f}× (literature predicts ~1.5-2×)")
fid_diff = jp_result.avg_fidelity - lp_result.avg_fidelity
print(f"  Fidelity difference (JP-LP): {fid_diff*100:+.2f}%")

# =============================================================================
# GRAHAM ET AL. PRL 2019 - CESIUM GATES
# =============================================================================
print_header("GRAHAM ET AL. PRL 2019 - CESIUM GATES")
print("Reference: PRL 123, 230501 (2019)")
print("Key results: Cs133, n=70, CZ fidelity = 89% raw / 97.5% SPAM-corrected")
print()

laser_1_graham = LaserParameters(power=50e-6, waist=50e-6, linewidth_hz=2000)
laser_2_graham = LaserParameters(power=500e-3, waist=50e-6, linewidth_hz=2000)
excitation_graham = TwoPhotonExcitationConfig(
    laser_1=laser_1_graham,
    laser_2=laser_2_graham,
    Delta_e=2*np.pi*1e9,
)
noise_graham = NoiseSourceConfig(include_motional_dephasing=True)
lp_inputs_graham = LPSimulationInputs(
    excitation=excitation_graham,
    noise=noise_graham,
    pulse_shape="square",
)

graham_result = simulate_CZ_gate(
    simulation_inputs=lp_inputs_graham,
    species="Cs133",
    n_rydberg=70,
    temperature=10e-6,
    spacing_factor=3.5,
    tweezer_power=30e-3,
    tweezer_waist=1.0e-6,
    include_noise=True,
)

print("Simulated vs Literature:")
print_comparison("Gate fidelity", graham_result.avg_fidelity, 0.975, "")  # SPAM-corrected
print_comparison("Gate time", graham_result.tau_total * 1e9, 400, "ns")
print()
print(f"  Note: Cs133 has stronger blockade (larger C6) but different transition wavelengths")
print(f"  Per-state fidelities: {graham_result.fidelities}")

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print_header("SUMMARY: SIMULATION vs LITERATURE")
print()
print(f"  {'Experiment':<30s} {'Lit. F':>10s} {'Sim. F':>10s} {'Match':>10s}")
print(f"  {'-'*65}")
print(f"  {'Levine 2019 (Rb87, LP)':<30s} {'97.4%':>10s} {levine_result.avg_fidelity*100:>9.1f}% {'✓' if abs(levine_result.avg_fidelity - 0.974) < 0.03 else '✗':>10s}")
print(f"  {'Evered 2023 (Rb87, LP)':<30s} {'99.5%':>10s} {evered_result.avg_fidelity*100:>9.1f}% {'✓' if abs(evered_result.avg_fidelity - 0.995) < 0.02 else '~':>10s}")
print(f"  {'Graham 2019 (Cs133, LP)':<30s} {'97.5%':>10s} {graham_result.avg_fidelity*100:>9.1f}% {'✓' if abs(graham_result.avg_fidelity - 0.975) < 0.03 else '✗':>10s}")
print()
print("Note: Literature fidelities often include postselection or SPAM correction.")
print("Our simulation reports raw gate fidelity without postselection.")
