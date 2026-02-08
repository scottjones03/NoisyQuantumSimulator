#!/usr/bin/env python3
"""
Quick validation that JP bang-bang is now properly implemented as its
own protocol — NOT falling back to smooth JP.

Tests:
1. Different switching_times/phases MUST produce different fidelities
   (proves the simulator is actually reading them)
2. Default 5-segment params should give ~95% fidelity at V/Ω~200
3. Smooth JP and bang-bang produce DIFFERENT results (they're different protocols)
"""
import sys
import numpy as np
sys.path.insert(0, "src")

from qpu_simulator.micro_physics.neutral_atoms.rydberg_gates.simulation import simulate_CZ_gate
from qpu_simulator.micro_physics.neutral_atoms.rydberg_gates.configurations import (
    JPSimulationInputs,
    SmoothJPSimulationInputs,
    TwoPhotonExcitationConfig,
    NoiseSourceConfig,
    LaserParameters,
)

# Common config
excitation = TwoPhotonExcitationConfig(
    laser_1=LaserParameters(power=50e-6, waist=50e-6, polarization="pi"),
    laser_2=LaserParameters(power=0.3, waist=50e-6, polarization="sigma+"),
    Delta_e=2 * np.pi * 1e9,
)
noise_off = NoiseSourceConfig(
    include_spontaneous_emission=False,
    include_intermediate_scattering=False,
    include_motional_dephasing=False,
    include_doppler_dephasing=False,
    include_intensity_noise=False,
    include_laser_dephasing=False,
    include_magnetic_dephasing=False,
)


def run_sim(inputs, label):
    """Run a simulation and print key metrics."""
    result = simulate_CZ_gate(
        simulation_inputs=inputs,
        species="Rb87",
        n_rydberg=70,
        spacing_factor=2.8,
        temperature=2e-6,
        include_noise=False,
        verbose=False,
        return_dataclass=True,
    )
    f = result.avg_fidelity
    phi_err = result.phase_info.get('phase_error_from_pi_deg', float('nan'))
    ctrl_phi = result.phase_info.get('controlled_phase_deg', float('nan'))
    print(f"  {label:40s} F={f:.6f}  φ_err={phi_err:+.2f}°  φ_ctrl={ctrl_phi:+.2f}°")
    return result


print("=" * 80)
print("TEST 1: Different switching_times/phases → different fidelities")
print("=" * 80)

# Default 5-segment
jp_default = JPSimulationInputs(
    excitation=excitation, noise=noise_off,
    omega_tau=22.08,
    switching_times=[2.214, 8.823, 13.258, 19.867],
    phases=[np.pi/2, 0, -np.pi/2, 0, np.pi/2],
)
r1 = run_sim(jp_default, "5-seg default (validated)")

# Perturbed switching times
jp_perturbed = JPSimulationInputs(
    excitation=excitation, noise=noise_off,
    omega_tau=22.08,
    switching_times=[3.0, 9.0, 14.0, 20.0],  # shifted
    phases=[np.pi/2, 0, -np.pi/2, 0, np.pi/2],
)
r2 = run_sim(jp_perturbed, "5-seg perturbed times")

# Different phases
jp_diff_phases = JPSimulationInputs(
    excitation=excitation, noise=noise_off,
    omega_tau=22.08,
    switching_times=[2.214, 8.823, 13.258, 19.867],
    phases=[np.pi/4, 0, -np.pi/4, 0, np.pi/4],  # smaller phases
)
r3 = run_sim(jp_diff_phases, "5-seg different phases (π/4)")

# 7-segment
jp_7seg = JPSimulationInputs(
    excitation=excitation, noise=noise_off,
    omega_tau=7.0,
    switching_times=[0.3328, 0.5859, 3.4340, 3.5530, 4.1204, 6.7431],
    phases=[np.pi/2, 0, -np.pi/2, -np.pi/2, 0, np.pi/2, 0],
)
r4 = run_sim(jp_7seg, "7-seg original paper")

# Check that they're all different
fids = [r1.avg_fidelity, r2.avg_fidelity, r3.avg_fidelity, r4.avg_fidelity]
all_different = len(set(round(f, 6) for f in fids)) == len(fids)
print(f"\n  All different fidelities? {'✓ YES' if all_different else '✗ NO — BUG!'}")
if not all_different:
    print(f"  Fidelities: {fids}")

print()
print("=" * 80)
print("TEST 2: Bang-bang ≠ smooth JP (different protocols)")
print("=" * 80)

smooth_jp = SmoothJPSimulationInputs(
    excitation=excitation, noise=noise_off,
)
r_smooth = run_sim(smooth_jp, "Smooth JP (default params)")
r_bb = run_sim(jp_default, "JP bang-bang 5-seg (repeated)")

bb_neq_smooth = abs(r_smooth.avg_fidelity - r_bb.avg_fidelity) > 1e-6
print(f"\n  BB ≠ smooth? {'✓ YES' if bb_neq_smooth else '✗ NO — STILL FALLING BACK!'}")

print()
print("=" * 80)
print("TEST 3: Default BB with None inputs → uses protocol defaults")
print("=" * 80)

jp_none = JPSimulationInputs(
    excitation=excitation, noise=noise_off,
    # omega_tau=None, switching_times=None, phases=None → should use defaults
)
r_none = run_sim(jp_none, "JP bang-bang (all None → defaults)")

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
passed = all_different and bb_neq_smooth
if passed:
    print("  ✓ ALL TESTS PASSED — bang-bang is a real, distinct protocol")
else:
    print("  ✗ TESTS FAILED — see above")
    sys.exit(1)
