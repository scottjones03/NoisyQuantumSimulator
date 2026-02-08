#!/usr/bin/env python3
"""Sweep apparatus parameters to find V/Omega in JP-validated range [10-200]."""

import numpy as np
from src.qpu_simulator.micro_physics.neutral_atoms.rydberg_gates.laser_physics import (
    two_photon_rabi, single_photon_rabi, laser_E0, rydberg_blockade
)
from src.qpu_simulator.micro_physics.neutral_atoms.rydberg_gates.atom_database import (
    get_C6, ATOM_DB, get_default_intermediate_state
)
from src.qpu_simulator.micro_physics.neutral_atoms.rydberg_gates.trap_physics import (
    tweezer_spacing
)


def compute_v_omega(laser1_power, laser2_power, Delta_e, spacing, n_rydberg=70, waist=50e-6, species="Rb87"):
    """Compute V/Omega and Omega for given apparatus parameters."""
    atom = ATOM_DB[species]
    
    # Electric fields
    E0_1 = laser_E0(laser1_power, waist)
    E0_2 = laser_E0(laser2_power, waist)
    
    # Dipole moments
    intermediate_state = get_default_intermediate_state(species)
    dipole_1e = atom["intermediate_states"][intermediate_state]["dipole_from_ground"]
    n_ref = atom["n_ref"]
    dipole_er_ref = atom["dipole_intermediate_to_rydberg_ref"]
    dipole_er = dipole_er_ref * (n_rydberg / n_ref)**(-1.5)
    
    # Rabi frequencies
    Omega1 = single_photon_rabi(dipole_1e, E0_1)
    Omega2 = single_photon_rabi(dipole_er, E0_2)
    Omega = two_photon_rabi(Omega1, Omega2, Delta_e)
    
    # Blockade
    C6 = get_C6(n_rydberg, species)
    trap_wavelength = atom["trap_wavelength"]
    R = tweezer_spacing(trap_wavelength, 0.5, spacing)
    V = rydberg_blockade(C6, R)
    
    return V / Omega, Omega / (2 * np.pi * 1e6)


print("=== Varying Delta_e (GHz) at default apparatus ===")
for d in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]:
    vo, om = compute_v_omega(50e-6, 0.3, 2 * np.pi * d * 1e9, 2.8)
    marker = " <-- in range" if 10 <= vo <= 200 else ""
    print(f"  Delta_e = {d:.1f} GHz -> V/Omega = {vo:.1f}, Omega/2pi = {om:.2f} MHz{marker}")

print()
print("=== Varying spacing factor at Delta_e = 1 GHz ===")
for sp in [1.5, 2.0, 2.5, 2.8, 3.0, 3.5, 4.0]:
    vo, om = compute_v_omega(50e-6, 0.3, 2 * np.pi * 1e9, sp)
    marker = " <-- in range" if 10 <= vo <= 200 else ""
    print(f"  spacing = {sp:.1f} -> V/Omega = {vo:.1f}{marker}")

print()
print("=== Varying laser 2 power at Delta_e = 1 GHz, spacing = 2.8 ===")
for p2 in [0.1, 0.3, 0.5, 1.0, 2.0, 5.0]:
    vo, om = compute_v_omega(50e-6, p2, 2 * np.pi * 1e9, 2.8)
    marker = " <-- in range" if 10 <= vo <= 200 else ""
    print(f"  P2 = {p2*1000:.0f} mW -> V/Omega = {vo:.1f}, Omega/2pi = {om:.2f} MHz{marker}")

print()
print("=== Practical combos for V/Omega ~ 100-200 ===")
combos = [
    (50e-6, 0.3, 0.5, 2.8, "spacing=2.8, De=0.5GHz, P2=300mW"),
    (50e-6, 1.0, 1.0, 2.8, "spacing=2.8, De=1GHz, P2=1W"),
    (50e-6, 0.3, 1.0, 2.0, "spacing=2.0, De=1GHz, P2=300mW"),
    (100e-6, 0.3, 0.7, 2.8, "spacing=2.8, De=0.7GHz, P1=100uW"),
    (50e-6, 0.3, 0.3, 2.8, "spacing=2.8, De=0.3GHz, P2=300mW"),
    (50e-6, 2.0, 1.0, 2.8, "spacing=2.8, De=1GHz, P2=2W"),
]
for p1, p2, de_ghz, sp, label in combos:
    vo, om = compute_v_omega(p1, p2, 2 * np.pi * de_ghz * 1e9, sp)
    marker = " *** GOOD ***" if 100 <= vo <= 200 else (" <-- in range" if 10 <= vo <= 200 else "")
    print(f"  {label}: V/Omega = {vo:.1f}, Omega/2pi = {om:.2f} MHz{marker}")
