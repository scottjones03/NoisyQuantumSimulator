#!/usr/bin/env python3
"""Diagnose V/Omega for different Delta_e values."""
import numpy as np
import sys
sys.path.insert(0, '.')
from src.qpu_simulator.micro_physics.neutral_atoms.rydberg_gates.laser_physics import (
    laser_E0, single_photon_rabi, two_photon_rabi
)
from src.qpu_simulator.micro_physics.neutral_atoms.rydberg_gates.simulation import (
    get_atom_properties, get_default_intermediate_state, tweezer_spacing
)
from src.qpu_simulator.micro_physics.neutral_atoms.rydberg_gates.configurations import (
    AtomicConfiguration
)
from src.qpu_simulator.micro_physics.neutral_atoms.rydberg_gates.simulation import (
    get_C6, rydberg_blockade
)

# Fixed laser params (same as both optimizers)
P1, w1 = 50e-6, 50e-6
P2, w2 = 0.3, 50e-6
n_rydberg = 70
species = "Rb87"
NA = 0.5
spacing_factor = 2.8

config = AtomicConfiguration(species=species, n_rydberg=n_rydberg, L_rydberg="S")
atom = get_atom_properties(species)
trap_wavelength = atom["trap_wavelength"]
R = tweezer_spacing(trap_wavelength, NA, spacing_factor)

E0_1 = laser_E0(P1, w1)
E0_2 = laser_E0(P2, w2)

intermediate_state = get_default_intermediate_state(species)
dipole_1e = atom["intermediate_states"][intermediate_state]["dipole_from_ground"]
n_ref = atom["n_ref"]
dipole_er_ref = atom["dipole_intermediate_to_rydberg_ref"]
dipole_er = dipole_er_ref * (n_rydberg / n_ref)**(-1.5)

Omega1 = single_photon_rabi(dipole_1e, E0_1)
Omega2 = single_photon_rabi(dipole_er, E0_2)

print(f"Omega1/2pi = {Omega1/(2*np.pi*1e6):.2f} MHz")
print(f"Omega2/2pi = {Omega2/(2*np.pi*1e6):.2f} MHz")
print(f"R = {R*1e6:.2f} um")

# C6 coefficient
C6 = get_C6(n_rydberg, species)
V = rydberg_blockade(C6, R)
print(f"C6 = {C6:.3e}")
print(f"V/2pi = {V/(2*np.pi*1e6):.2f} MHz")
print()

for Delta_e_GHz in [0.8, 1.0, 2.0, 3.0, 5.0, 7.8, 10.0]:
    Delta_e_rads = 2*np.pi*Delta_e_GHz*1e9
    Omega = two_photon_rabi(Omega1, Omega2, Delta_e_rads)
    V_over_Omega = V / Omega
    Omega_MHz = Omega / (2*np.pi*1e6)
    print(f"Delta_e = {Delta_e_GHz:5.1f} GHz  ->  Omega/2pi = {Omega_MHz:8.3f} MHz,  V/Omega = {V_over_Omega:8.1f}")

print()
print("--- Old optimizer bug: Delta_e in plain Hz not rad/s ---")
Delta_e_buggy = 5e9  # plain Hz, not rad/s!
Omega_buggy = two_photon_rabi(Omega1, Omega2, Delta_e_buggy)
V_over_Omega_buggy = V / Omega_buggy
print(f"Delta_e = 5e9 Hz (bug)  ->  Omega/2pi = {Omega_buggy/(2*np.pi*1e6):.3f} MHz,  V/Omega = {V_over_Omega_buggy:.1f}")
