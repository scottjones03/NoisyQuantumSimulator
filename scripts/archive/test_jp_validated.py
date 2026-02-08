#!/usr/bin/env python3
"""Quick test of validated JP parameters"""

import numpy as np
from qutip import tensor, mesolve, basis
import sys
sys.path.insert(0, 'src')

from qpu_simulator.micro_physics.neutral_atoms.rydberg_gates.hamiltonians import (
    build_hilbert_space, build_phase_modulated_hamiltonian,
)
from qpu_simulator.micro_physics.neutral_atoms.rydberg_gates.protocols import (
    JP_DEFAULT, JP_SWITCHING_TIMES_VALIDATED, JP_PHASES_VALIDATED, JP_OMEGA_TAU_VALIDATED
)

hs = build_hilbert_space(3)
b0, b1 = hs.basis['0'], hs.basis['1']
psi_01, psi_11 = tensor(b0, b1), tensor(b1, b1)
Omega = 2 * np.pi * 5e6
V = 200 * Omega

def get_overlap(psi1, psi2):
    result = psi1.dag() * psi2
    if hasattr(result, 'tr'): return result.tr()
    elif hasattr(result, 'full'): return complex(result.full()[0, 0])
    return complex(result)

print("Testing JP_DEFAULT parameters from protocols.py")
print(f"  omega_tau = {JP_DEFAULT.omega_tau}")
print(f"  switching_times = {JP_DEFAULT.switching_times}")
print(f"  phases = {[f'{p/np.pi:.2f}π' for p in JP_DEFAULT.phases]}")

# Simulate
omega_tau = JP_DEFAULT.omega_tau
switching = JP_DEFAULT.switching_times
phases = JP_DEFAULT.phases

tau = omega_tau / Omega
boundaries = [0] + [t/Omega for t in switching] + [tau]

psi_01_f, psi_11_f = psi_01, psi_11
for i, phase in enumerate(phases):
    dt = boundaries[i+1] - boundaries[i]
    H = build_phase_modulated_hamiltonian(Omega=Omega, phase=phase, V=V, hs=hs, Delta=0)
    tlist = np.linspace(0, dt, 30)
    psi_01_f = mesolve(H, psi_01_f, tlist, c_ops=[]).states[-1]
    psi_11_f = mesolve(H, psi_11_f, tlist, c_ops=[]).states[-1]

ov01, ov11 = get_overlap(psi_01, psi_01_f), get_overlap(psi_11, psi_11_f)
ctrl = np.angle(ov11) - 2*np.angle(ov01)
ctrl = np.arctan2(np.sin(ctrl), np.cos(ctrl))
phase_err = min(abs(ctrl - np.pi), abs(ctrl + np.pi))
fid = abs(ov01)**2 * abs(ov11)**2 * np.cos(phase_err/2)**2

print(f"\nResults:")
print(f"  |01⟩ overlap = {abs(ov01):.4f}")
print(f"  |11⟩ overlap = {abs(ov11):.4f}")
print(f"  Controlled phase = {np.degrees(ctrl):.1f}°")
print(f"  Fidelity = {fid:.4f}")

if fid > 0.95:
    print(f"\n✓ JP_DEFAULT parameters work! Fidelity = {fid:.4f}")
else:
    print(f"\n✗ Fidelity too low: {fid:.4f}")
