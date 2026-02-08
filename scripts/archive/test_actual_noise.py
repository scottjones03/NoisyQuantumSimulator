#!/usr/bin/env python3
"""Test noise operators with actual simulation setup."""

import sys
sys.path.insert(0, '/Users/scottjones_admin/Library/Mobile Documents/com~apple~CloudDocs/Mac files/Repos/NoisyQuantumSimulator/src')

import numpy as np
from qutip import tensor, mesolve, basis

from qpu_simulator.micro_physics.neutral_atoms.rydberg_gates.noise_models import (
    build_all_noise_operators
)
from qpu_simulator.micro_physics.neutral_atoms.rydberg_gates.hamiltonians import (
    build_hilbert_space
)

# Build 3-level Hilbert space
hs = build_hilbert_space(3)
print(f"Hilbert space dimension: {hs.dim}")

# Get basis states
b0 = hs.basis['0']
b1 = hs.basis['1']
br = hs.basis['r']
I = hs.identity

# Initial state: |11⟩ (this is the state that gets Rydberg blockade)
psi_11 = tensor(b1, b1)
print(f"Initial state |11⟩")

# Also test with Rydberg: |1r⟩
psi_1r = tensor(b1, br)
print(f"Also testing |1r⟩ (has Rydberg population)")

# Build noise operators with realistic rates
gamma_r = 1000  # 1 kHz Rydberg decay
gamma_phi = 1e4  # 10 kHz dephasing
c_ops, noise_dict = build_all_noise_operators(
    hs=hs,
    gamma_r=gamma_r,
    gamma_phi_laser=gamma_phi,
)

print(f"\nNumber of collapse operators: {len(c_ops)}")
print(f"Noise dict: {noise_dict}")

# Check which operators are non-zero on |11⟩
print(f"\nOperator action on |11⟩:")
for i, L in enumerate(c_ops):
    result = L * psi_11
    norm = result.norm()
    print(f"  L[{i}]|11⟩ norm: {norm:.4f}")

# Check which operators are non-zero on |1r⟩
print(f"\nOperator action on |1r⟩:")
for i, L in enumerate(c_ops):
    result = L * psi_1r
    norm = result.norm()
    print(f"  L[{i}]|1r⟩ norm: {norm:.4f}")

# Test time evolution with |1r⟩ (should decay!)
H = tensor(0*I, I)  # Zero Hamiltonian
tlist = np.linspace(0, 1e-3, 100)  # 1 ms

print(f"\n--- Time evolution of |1r⟩ with decay ---")
result = mesolve(H, psi_1r, tlist, c_ops=c_ops)
final = result.states[-1]

print(f"Final state: isket={final.isket}, isoper={final.isoper}")
print(f"Trace: {final.tr():.6f}")
print(f"Purity: {np.real((final*final).tr()):.6f}")

# Population in each state
proj_1r = psi_1r * psi_1r.dag()
pop_1r = np.real((proj_1r * final).tr())
print(f"Population in |1r⟩: {pop_1r:.4f} (expected: exp(-{gamma_r}*1e-3) = {np.exp(-gamma_r*1e-3):.4f})")

# Check dephasing: Rydberg dephasing acts via |r⟩⟨r| projector
# Create superposition |ψ⟩ = (|11⟩ + |1r⟩)/√2
psi_super = (psi_11 + psi_1r).unit()
print(f"\n--- Time evolution of superposition (|11⟩ + |1r⟩)/√2 ---")
result_super = mesolve(H, psi_super, tlist, c_ops=c_ops)
final_super = result_super.states[-1]
print(f"Final purity: {np.real((final_super*final_super).tr()):.6f}")
print(f"(Should be < 1 due to dephasing and decay)")
