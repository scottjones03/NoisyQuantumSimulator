#!/usr/bin/env python3
"""Direct test of mesolve with collapse operators."""

from qutip import basis, tensor, mesolve, qeye
import numpy as np

# Simple test: does mesolve with c_ops reduce purity?
b0 = basis(3, 0)
b1 = basis(3, 1)
br = basis(3, 2)

# Initial state |01⟩ means atom1=0, atom2=1
psi0 = tensor(b0, b1)
print(f"Initial state |01⟩ (atom1=|0⟩, atom2=|1⟩)")

# Simple Hamiltonian - identity for no evolution
I = qeye(3)
H = tensor(0*I, I)  # Zero Hamiltonian

# Decay operator: |1⟩ → |0⟩ with rate gamma on SECOND atom (which is in |1⟩)
gamma = 1e5  # 100 kHz
sigma_10 = b0 * b1.dag()  # |0⟩⟨1|
L = np.sqrt(gamma) * tensor(qeye(3), sigma_10)  # Identity on atom1, decay on atom2

print(f"Collapse operator L = I ⊗ sqrt({gamma}) * |0><1|")
print(f"L shape: {L.shape}")
print(f"L norm: {np.sqrt((L.dag()*L).tr()):.1f}")

# Check that L|01⟩ ≠ 0 - it should take |01⟩ → |00⟩
L_on_psi = L * psi0
print(f"L|01⟩ norm (should be ~{np.sqrt(gamma):.1f}): {L_on_psi.norm():.1f}")
print(f"L|01⟩ = {np.sqrt(gamma)} * |00⟩")

# Time evolution
tlist = np.linspace(0, 1e-5, 100)  # 10 us

print("\nRunning mesolve with collapse operator...")
result = mesolve(H, psi0, tlist, c_ops=[L])
final = result.states[-1]

print(f'Final state: isket={final.isket}, isoper={final.isoper}')
print(f'Trace: {final.tr():.6f}')
print(f'Purity: {np.real((final*final).tr()):.6f}')
print(f'Expected decay: exp(-gamma*t) = {np.exp(-gamma*1e-5):.6f}')

# Population in original state |01⟩
proj = psi0 * psi0.dag()
pop_01 = np.real((proj * final).tr())
pop_00 = np.real((tensor(b0,b0)*tensor(b0,b0).dag() * final).tr())
print(f'Population in |01⟩: {pop_01:.6f} (expected: {np.exp(-gamma*1e-5):.6f})')
print(f'Population in |00⟩: {pop_00:.6f} (expected: {1-np.exp(-gamma*1e-5):.6f})')
