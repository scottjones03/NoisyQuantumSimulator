#!/usr/bin/env python3
"""Test phase extraction with density matrix."""

import numpy as np
from qutip import Qobj, tensor, basis, ket2dm

# 3-level single atom
b0 = basis(3, 0)
b1 = basis(3, 1)
br = basis(3, 2)

# Two-atom state
psi_01 = tensor(b0, b1)  # Target
print(f"psi_01 shape: {psi_01.shape}")
print(f"psi_01 is ket: {psi_01.isket}")

# Simulated output as ket (unitary case)
output_ket = np.exp(1j * 0.5) * psi_01  # phase = 0.5 rad
print(f"\nKet output:")
print(f"  shape: {output_ket.shape}")
overlap_ket = psi_01.dag() * output_ket
print(f"  overlap type: {type(overlap_ket)}")
if hasattr(overlap_ket, 'full'):
    val = overlap_ket.full()
    print(f"  overlap.full() shape: {val.shape}")
    print(f"  overlap value: {val[0,0]}")
    print(f"  extracted phase: {np.angle(val[0,0]):.4f} rad (expected 0.5)")

# Simulated output as density matrix (mesolve case)
output_dm = ket2dm(output_ket)
print(f"\nDensity matrix output:")
print(f"  shape: {output_dm.shape}")
overlap_dm = psi_01.dag() * output_dm
print(f"  overlap type: {type(overlap_dm)}")
if hasattr(overlap_dm, 'full'):
    val = overlap_dm.full()
    print(f"  overlap.full() shape: {val.shape}")
    print(f"  This is NOT a scalar! It's a row vector!")
    
# Correct way to get expectation value from density matrix
print(f"\nCorrect approach for density matrix:")
proj_01 = psi_01 * psi_01.dag()  # |01⟩⟨01|
expectation = (proj_01 * output_dm).tr()
print(f"  <01|ρ|01> = {expectation:.4f}")
print(f"  This gives probability, NOT phase!")

# To get phase from density matrix we need off-diagonal element
# But for pure state we can use the fact that ρ = |ψ⟩⟨ψ|
print(f"\nPhase from density matrix sqrt:")
from scipy.linalg import sqrtm
dm_array = output_dm.full()
sqrt_dm = sqrtm(dm_array)
# The sqrt of a pure state density matrix is proportional to the ket
# But with normalization issues...
print(f"  sqrt(ρ) diagonal: {np.diag(sqrt_dm)[:5]}")
