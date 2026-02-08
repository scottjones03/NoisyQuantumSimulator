#!/usr/bin/env python3
"""
Quick check: Does fidelity correctly penalize |11⟩ → +|11⟩?
"""
import numpy as np
import qutip as qt

ket_1 = qt.basis(3, 1)
state_11 = qt.tensor(ket_1, ket_1)
target_minus = -state_11

# If final state is +|11⟩ (no phase flip)
overlap_plus = np.abs(complex(target_minus.dag() * state_11))**2
print(f"If final = +|11⟩: |⟨-11|+11⟩|² = {overlap_plus:.6f}")

# If final state is -|11⟩ (correct phase flip)
overlap_minus = np.abs(complex(target_minus.dag() * (-state_11)))**2
print(f"If final = -|11⟩: |⟨-11|-11⟩|² = {overlap_minus:.6f}")

# The issue is: -1 and +1 are just phases
# ⟨-ψ|+ψ⟩ = -⟨ψ|ψ⟩ = -1
# |⟨-ψ|+ψ⟩|² = |-1|² = 1
print("\nSo |⟨-11|+11⟩|² = |(-1)|² = 1, not 0!")
print("The fidelity metric can't distinguish +|11⟩ from -|11⟩!")
