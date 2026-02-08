#!/usr/bin/env python3
"""Test to check if mesolve returns ket or density matrix with collapse ops."""

import numpy as np
from qutip import Qobj, tensor, basis, mesolve, ket2dm

# Simple 2-level system
b0 = basis(2, 0)
b1 = basis(2, 1)
psi0 = b0

# Simple Hamiltonian
H = np.pi * (b0 * b1.dag() + b1 * b0.dag())  # X rotation

# Collapse operator (decay)
c_op = np.sqrt(0.1) * b0 * b1.dag()

# Without collapse
tlist = np.linspace(0, 1, 10)
result_no_c = mesolve(H, psi0, tlist, c_ops=[])
print("Without collapse ops:")
print(f"  Final state type: {type(result_no_c.states[-1])}")
print(f"  Is ket: {result_no_c.states[-1].isket}")

# With collapse
result_with_c = mesolve(H, psi0, tlist, c_ops=[c_op])
print("\nWith collapse ops:")
print(f"  Final state type: {type(result_with_c.states[-1])}")
print(f"  Is ket: {result_with_c.states[-1].isket if hasattr(result_with_c.states[-1], 'isket') else 'N/A'}")
print(f"  Is oper (density matrix): {result_with_c.states[-1].isoper if hasattr(result_with_c.states[-1], 'isoper') else 'N/A'}")
print(f"  Shape: {result_with_c.states[-1].shape}")

# Trace should be 1 for density matrix
if result_with_c.states[-1].isoper:
    print(f"  Trace: {result_with_c.states[-1].tr()}")
