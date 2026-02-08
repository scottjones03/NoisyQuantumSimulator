#!/usr/bin/env python3
"""Refine the 5-segment solution that got 92% fidelity"""

import numpy as np
from qutip import tensor, mesolve, basis
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, 'src')

from qpu_simulator.micro_physics.neutral_atoms.rydberg_gates.hamiltonians import (
    build_hilbert_space, build_phase_modulated_hamiltonian,
)

hs = build_hilbert_space(3)
b0, b1, br = hs.basis['0'], hs.basis['1'], hs.basis['r']
psi_01, psi_11 = tensor(b0, b1), tensor(b1, b1)
Omega = 2 * np.pi * 5e6
V = 200 * Omega

def get_overlap(psi1, psi2):
    result = psi1.dag() * psi2
    if hasattr(result, 'tr'): return result.tr()
    elif hasattr(result, 'full'): return complex(result.full()[0, 0])
    return complex(result)

def eval_5seg(omega_tau, t1_frac, t2_frac, phases):
    """Evaluate 5-segment symmetric config."""
    t1 = t1_frac * omega_tau
    t2 = t2_frac * omega_tau
    t3 = omega_tau - t2
    t4 = omega_tau - t1
    switching = [t1, t2, t3, t4]
    
    tau = omega_tau / Omega
    boundaries = [0] + [t/Omega for t in switching] + [tau]
    
    psi_01_f, psi_11_f = psi_01, psi_11
    for i, phase in enumerate(phases):
        dt = boundaries[i+1] - boundaries[i]
        if dt <= 0: continue
        H = build_phase_modulated_hamiltonian(Omega=Omega, phase=phase, V=V, hs=hs, Delta=0)
        tlist = np.linspace(0, dt, 30)
        psi_01_f = mesolve(H, psi_01_f, tlist, c_ops=[]).states[-1]
        psi_11_f = mesolve(H, psi_11_f, tlist, c_ops=[]).states[-1]
    
    ov01, ov11 = get_overlap(psi_01, psi_01_f), get_overlap(psi_11, psi_11_f)
    ctrl = np.angle(ov11) - 2*np.angle(ov01)
    ctrl = np.arctan2(np.sin(ctrl), np.cos(ctrl))
    phase_err = min(abs(ctrl - np.pi), abs(ctrl + np.pi))
    fid = abs(ov01)**2 * abs(ov11)**2 * np.cos(phase_err/2)**2
    return abs(ov01), abs(ov11), np.degrees(ctrl), fid, switching

# Starting point from previous search
print("="*60)
print("REFINING 5-SEGMENT SOLUTION")
print("="*60)

phases = [np.pi/2, 0, -np.pi/2, 0, np.pi/2]

# Fine grid search around best
best = (0, None)
print("\nFine grid search...")
for omega_tau in np.linspace(18, 26, 40):
    for t1_frac in np.linspace(0.02, 0.12, 20):
        for t2_frac in np.linspace(0.20, 0.45, 25):
            if t2_frac <= t1_frac + 0.05: continue
            ov01, ov11, ctrl, fid, sw = eval_5seg(omega_tau, t1_frac, t2_frac, phases)
            if fid > best[0]:
                best = (fid, {'omega_tau': omega_tau, 't1_frac': t1_frac, 't2_frac': t2_frac,
                             'switch': sw, 'ov01': ov01, 'ov11': ov11, 'ctrl': ctrl})

b = best[1]
print(f"\nBest from grid: Fid={best[0]:.4f}")
print(f"  Ωτ={b['omega_tau']:.4f}, t1_frac={b['t1_frac']:.4f}, t2_frac={b['t2_frac']:.4f}")
print(f"  Overlaps: |01⟩={b['ov01']:.4f}, |11⟩={b['ov11']:.4f}, ctrl={b['ctrl']:.1f}°")

# Now optimize with scipy
print("\nOptimizing with scipy...")

def objective(x):
    omega_tau, t1_frac, t2_frac = x
    if t2_frac <= t1_frac + 0.02 or t1_frac < 0.01 or t2_frac > 0.49:
        return 1.0
    ov01, ov11, ctrl, fid, _ = eval_5seg(omega_tau, t1_frac, t2_frac, phases)
    return -fid

x0 = [b['omega_tau'], b['t1_frac'], b['t2_frac']]
result = minimize(objective, x0, method='Nelder-Mead', 
                  options={'xatol': 0.001, 'fatol': 1e-5, 'maxiter': 200})

omega_tau_opt, t1_opt, t2_opt = result.x
ov01, ov11, ctrl, fid, sw = eval_5seg(omega_tau_opt, t1_opt, t2_opt, phases)

print(f"\nOptimized: Fid={fid:.4f}")
print(f"  Ωτ = {omega_tau_opt:.4f}")
print(f"  t1_frac = {t1_opt:.4f}, t2_frac = {t2_opt:.4f}")
print(f"  Switching (Ωt): {[f'{s:.4f}' for s in sw]}")
print(f"  Overlaps: |01⟩={ov01:.4f}, |11⟩={ov11:.4f}")
print(f"  Controlled phase = {ctrl:.1f}°")
print(f"  Phases: {[f'{p/np.pi:.2f}π' for p in phases]}")

# Try other phase patterns
print("\n--- Trying other phase patterns ---")
patterns = [
    [np.pi/2, 0, -np.pi/2, 0, np.pi/2],
    [-np.pi/2, 0, np.pi/2, 0, -np.pi/2],
    [0, np.pi/2, 0, np.pi/2, 0],
    [0, -np.pi/2, 0, -np.pi/2, 0],
    [np.pi/2, -np.pi/2, 0, -np.pi/2, np.pi/2],
    [-np.pi/2, np.pi/2, 0, np.pi/2, -np.pi/2],
]

for p in patterns:
    ov01, ov11, ctrl, fid, _ = eval_5seg(omega_tau_opt, t1_opt, t2_opt, p)
    print(f"  {[f'{x/np.pi:.1f}π' for x in p]}: Fid={fid:.4f}, ctrl={ctrl:+.1f}°")

print("\n" + "="*60)
if fid > 0.99:
    print("✓ FOUND WORKING PARAMETERS!")
    print(f"\nTo use in protocols.py:")
    print(f"  omega_tau = {omega_tau_opt:.4f}")
    print(f"  switching_times = {[round(s, 4) for s in sw]}")
    print(f"  phases = [π/2, 0, -π/2, 0, π/2]")
else:
    print(f"Best fidelity: {fid:.4f} - need more segments or different approach")
