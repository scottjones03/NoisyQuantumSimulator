#!/usr/bin/env python3
"""7-segment search to get >99% fidelity"""

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

def eval_segments(switching_omega, phases, omega_tau):
    """Evaluate arbitrary segment config."""
    tau = omega_tau / Omega
    boundaries = [0] + [t/Omega for t in switching_omega] + [tau]
    
    psi_01_f, psi_11_f = psi_01, psi_11
    for i, phase in enumerate(phases):
        dt = boundaries[i+1] - boundaries[i]
        if dt <= 0: continue
        H = build_phase_modulated_hamiltonian(Omega=Omega, phase=phase, V=V, hs=hs, Delta=0)
        tlist = np.linspace(0, dt, max(15, int(40*dt/tau)))
        psi_01_f = mesolve(H, psi_01_f, tlist, c_ops=[]).states[-1]
        psi_11_f = mesolve(H, psi_11_f, tlist, c_ops=[]).states[-1]
    
    ov01, ov11 = get_overlap(psi_01, psi_01_f), get_overlap(psi_11, psi_11_f)
    ctrl = np.angle(ov11) - 2*np.angle(ov01)
    ctrl = np.arctan2(np.sin(ctrl), np.cos(ctrl))
    phase_err = min(abs(ctrl - np.pi), abs(ctrl + np.pi))
    fid = abs(ov01)**2 * abs(ov11)**2 * np.cos(phase_err/2)**2
    return abs(ov01), abs(ov11), np.degrees(ctrl), fid

print("="*60)
print("7-SEGMENT SEARCH")
print("="*60)

# 7 segments = 6 switching times
# Try symmetric pattern like JP paper: phases = [π/2, 0, -π/2, -π/2, 0, π/2, 0]
phases_jp = [np.pi/2, 0, -np.pi/2, -np.pi/2, 0, np.pi/2, 0]

# The structure: short-short-LONG-short-short-LONG-short
# Parameterize as fractions of total time

def eval_7seg_symmetric(omega_tau, f1, f2, f3):
    """7-segment with symmetric structure."""
    # Pattern: [0, t1, t2, t3, tau-t3, tau-t2, tau-t1, tau]
    # Durations: f1, f2-f1, f3-f2, (tau-2*t3), f3-f2, f2-f1, f1 (symmetric)
    t1 = f1 * omega_tau
    t2 = f2 * omega_tau
    t3 = f3 * omega_tau
    t4 = omega_tau - t3
    t5 = omega_tau - t2
    t6 = omega_tau - t1
    switching = [t1, t2, t3, t4, t5, t6]
    return switching, eval_segments(switching, phases_jp, omega_tau)

best = (0, None)
print("\nSearching 7-segment symmetric...")

for omega_tau in np.linspace(6, 10, 20):
    for f1 in np.linspace(0.02, 0.12, 10):
        for f2 in np.linspace(f1+0.02, 0.25, 10):
            for f3 in np.linspace(f2+0.05, 0.49, 10):
                sw, (ov01, ov11, ctrl, fid) = eval_7seg_symmetric(omega_tau, f1, f2, f3)
                if fid > best[0]:
                    best = (fid, {'omega_tau': omega_tau, 'f1': f1, 'f2': f2, 'f3': f3,
                                 'sw': sw, 'ov01': ov01, 'ov11': ov11, 'ctrl': ctrl})

b = best[1]
print(f"\nBest 7-seg symmetric: Fid={best[0]:.4f}")
print(f"  Ωτ={b['omega_tau']:.4f}")
print(f"  Switching: {[f'{s:.4f}' for s in b['sw']]}")
print(f"  Overlaps: |01⟩={b['ov01']:.4f}, |11⟩={b['ov11']:.4f}, ctrl={b['ctrl']:.1f}°")

# Optimize
print("\nOptimizing...")
def objective(x):
    omega_tau, f1, f2, f3 = x
    if f1 < 0.01 or f2 <= f1+0.01 or f3 <= f2+0.02 or f3 > 0.49:
        return 1.0
    _, (_, _, _, fid) = eval_7seg_symmetric(omega_tau, f1, f2, f3)
    return -fid

x0 = [b['omega_tau'], b['f1'], b['f2'], b['f3']]
result = minimize(objective, x0, method='Nelder-Mead',
                  options={'xatol': 0.0001, 'fatol': 1e-6, 'maxiter': 300})

omega_tau, f1, f2, f3 = result.x
sw, (ov01, ov11, ctrl, fid) = eval_7seg_symmetric(omega_tau, f1, f2, f3)

print(f"\nOptimized 7-segment: Fid={fid:.4f}")
print(f"  Ωτ = {omega_tau:.4f}")
print(f"  Switching (Ωt): {[f'{s:.4f}' for s in sw]}")
print(f"  Overlaps: |01⟩={ov01:.4f}, |11⟩={ov11:.4f}")
print(f"  Controlled phase = {ctrl:.1f}°")
print(f"  Phases: [π/2, 0, -π/2, -π/2, 0, π/2, 0]")

# Also try the 5-segment optimized solution at higher resolution
print("\n" + "="*60)
print("COMPARE: 5-segment vs 7-segment")
print("="*60)

# 5-segment from before
phases_5 = [np.pi/2, 0, -np.pi/2, 0, np.pi/2]
omega_tau_5 = 22.08
sw_5 = [2.214, 8.823, 13.258, 19.867]
ov01_5, ov11_5, ctrl_5, fid_5 = eval_segments(sw_5, phases_5, omega_tau_5)
print(f"5-segment: Fid={fid_5:.4f}, |01⟩={ov01_5:.4f}, |11⟩={ov11_5:.4f}")

print(f"7-segment: Fid={fid:.4f}, |01⟩={ov01:.4f}, |11⟩={ov11:.4f}")

print("\n" + "="*60)
if fid > 0.99 or fid_5 > 0.99:
    winner = "7-seg" if fid > fid_5 else "5-seg"
    print(f"✓ SUCCESS! Use {winner} parameters")
else:
    print(f"Best achieved: {max(fid, fid_5):.4f}")
    print("May need more segments or continuous phase modulation")
