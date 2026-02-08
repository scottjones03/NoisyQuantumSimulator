#!/usr/bin/env python3
"""
Fast JP parameter search - focus on controlled phase = ±180°
"""

import numpy as np
from qutip import tensor, mesolve, basis
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, 'src')

from qpu_simulator.micro_physics.neutral_atoms.rydberg_gates.hamiltonians import (
    build_hilbert_space,
    build_phase_modulated_hamiltonian,
)

hs = build_hilbert_space(3)
b0, b1, br = hs.basis['0'], hs.basis['1'], hs.basis['r']
psi_01 = tensor(b0, b1)
psi_11 = tensor(b1, b1)

Omega = 2 * np.pi * 5e6
V = 200 * Omega


def evolve_fast(H, psi0, t_total):
    """Fast evolution with minimal steps."""
    tlist = np.linspace(0, t_total, 20)
    result = mesolve(H, psi0, tlist, c_ops=[])
    return result.states[-1]


def get_overlap(psi1, psi2):
    result = psi1.dag() * psi2
    if hasattr(result, 'tr'):
        return result.tr()
    elif hasattr(result, 'full'):
        return complex(result.full()[0, 0])
    else:
        return complex(result)


def eval_config(switching_omega, phases, omega_tau):
    """Evaluate bang-bang config. Returns (overlap_01, overlap_11, ctrl_phase_deg, fidelity)."""
    tau = omega_tau / Omega
    boundaries = [0] + [t/Omega for t in switching_omega] + [tau]
    
    psi_01_f, psi_11_f = psi_01, psi_11
    
    for i, phase in enumerate(phases):
        dt = boundaries[i+1] - boundaries[i]
        if dt <= 0:
            continue
        H = build_phase_modulated_hamiltonian(Omega=Omega, phase=phase, V=V, hs=hs, Delta=0)
        psi_01_f = evolve_fast(H, psi_01_f, dt)
        psi_11_f = evolve_fast(H, psi_11_f, dt)
    
    ov01, ov11 = get_overlap(psi_01, psi_01_f), get_overlap(psi_11, psi_11_f)
    ctrl = np.angle(ov11) - 2*np.angle(ov01)
    ctrl = np.arctan2(np.sin(ctrl), np.cos(ctrl))
    
    phase_err = min(abs(ctrl - np.pi), abs(ctrl + np.pi))
    fid = abs(ov01)**2 * abs(ov11)**2 * np.cos(phase_err/2)**2
    
    return abs(ov01), abs(ov11), np.degrees(ctrl), fid


print("="*60)
print("FAST JP PARAMETER SEARCH")
print("="*60)

# Test 1: Simple 3-segment symmetric search
print("\n--- 3-SEGMENT SYMMETRIC ---")
best = (0, None)
for omega_tau in np.linspace(2*np.pi, 4*np.pi, 30):
    for t1_frac in np.linspace(0.05, 0.45, 15):
        t1 = t1_frac * omega_tau
        t2 = omega_tau - t1
        for p in [[np.pi/2, 0, np.pi/2], [-np.pi/2, 0, -np.pi/2], 
                  [np.pi/2, -np.pi/2, np.pi/2], [0, np.pi/2, 0]]:
            ov01, ov11, ctrl, fid = eval_config([t1, t2], p, omega_tau)
            if fid > best[0]:
                best = (fid, {'omega_tau': omega_tau, 'switch': [t1, t2], 'phases': p, 
                             'ov01': ov01, 'ov11': ov11, 'ctrl': ctrl})

if best[1]:
    b = best[1]
    print(f"Best: Fid={best[0]:.4f}, Ωτ={b['omega_tau']:.3f}, ctrl={b['ctrl']:.1f}°")


# Test 2: 5-segment symmetric 
print("\n--- 5-SEGMENT SYMMETRIC ---")
best = (0, None)
for omega_tau in np.linspace(4*np.pi, 8*np.pi, 20):
    for t1_frac in np.linspace(0.02, 0.15, 8):
        for t2_frac in np.linspace(0.15, 0.45, 8):
            if t2_frac <= t1_frac:
                continue
            t1 = t1_frac * omega_tau
            t2 = t2_frac * omega_tau
            t3 = omega_tau - t2
            t4 = omega_tau - t1
            for p in [[np.pi/2, 0, -np.pi/2, 0, np.pi/2], 
                      [-np.pi/2, 0, np.pi/2, 0, -np.pi/2],
                      [0, np.pi/2, 0, np.pi/2, 0],
                      [np.pi/2, -np.pi/2, 0, -np.pi/2, np.pi/2]]:
                ov01, ov11, ctrl, fid = eval_config([t1, t2, t3, t4], p, omega_tau)
                if fid > best[0]:
                    best = (fid, {'omega_tau': omega_tau, 'switch': [t1, t2, t3, t4], 
                                 'phases': p, 'ov01': ov01, 'ov11': ov11, 'ctrl': ctrl})

if best[1]:
    b = best[1]
    print(f"Best: Fid={best[0]:.4f}, Ωτ={b['omega_tau']:.3f}, ctrl={b['ctrl']:.1f}°")
    print(f"  Overlaps: |01⟩={b['ov01']:.4f}, |11⟩={b['ov11']:.4f}")


# Test 3: Check stored JP params with CORRECT phase definition  
print("\n--- STORED JP PARAMS (RE-CHECK) ---")
switching = [0.3328, 0.5859, 3.434, 3.553, 4.1204, 6.7431]
phases = [np.pi/2, 0, -np.pi/2, -np.pi/2, 0, np.pi/2, 0]
omega_tau = 7.0
ov01, ov11, ctrl, fid = eval_config(switching, phases, omega_tau)
print(f"Fidelity={fid:.4f}, ctrl_phase={ctrl:.1f}°")
print(f"Overlaps: |01⟩={ov01:.4f}, |11⟩={ov11:.4f}")


# Test 4: What if stored params are for DIFFERENT V/Omega?
print("\n--- STORED JP AT DIFFERENT V/Ω ---")
for V_ratio in [10, 20, 50, 100, 200, 500, 1000]:
    V_test = V_ratio * Omega
    # Temporarily override V
    def eval_V(sw, ph, ot, V_val):
        tau = ot / Omega
        boundaries = [0] + [t/Omega for t in sw] + [tau]
        psi_01_f, psi_11_f = psi_01, psi_11
        for i, phase in enumerate(ph):
            dt = boundaries[i+1] - boundaries[i]
            if dt <= 0: continue
            H = build_phase_modulated_hamiltonian(Omega=Omega, phase=phase, V=V_val, hs=hs, Delta=0)
            psi_01_f = evolve_fast(H, psi_01_f, dt)
            psi_11_f = evolve_fast(H, psi_11_f, dt)
        ov01, ov11 = get_overlap(psi_01, psi_01_f), get_overlap(psi_11, psi_11_f)
        ctrl = np.angle(ov11) - 2*np.angle(ov01)
        ctrl = np.arctan2(np.sin(ctrl), np.cos(ctrl))
        phase_err = min(abs(ctrl - np.pi), abs(ctrl + np.pi))
        fid = abs(ov01)**2 * abs(ov11)**2 * np.cos(phase_err/2)**2
        return fid, np.degrees(ctrl)
    
    fid, ctrl = eval_V(switching, phases, omega_tau, V_test)
    marker = "✓" if fid > 0.99 else ""
    print(f"  V/Ω={V_ratio:4d}: Fid={fid:.4f}, ctrl={ctrl:+7.1f}° {marker}")


print("\n--- CONCLUSION ---")
print("If no V/Ω gives high fidelity, the stored parameters may be wrong")
print("or there's a sign/convention mismatch in how phases are applied.")
