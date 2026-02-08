#!/usr/bin/env python3
"""
JP Protocol Parameter Search - Phase-Focused
=============================================

The basic physics is correct. The issue is that the stored JP parameters
give the wrong controlled phase (-27° instead of ±180°).

This script searches for bang-bang parameters that produce the CORRECT
controlled phase φ_ctrl = φ₁₁ - 2φ₀₁ ≈ ±π
"""

import numpy as np
from qutip import tensor, mesolve, basis
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, 'src')

from qpu_simulator.micro_physics.neutral_atoms.rydberg_gates.hamiltonians import (
    build_hilbert_space,
    build_phase_modulated_hamiltonian,
)

# Build Hilbert space
hs = build_hilbert_space(3)
b0 = hs.basis['0']
b1 = hs.basis['1']
br = hs.basis['r']

psi_01 = tensor(b0, b1)
psi_11 = tensor(b1, b1)


def get_overlap(psi1, psi2):
    """Get complex overlap ⟨ψ1|ψ2⟩."""
    if psi1.type == 'oper':
        evals, evecs = psi1.eigenstates()
        psi1 = evecs[-1]
    if psi2.type == 'oper':
        evals, evecs = psi2.eigenstates()
        psi2 = evecs[-1]
    result = psi1.dag() * psi2
    if hasattr(result, 'tr'):
        return result.tr()
    elif hasattr(result, 'full'):
        return complex(result.full()[0, 0])
    else:
        return complex(result)


def evolve(H, psi0, t_total, n_steps=50):
    """Evolve state under Hamiltonian."""
    tlist = np.linspace(0, t_total, n_steps)
    result = mesolve(H, psi0, tlist, c_ops=[], options={'atol': 1e-10, 'rtol': 1e-8})
    return result.states[-1]


def evaluate_bangbang(switching_times_omega, phases, Omega, V, omega_tau_total):
    """
    Evaluate a bang-bang configuration.
    
    Parameters
    ----------
    switching_times_omega : list
        Switching times in units of Ωt (dimensionless)
    phases : list
        Phase for each segment (radians)
    Omega, V : float
        Rabi and blockade frequencies
    omega_tau_total : float
        Total gate time in units of Ωt
        
    Returns
    -------
    dict with overlap_01, overlap_11, phase_01, phase_11, ctrl_phase, fidelity
    """
    tau_total = omega_tau_total / Omega
    
    # Build boundaries
    boundaries_omega = [0.0] + list(switching_times_omega) + [omega_tau_total]
    boundaries = [t / Omega for t in boundaries_omega]
    
    n_segments = len(phases)
    
    # Evolve
    psi_01_f = psi_01
    psi_11_f = psi_11
    
    for i in range(n_segments):
        dt = boundaries[i+1] - boundaries[i]
        if dt <= 0:
            continue
        H = build_phase_modulated_hamiltonian(
            Omega=Omega, phase=phases[i], V=V, hs=hs, Delta=0
        )
        psi_01_f = evolve(H, psi_01_f, dt, n_steps=max(10, int(30 * dt / tau_total)))
        psi_11_f = evolve(H, psi_11_f, dt, n_steps=max(10, int(30 * dt / tau_total)))
    
    ov_01 = get_overlap(psi_01, psi_01_f)
    ov_11 = get_overlap(psi_11, psi_11_f)
    
    overlap_01 = abs(ov_01)
    overlap_11 = abs(ov_11)
    phase_01 = np.angle(ov_01)
    phase_11 = np.angle(ov_11)
    
    ctrl_phase = phase_11 - 2*phase_01
    ctrl_phase_wrapped = np.arctan2(np.sin(ctrl_phase), np.cos(ctrl_phase))
    
    # Fidelity: want overlaps ~1 and ctrl_phase ~±π
    phase_error = min(abs(ctrl_phase_wrapped - np.pi), abs(ctrl_phase_wrapped + np.pi))
    fidelity = overlap_01**2 * overlap_11**2 * np.cos(phase_error/2)**2
    
    return {
        'overlap_01': overlap_01,
        'overlap_11': overlap_11,
        'phase_01': phase_01,
        'phase_11': phase_11,
        'ctrl_phase': ctrl_phase_wrapped,
        'ctrl_phase_deg': np.degrees(ctrl_phase_wrapped),
        'fidelity': fidelity,
    }


def search_7segment_jp():
    """
    Search for 7-segment JP parameters (like the stored ones).
    
    Structure: [φ₀, φ₁, φ₂, φ₃, φ₄, φ₅, φ₆] with 6 switching times
    Phases from {-π/2, 0, +π/2}
    """
    print("="*70)
    print("SEARCH: 7-SEGMENT BANG-BANG (LIKE STORED JP)")
    print("="*70)
    
    Omega = 2 * np.pi * 5e6
    V = 200 * Omega  # Match stored parameters
    
    phase_opts = [-np.pi/2, 0, np.pi/2]
    
    best_fidelity = 0
    best_config = None
    
    # Search over total time
    for omega_tau in np.linspace(6.0, 8.0, 20):
        # Generate candidate switching times
        # The structure seems to be: short, short, LONG, short, short, LONG, short
        # Let's parameterize this way
        for t1 in np.linspace(0.2, 0.5, 5):
            for t2 in np.linspace(0.4, 0.8, 5):
                if t2 <= t1:
                    continue
                for t3 in np.linspace(2.5, 4.0, 5):
                    if t3 <= t2:
                        continue
                    for t4 in np.linspace(t3+0.05, t3+0.3, 3):
                        for t5 in np.linspace(t4+0.3, t4+1.0, 4):
                            if t5 >= omega_tau:
                                continue
                            for t6 in np.linspace(t5+1.5, omega_tau-0.1, 4):
                                if t6 >= omega_tau:
                                    continue
                                
                                switching = [t1, t2, t3, t4, t5, t6]
                                
                                # Try different phase patterns
                                # The stored one is [π/2, 0, -π/2, -π/2, 0, π/2, 0]
                                # Try some variations
                                for pattern in [
                                    [np.pi/2, 0, -np.pi/2, -np.pi/2, 0, np.pi/2, 0],  # Stored
                                    [-np.pi/2, 0, np.pi/2, np.pi/2, 0, -np.pi/2, 0],  # Negated
                                    [np.pi/2, -np.pi/2, 0, 0, -np.pi/2, np.pi/2, 0],
                                    [0, np.pi/2, -np.pi/2, -np.pi/2, np.pi/2, 0, np.pi/2],
                                ]:
                                    result = evaluate_bangbang(switching, pattern, Omega, V, omega_tau)
                                    
                                    if result['fidelity'] > best_fidelity:
                                        best_fidelity = result['fidelity']
                                        best_config = {
                                            'omega_tau': omega_tau,
                                            'switching_times': switching,
                                            'phases': pattern,
                                            **result
                                        }
    
    print(f"\nBest 7-segment result:")
    if best_config:
        print(f"  Ωτ = {best_config['omega_tau']:.4f}")
        print(f"  Switching (Ωt): {[f'{t:.4f}' for t in best_config['switching_times']]}")
        print(f"  Phases: {[f'{p/np.pi:.2f}π' for p in best_config['phases']]}")
        print(f"  |01⟩ overlap = {best_config['overlap_01']:.4f}")
        print(f"  |11⟩ overlap = {best_config['overlap_11']:.4f}")
        print(f"  Controlled phase = {best_config['ctrl_phase_deg']:.1f}°")
        print(f"  Fidelity = {best_config['fidelity']:.4f}")
    
    return best_config


def optimize_continuous():
    """
    Use scipy optimization to find optimal switching times.
    Fix phases to the stored pattern and optimize switching times.
    """
    print("\n" + "="*70)
    print("OPTIMIZATION: CONTINUOUS SWITCHING TIMES")
    print("="*70)
    
    Omega = 2 * np.pi * 5e6
    V = 200 * Omega
    
    phases_fixed = [np.pi/2, 0, -np.pi/2, -np.pi/2, 0, np.pi/2, 0]
    
    def objective(x):
        """Objective: minimize negative fidelity."""
        omega_tau = x[0]
        switching = sorted(x[1:7])  # Ensure sorted order
        
        # Check validity
        if switching[0] < 0.1 or switching[-1] > omega_tau - 0.1:
            return 1.0
        for i in range(len(switching)-1):
            if switching[i+1] - switching[i] < 0.05:
                return 1.0
        
        result = evaluate_bangbang(switching, phases_fixed, Omega, V, omega_tau)
        return -result['fidelity']
    
    # Initial guess from stored parameters
    x0 = [7.0, 0.3328, 0.5859, 3.434, 3.553, 4.1204, 6.7431]
    
    # Bounds
    bounds = [
        (5.0, 10.0),  # omega_tau
        (0.1, 1.5),   # t1
        (0.2, 2.0),   # t2
        (1.5, 5.0),   # t3
        (1.6, 5.5),   # t4
        (2.5, 7.0),   # t5
        (4.0, 9.0),   # t6
    ]
    
    print("Running differential evolution optimization...")
    result = differential_evolution(objective, bounds, maxiter=200, seed=42,
                                   workers=1, updating='deferred', polish=True)
    
    omega_tau_opt = result.x[0]
    switching_opt = sorted(result.x[1:7])
    
    eval_result = evaluate_bangbang(switching_opt, phases_fixed, Omega, V, omega_tau_opt)
    
    print(f"\nOptimized result:")
    print(f"  Ωτ = {omega_tau_opt:.4f}")
    print(f"  Switching (Ωt): {[f'{t:.4f}' for t in switching_opt]}")
    print(f"  Phases: {[f'{p/np.pi:.2f}π' for p in phases_fixed]}")
    print(f"  |01⟩ overlap = {eval_result['overlap_01']:.4f}")
    print(f"  |11⟩ overlap = {eval_result['overlap_11']:.4f}")
    print(f"  Controlled phase = {eval_result['ctrl_phase_deg']:.1f}°")
    print(f"  Fidelity = {eval_result['fidelity']:.4f}")
    
    return {
        'omega_tau': omega_tau_opt,
        'switching_times': switching_opt,
        'phases': phases_fixed,
        **eval_result
    }


def optimize_all_phases():
    """
    Optimize both switching times AND phase choices using differential evolution.
    """
    print("\n" + "="*70)
    print("OPTIMIZATION: SWITCHING TIMES + PHASES")
    print("="*70)
    
    Omega = 2 * np.pi * 5e6
    V = 200 * Omega
    
    phase_options = np.array([-np.pi/2, 0, np.pi/2])
    
    def objective(x):
        """Objective: minimize negative fidelity."""
        omega_tau = x[0]
        switching = sorted(x[1:7])
        phase_indices = np.round(x[7:14]).astype(int) % 3
        phases = [phase_options[i] for i in phase_indices]
        
        # Check validity
        if switching[0] < 0.1 or switching[-1] > omega_tau - 0.1:
            return 1.0
        for i in range(len(switching)-1):
            if switching[i+1] - switching[i] < 0.03:
                return 1.0
        
        result = evaluate_bangbang(switching, phases, Omega, V, omega_tau)
        return -result['fidelity']
    
    # Bounds: switching times + phase indices
    bounds = [
        (5.0, 12.0),  # omega_tau
        (0.05, 2.0),  # t1
        (0.1, 3.0),   # t2
        (0.5, 6.0),   # t3
        (0.6, 7.0),   # t4
        (1.0, 9.0),   # t5
        (2.0, 11.0),  # t6
        (0, 2.99), (0, 2.99), (0, 2.99), (0, 2.99),  # phase indices
        (0, 2.99), (0, 2.99), (0, 2.99),
    ]
    
    print("Running differential evolution (this may take a minute)...")
    result = differential_evolution(objective, bounds, maxiter=500, seed=42,
                                   workers=1, updating='deferred', polish=True,
                                   popsize=20)
    
    omega_tau_opt = result.x[0]
    switching_opt = sorted(result.x[1:7])
    phase_indices = np.round(result.x[7:14]).astype(int) % 3
    phases_opt = [phase_options[i] for i in phase_indices]
    
    eval_result = evaluate_bangbang(switching_opt, phases_opt, Omega, V, omega_tau_opt)
    
    print(f"\nFully optimized result:")
    print(f"  Ωτ = {omega_tau_opt:.4f}")
    print(f"  Switching (Ωt): {[f'{t:.4f}' for t in switching_opt]}")
    print(f"  Phases: {[f'{p/np.pi:.2f}π' for p in phases_opt]}")
    print(f"  |01⟩ overlap = {eval_result['overlap_01']:.4f}")
    print(f"  |11⟩ overlap = {eval_result['overlap_11']:.4f}")
    print(f"  Phase 01 = {np.degrees(eval_result['phase_01']):.1f}°")
    print(f"  Phase 11 = {np.degrees(eval_result['phase_11']):.1f}°")
    print(f"  Controlled phase = {eval_result['ctrl_phase_deg']:.1f}°")
    print(f"  Fidelity = {eval_result['fidelity']:.4f}")
    
    return {
        'omega_tau': omega_tau_opt,
        'switching_times': switching_opt,
        'phases': phases_opt,
        **eval_result
    }


def test_at_different_V_ratios(config):
    """Test optimized configuration at different V/Ω ratios."""
    print("\n" + "="*70)
    print("TEST: OPTIMIZED PARAMS AT DIFFERENT V/Ω")
    print("="*70)
    
    Omega = 2 * np.pi * 5e6
    
    print(f"\n{'V/Ω':>10} {'|01⟩':>10} {'|11⟩':>10} {'φ_ctrl':>12} {'Fidelity':>10}")
    print("-"*60)
    
    for V_ratio in [10, 25, 50, 100, 200, 500]:
        V = V_ratio * Omega
        result = evaluate_bangbang(
            config['switching_times'], 
            config['phases'], 
            Omega, V, 
            config['omega_tau']
        )
        print(f"{V_ratio:>10} {result['overlap_01']:>10.4f} {result['overlap_11']:>10.4f} "
              f"{result['ctrl_phase_deg']:>+12.1f}° {result['fidelity']:>10.4f}")


if __name__ == "__main__":
    print("="*70)
    print("JP PROTOCOL PARAMETER OPTIMIZATION")
    print("="*70)
    print("\nSearching for bang-bang parameters that produce correct CZ phase...\n")
    
    # Method 1: Grid search
    config_grid = search_7segment_jp()
    
    # Method 2: Continuous optimization of switching times
    config_cont = optimize_continuous()
    
    # Method 3: Full optimization including phases
    config_full = optimize_all_phases()
    
    # Test best result at different V/Ω
    best = config_full if config_full['fidelity'] > config_cont['fidelity'] else config_cont
    test_at_different_V_ratios(best)
    
    print("\n" + "="*70)
    print("FINAL RESULT")
    print("="*70)
    if best['fidelity'] > 0.99:
        print(f"✓ Found working JP parameters with fidelity {best['fidelity']:.4f}!")
        print(f"\nUpdate protocols.py with:")
        print(f"  omega_tau = {best['omega_tau']:.4f}")
        print(f"  switching_times = {[round(t, 4) for t in best['switching_times']]}")
        print(f"  phases = {[round(p/np.pi, 2) for p in best['phases']]}  # in units of π")
    else:
        print(f"✗ Best fidelity achieved: {best['fidelity']:.4f}")
        print(f"  Controlled phase: {best['ctrl_phase_deg']:.1f}° (need ±180°)")
        print(f"\nThe 7-segment bang-bang may not be sufficient for CZ.")
        print(f"Consider: more segments, or LP-style detuning + phase control.")
