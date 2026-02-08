#!/usr/bin/env python3
"""
Debug JP Protocol Physics
=========================

This script investigates why the JP (Jandura-Pupillo) protocol doesn't produce
a correct CZ gate. We'll:

1. Verify basic Rabi physics (oscillation frequencies)
2. Verify phase modulation effects
3. Search for single-pulse CZ parameters with Δ=0
4. Search for bang-bang parameters that work

The goal is to find parameters that work, not copy them from papers.
"""

import numpy as np
from qutip import tensor, mesolve, basis, Qobj
import warnings
warnings.filterwarnings('ignore')

# Import from our codebase
import sys
sys.path.insert(0, 'src')

from qpu_simulator.micro_physics.neutral_atoms.rydberg_gates.hamiltonians import (
    build_hilbert_space,
    build_laser_hamiltonian,
    build_detuning_hamiltonian,
    build_interaction_hamiltonian,
    build_phase_modulated_hamiltonian,
)

# Build Hilbert space
hs = build_hilbert_space(3)
b0 = hs.basis['0']
b1 = hs.basis['1']
br = hs.basis['r']

# Two-atom basis states
psi_00 = tensor(b0, b0)
psi_01 = tensor(b0, b1)
psi_10 = tensor(b1, b0)
psi_11 = tensor(b1, b1)
psi_0r = tensor(b0, br)
psi_r0 = tensor(br, b0)
psi_1r = tensor(b1, br)
psi_r1 = tensor(br, b1)
psi_rr = tensor(br, br)

# Symmetric/antisymmetric states for |11⟩ dynamics
psi_plus = (psi_1r + psi_r1).unit()  # Bright state
psi_minus = (psi_1r - psi_r1).unit()  # Dark state


def get_overlap(psi1, psi2):
    """Get complex overlap ⟨ψ1|ψ2⟩."""
    if psi1.type == 'oper':
        # Density matrix - extract dominant eigenvector
        evals, evecs = psi1.eigenstates()
        psi1 = evecs[-1]
    if psi2.type == 'oper':
        evals, evecs = psi2.eigenstates()
        psi2 = evecs[-1]
    result = psi1.dag() * psi2
    # Handle both Qobj and scalar results
    if hasattr(result, 'tr'):
        return result.tr()
    elif hasattr(result, 'full'):
        return complex(result.full()[0, 0])
    else:
        return complex(result)


def evolve(H, psi0, t_total, n_steps=100):
    """Evolve state under Hamiltonian."""
    tlist = np.linspace(0, t_total, n_steps)
    result = mesolve(H, psi0, tlist, c_ops=[], options={'atol': 1e-10, 'rtol': 1e-8})
    return result.states[-1]


def get_phase(overlap):
    """Extract phase from complex overlap."""
    return np.angle(overlap)


# =============================================================================
# TEST 1: Verify Rabi oscillation frequencies
# =============================================================================
def test_rabi_frequencies():
    """
    Verify that:
    - |01⟩ ↔ |0r⟩ oscillates at frequency Ω
    - |11⟩ ↔ |+⟩ oscillates at frequency √2·Ω (due to collective enhancement)
    """
    print("="*70)
    print("TEST 1: RABI OSCILLATION FREQUENCIES")
    print("="*70)
    
    Omega = 2 * np.pi * 1e6  # 1 MHz for easy math
    V = 100 * Omega  # Strong blockade
    
    # Build Hamiltonian with Δ=0 (resonant)
    H = build_phase_modulated_hamiltonian(Omega=Omega, phase=0, V=V, hs=hs, Delta=0)
    
    # Expected π-pulse times
    t_pi_01 = np.pi / Omega  # For |01⟩
    t_pi_11 = np.pi / (np.sqrt(2) * Omega)  # For |11⟩ (√2 enhancement)
    
    print(f"\nΩ/(2π) = {Omega/(2*np.pi)/1e6:.2f} MHz")
    print(f"V/Ω = {V/Omega:.0f}")
    print(f"\nExpected π-pulse times:")
    print(f"  |01⟩: t_π = π/Ω = {t_pi_01*1e6:.3f} μs")
    print(f"  |11⟩: t_π = π/(√2·Ω) = {t_pi_11*1e6:.3f} μs")
    
    # Evolve |01⟩ for t_pi_01 and check it goes to |0r⟩
    psi_01_evolved = evolve(H, psi_01, t_pi_01)
    overlap_0r = abs(get_overlap(psi_0r, psi_01_evolved))**2
    print(f"\n|01⟩ after t=π/Ω:")
    print(f"  P(|0r⟩) = {overlap_0r:.4f} (expect ~1.0)")
    
    # Evolve |01⟩ for 2*t_pi_01 and check it returns to |01⟩
    psi_01_full = evolve(H, psi_01, 2*t_pi_01)
    overlap_01_return = abs(get_overlap(psi_01, psi_01_full))**2
    phase_01 = get_phase(get_overlap(psi_01, psi_01_full))
    print(f"|01⟩ after t=2π/Ω (full Rabi cycle):")
    print(f"  P(|01⟩) = {overlap_01_return:.4f} (expect ~1.0)")
    print(f"  Phase = {np.degrees(phase_01):.1f}° (expect ~-180° or +180°)")
    
    # Evolve |11⟩ for t_pi_11 and check population transfer
    psi_11_evolved = evolve(H, psi_11, t_pi_11)
    overlap_plus = abs(get_overlap(psi_plus, psi_11_evolved))**2
    print(f"\n|11⟩ after t=π/(√2·Ω):")
    print(f"  P(|+⟩) = {overlap_plus:.4f} (expect ~1.0)")
    
    # Evolve |11⟩ for 2*t_pi_11 and check return
    psi_11_full = evolve(H, psi_11, 2*t_pi_11)
    overlap_11_return = abs(get_overlap(psi_11, psi_11_full))**2
    phase_11 = get_phase(get_overlap(psi_11, psi_11_full))
    print(f"|11⟩ after t=2π/(√2·Ω) (full Rabi cycle):")
    print(f"  P(|11⟩) = {overlap_11_return:.4f} (expect ~1.0)")
    print(f"  Phase = {np.degrees(phase_11):.1f}° (expect ~-180° or +180°)")
    
    # Check frequency ratio
    print(f"\nFrequency ratio check:")
    print(f"  t_π(|01⟩) / t_π(|11⟩) = {t_pi_01/t_pi_11:.4f} (expect √2 ≈ 1.414)")
    
    return overlap_0r > 0.95 and overlap_plus > 0.95


# =============================================================================
# TEST 2: Single-pulse CZ with detuning (like LP but single pulse)
# =============================================================================
def search_single_pulse_cz():
    """
    Search for single-pulse parameters (Ωτ, Δ/Ω) that produce CZ.
    
    Requirements:
    - |01⟩ returns to |01⟩
    - |11⟩ returns to |11⟩  
    - Controlled phase = φ₁₁ - 2φ₀₁ ≈ ±π
    """
    print("\n" + "="*70)
    print("TEST 2: SEARCH FOR SINGLE-PULSE CZ (WITH DETUNING)")
    print("="*70)
    print("\nSearching (Ωτ, Δ/Ω) space for CZ gate...")
    
    Omega = 2 * np.pi * 5e6
    V = 50 * Omega  # V/Ω = 50
    
    best_fidelity = 0
    best_params = None
    
    results = []
    
    # Coarse scan
    for omega_tau in np.linspace(4.0, 10.0, 25):
        for delta_ratio in np.linspace(0.0, 0.6, 25):
            tau = omega_tau / Omega
            Delta = delta_ratio * Omega
            
            H = build_phase_modulated_hamiltonian(Omega=Omega, phase=0, V=V, hs=hs, Delta=Delta)
            
            # Evolve all states
            psi_01_f = evolve(H, psi_01, tau, n_steps=50)
            psi_11_f = evolve(H, psi_11, tau, n_steps=50)
            
            # Get overlaps and phases
            ov_01 = get_overlap(psi_01, psi_01_f)
            ov_11 = get_overlap(psi_11, psi_11_f)
            
            overlap_01 = abs(ov_01)
            overlap_11 = abs(ov_11)
            phase_01 = get_phase(ov_01)
            phase_11 = get_phase(ov_11)
            
            # Controlled phase
            ctrl_phase = phase_11 - 2*phase_01
            ctrl_phase_wrapped = np.arctan2(np.sin(ctrl_phase), np.cos(ctrl_phase))
            
            # CZ fidelity metric
            phase_error = min(abs(ctrl_phase_wrapped - np.pi), abs(ctrl_phase_wrapped + np.pi))
            fidelity = overlap_01 * overlap_11 * (1 - phase_error/np.pi)
            
            results.append((omega_tau, delta_ratio, overlap_01, overlap_11, 
                          np.degrees(ctrl_phase_wrapped), fidelity))
            
            if fidelity > best_fidelity:
                best_fidelity = fidelity
                best_params = (omega_tau, delta_ratio, overlap_01, overlap_11, 
                             np.degrees(ctrl_phase_wrapped))
    
    print(f"\nBest single-pulse result:")
    print(f"  Ωτ = {best_params[0]:.3f}")
    print(f"  Δ/Ω = {best_params[1]:.4f}")
    print(f"  |01⟩ overlap = {best_params[2]:.4f}")
    print(f"  |11⟩ overlap = {best_params[3]:.4f}")
    print(f"  Controlled phase = {best_params[4]:.1f}° (need ±180°)")
    print(f"  Fidelity metric = {best_fidelity:.4f}")
    
    # Note: Single pulse with constant phase likely CAN'T produce CZ
    # because |01⟩ and |11⟩ have different Rabi frequencies
    print("\nNote: Single constant-phase pulse likely cannot produce CZ")
    print("because Ω_eff(|11⟩) = √2·Ω ≠ Ω_eff(|01⟩) = Ω")
    
    return best_params, results


# =============================================================================
# TEST 3: Bang-bang phase modulation search
# =============================================================================
def search_bangbang_simple():
    """
    Search for simple bang-bang parameters with 3 segments.
    
    Phases cycle through: [φ₁, φ₂, φ₃]
    Switching at: [t₁, t₂] (in units of Ωt)
    """
    print("\n" + "="*70)
    print("TEST 3: SIMPLE BANG-BANG SEARCH (3 SEGMENTS)")
    print("="*70)
    
    Omega = 2 * np.pi * 5e6
    V = 50 * Omega
    
    # Fix total time and search over switching points and phases
    omega_tau_total = 2 * np.pi  # One "natural" period
    tau_total = omega_tau_total / Omega
    
    phase_options = [-np.pi/2, 0, np.pi/2]
    
    best_fidelity = 0
    best_config = None
    
    print(f"\nSearching 3-segment bang-bang with Ωτ = {omega_tau_total:.3f} = 2π")
    print("Phase options: [-π/2, 0, +π/2]")
    
    # Grid search over switching times
    n_grid = 15
    for t1_frac in np.linspace(0.1, 0.5, n_grid):
        for t2_frac in np.linspace(0.5, 0.9, n_grid):
            if t2_frac <= t1_frac:
                continue
                
            t1 = t1_frac * tau_total
            t2 = t2_frac * tau_total
            
            # Try all phase combinations
            for p1 in phase_options:
                for p2 in phase_options:
                    for p3 in phase_options:
                        phases = [p1, p2, p3]
                        times = [0, t1, t2, tau_total]
                        
                        # Evolve through segments
                        psi_01_f = psi_01
                        psi_11_f = psi_11
                        
                        for i in range(3):
                            dt = times[i+1] - times[i]
                            H = build_phase_modulated_hamiltonian(
                                Omega=Omega, phase=phases[i], V=V, hs=hs, Delta=0
                            )
                            psi_01_f = evolve(H, psi_01_f, dt, n_steps=20)
                            psi_11_f = evolve(H, psi_11_f, dt, n_steps=20)
                        
                        # Compute metrics
                        ov_01 = get_overlap(psi_01, psi_01_f)
                        ov_11 = get_overlap(psi_11, psi_11_f)
                        
                        overlap_01 = abs(ov_01)
                        overlap_11 = abs(ov_11)
                        phase_01 = get_phase(ov_01)
                        phase_11 = get_phase(ov_11)
                        
                        ctrl_phase = phase_11 - 2*phase_01
                        ctrl_phase_wrapped = np.arctan2(np.sin(ctrl_phase), np.cos(ctrl_phase))
                        
                        phase_error = min(abs(ctrl_phase_wrapped - np.pi), 
                                        abs(ctrl_phase_wrapped + np.pi))
                        fidelity = overlap_01 * overlap_11 * (1 - phase_error/np.pi)
                        
                        if fidelity > best_fidelity:
                            best_fidelity = fidelity
                            best_config = {
                                'omega_tau': omega_tau_total,
                                't1_frac': t1_frac,
                                't2_frac': t2_frac,
                                'phases': phases,
                                'overlap_01': overlap_01,
                                'overlap_11': overlap_11,
                                'ctrl_phase_deg': np.degrees(ctrl_phase_wrapped),
                                'fidelity': fidelity,
                            }
    
    print(f"\nBest 3-segment bang-bang result:")
    if best_config:
        print(f"  Ωτ = {best_config['omega_tau']:.3f}")
        print(f"  Switching: [{best_config['t1_frac']:.2f}, {best_config['t2_frac']:.2f}] × τ")
        print(f"  Phases: {[f'{p/np.pi:.2f}π' for p in best_config['phases']]}")
        print(f"  |01⟩ overlap = {best_config['overlap_01']:.4f}")
        print(f"  |11⟩ overlap = {best_config['overlap_11']:.4f}")
        print(f"  Controlled phase = {best_config['ctrl_phase_deg']:.1f}°")
        print(f"  Fidelity = {best_config['fidelity']:.4f}")
    
    return best_config


# =============================================================================
# TEST 4: Extended bang-bang search (more segments, variable total time)
# =============================================================================
def search_bangbang_extended():
    """
    Extended search with 5 segments and variable total time.
    """
    print("\n" + "="*70)
    print("TEST 4: EXTENDED BANG-BANG SEARCH (5 SEGMENTS)")  
    print("="*70)
    
    Omega = 2 * np.pi * 5e6
    V = 50 * Omega
    
    phase_options = [-np.pi/2, 0, np.pi/2]
    
    best_fidelity = 0
    best_config = None
    
    # Search over different total times
    for omega_tau_total in [2*np.pi, 2.5*np.pi, 3*np.pi, 4*np.pi, 5*np.pi, 6*np.pi]:
        tau_total = omega_tau_total / Omega
        
        # Use symmetric patterns (reduces search space)
        # Pattern: [p1, p2, p3, p2, p1] (palindrome)
        for p1 in phase_options:
            for p2 in phase_options:
                for p3 in phase_options:
                    phases = [p1, p2, p3, p2, p1]
                    
                    # Symmetric timing: [t1, t2, 0.5, 1-t2, 1-t1]
                    for t1_frac in [0.1, 0.15, 0.2]:
                        for t2_frac in [0.25, 0.3, 0.35, 0.4]:
                            if t2_frac <= t1_frac:
                                continue
                            
                            # Symmetric switching times
                            switches = [
                                t1_frac * tau_total,
                                t2_frac * tau_total,
                                0.5 * tau_total,
                                (1-t2_frac) * tau_total,
                            ]
                            times = [0] + switches + [tau_total]
                            
                            # Evolve
                            psi_01_f = psi_01
                            psi_11_f = psi_11
                            
                            for i in range(5):
                                dt = times[i+1] - times[i]
                                if dt <= 0:
                                    continue
                                H = build_phase_modulated_hamiltonian(
                                    Omega=Omega, phase=phases[i], V=V, hs=hs, Delta=0
                                )
                                psi_01_f = evolve(H, psi_01_f, dt, n_steps=15)
                                psi_11_f = evolve(H, psi_11_f, dt, n_steps=15)
                            
                            # Metrics
                            ov_01 = get_overlap(psi_01, psi_01_f)
                            ov_11 = get_overlap(psi_11, psi_11_f)
                            
                            overlap_01 = abs(ov_01)
                            overlap_11 = abs(ov_11)
                            ctrl_phase = get_phase(ov_11) - 2*get_phase(ov_01)
                            ctrl_phase_wrapped = np.arctan2(np.sin(ctrl_phase), np.cos(ctrl_phase))
                            
                            phase_error = min(abs(ctrl_phase_wrapped - np.pi),
                                            abs(ctrl_phase_wrapped + np.pi))
                            fidelity = overlap_01 * overlap_11 * (1 - phase_error/np.pi)
                            
                            if fidelity > best_fidelity:
                                best_fidelity = fidelity
                                best_config = {
                                    'omega_tau': omega_tau_total,
                                    'switches_frac': [t1_frac, t2_frac, 0.5, 1-t2_frac],
                                    'phases': phases,
                                    'overlap_01': overlap_01,
                                    'overlap_11': overlap_11,
                                    'ctrl_phase_deg': np.degrees(ctrl_phase_wrapped),
                                    'fidelity': fidelity,
                                }
    
    print(f"\nBest 5-segment symmetric bang-bang:")
    if best_config:
        print(f"  Ωτ = {best_config['omega_tau']:.3f} = {best_config['omega_tau']/np.pi:.2f}π")
        print(f"  Switching fractions: {best_config['switches_frac']}")
        print(f"  Phases: {[f'{p/np.pi:.2f}π' for p in best_config['phases']]}")
        print(f"  |01⟩ overlap = {best_config['overlap_01']:.4f}")
        print(f"  |11⟩ overlap = {best_config['overlap_11']:.4f}")
        print(f"  Controlled phase = {best_config['ctrl_phase_deg']:.1f}°")
        print(f"  Fidelity = {best_config['fidelity']:.4f}")
    
    return best_config


# =============================================================================
# TEST 5: Fine-grained optimization around promising points
# =============================================================================
def optimize_bangbang(initial_config):
    """
    Fine-tune a promising bang-bang configuration.
    """
    print("\n" + "="*70)
    print("TEST 5: FINE-TUNING BEST CONFIGURATION")
    print("="*70)
    
    if initial_config is None or initial_config['fidelity'] < 0.5:
        print("No good initial configuration to optimize.")
        return None
    
    Omega = 2 * np.pi * 5e6
    V = 50 * Omega
    
    # Start from best config
    omega_tau_best = initial_config['omega_tau']
    
    best_fidelity = initial_config['fidelity']
    best_config = initial_config.copy()
    
    print(f"Starting from fidelity = {best_fidelity:.4f}")
    
    # Fine search around omega_tau
    for omega_tau in np.linspace(omega_tau_best * 0.9, omega_tau_best * 1.1, 20):
        tau_total = omega_tau / Omega
        
        phases = initial_config['phases']
        n_seg = len(phases)
        
        # For symmetric 5-segment, parameterize by first two fractions
        if n_seg == 5:
            base_t1 = initial_config['switches_frac'][0]
            base_t2 = initial_config['switches_frac'][1]
            
            for dt1 in np.linspace(-0.05, 0.05, 10):
                for dt2 in np.linspace(-0.05, 0.05, 10):
                    t1_frac = base_t1 + dt1
                    t2_frac = base_t2 + dt2
                    
                    if t1_frac <= 0 or t2_frac <= t1_frac or t2_frac >= 0.5:
                        continue
                    
                    switches = [
                        t1_frac * tau_total,
                        t2_frac * tau_total,
                        0.5 * tau_total,
                        (1-t2_frac) * tau_total,
                    ]
                    times = [0] + switches + [tau_total]
                    
                    psi_01_f = psi_01
                    psi_11_f = psi_11
                    
                    for i in range(5):
                        dt = times[i+1] - times[i]
                        if dt <= 0:
                            continue
                        H = build_phase_modulated_hamiltonian(
                            Omega=Omega, phase=phases[i], V=V, hs=hs, Delta=0
                        )
                        psi_01_f = evolve(H, psi_01_f, dt, n_steps=20)
                        psi_11_f = evolve(H, psi_11_f, dt, n_steps=20)
                    
                    ov_01 = get_overlap(psi_01, psi_01_f)
                    ov_11 = get_overlap(psi_11, psi_11_f)
                    
                    overlap_01 = abs(ov_01)
                    overlap_11 = abs(ov_11)
                    ctrl_phase = get_phase(ov_11) - 2*get_phase(ov_01)
                    ctrl_phase_wrapped = np.arctan2(np.sin(ctrl_phase), np.cos(ctrl_phase))
                    
                    phase_error = min(abs(ctrl_phase_wrapped - np.pi),
                                    abs(ctrl_phase_wrapped + np.pi))
                    fidelity = overlap_01 * overlap_11 * (1 - phase_error/np.pi)
                    
                    if fidelity > best_fidelity:
                        best_fidelity = fidelity
                        best_config = {
                            'omega_tau': omega_tau,
                            'switches_frac': [t1_frac, t2_frac, 0.5, 1-t2_frac],
                            'phases': phases,
                            'overlap_01': overlap_01,
                            'overlap_11': overlap_11,
                            'ctrl_phase_deg': np.degrees(ctrl_phase_wrapped),
                            'fidelity': fidelity,
                        }
    
    print(f"\nOptimized result:")
    print(f"  Ωτ = {best_config['omega_tau']:.4f} = {best_config['omega_tau']/np.pi:.4f}π")
    print(f"  Switching fractions: {[f'{x:.4f}' for x in best_config['switches_frac']]}")
    print(f"  Phases: {[f'{p/np.pi:.2f}π' for p in best_config['phases']]}")
    print(f"  |01⟩ overlap = {best_config['overlap_01']:.4f}")
    print(f"  |11⟩ overlap = {best_config['overlap_11']:.4f}")
    print(f"  Controlled phase = {best_config['ctrl_phase_deg']:.1f}°")
    print(f"  Fidelity = {best_config['fidelity']:.4f}")
    
    return best_config


# =============================================================================
# TEST 6: Check the stored JP parameters
# =============================================================================
def test_stored_jp_params():
    """
    Test the JP parameters currently stored in protocols.py
    """
    print("\n" + "="*70)
    print("TEST 6: VERIFY STORED JP PARAMETERS")
    print("="*70)
    
    Omega = 2 * np.pi * 5e6
    V = 200 * Omega  # V/Ω = 200 (the paper's optimization point)
    
    # Stored parameters
    omega_tau = 7.0
    switching_times = [0.3328, 0.5859, 3.4340, 3.5530, 4.1204, 6.7431]
    phases = [np.pi/2, 0, -np.pi/2, -np.pi/2, 0, np.pi/2, 0]
    
    tau_total = omega_tau / Omega
    
    print(f"Parameters from protocols.py:")
    print(f"  Ωτ = {omega_tau}")
    print(f"  V/Ω = {V/Omega:.0f}")
    print(f"  Switching times (Ωt): {switching_times}")
    print(f"  Phases: {[f'{p/np.pi:.2f}π' for p in phases]}")
    
    # Convert switching times to physical times
    switch_physical = [t / Omega for t in switching_times]
    boundaries = [0.0] + switch_physical + [tau_total]
    
    print(f"\nSegment durations (Ωt):")
    for i in range(len(phases)):
        dt_omega = (boundaries[i+1] - boundaries[i]) * Omega
        print(f"  Segment {i}: Ωdt = {dt_omega:.4f}, φ = {phases[i]/np.pi:.2f}π")
    
    # Evolve
    psi_01_f = psi_01
    psi_11_f = psi_11
    
    for i in range(len(phases)):
        dt = boundaries[i+1] - boundaries[i]
        H = build_phase_modulated_hamiltonian(
            Omega=Omega, phase=phases[i], V=V, hs=hs, Delta=0
        )
        psi_01_f = evolve(H, psi_01_f, dt, n_steps=30)
        psi_11_f = evolve(H, psi_11_f, dt, n_steps=30)
    
    ov_01 = get_overlap(psi_01, psi_01_f)
    ov_11 = get_overlap(psi_11, psi_11_f)
    
    overlap_01 = abs(ov_01)
    overlap_11 = abs(ov_11)
    phase_01 = np.degrees(get_phase(ov_01))
    phase_11 = np.degrees(get_phase(ov_11))
    
    ctrl_phase = get_phase(ov_11) - 2*get_phase(ov_01)
    ctrl_phase_wrapped = np.degrees(np.arctan2(np.sin(ctrl_phase), np.cos(ctrl_phase)))
    
    print(f"\nResults:")
    print(f"  |01⟩ overlap = {overlap_01:.4f}, phase = {phase_01:.1f}°")
    print(f"  |11⟩ overlap = {overlap_11:.4f}, phase = {phase_11:.1f}°")
    print(f"  Controlled phase = {ctrl_phase_wrapped:.1f}° (need ±180°)")
    
    is_good = overlap_01 > 0.95 and overlap_11 > 0.95 and abs(abs(ctrl_phase_wrapped) - 180) < 10
    print(f"\n{'✓ PASS' if is_good else '✗ FAIL'}: Stored JP parameters {'work' if is_good else 'do NOT work'}")
    
    return is_good


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("="*70)
    print("JP PROTOCOL PHYSICS INVESTIGATION")
    print("="*70)
    print("\nGoal: Find working bang-bang parameters for CZ gate with Δ=0")
    print("The Hamiltonian is correct - we just need the right parameters.\n")
    
    # Test 1: Basic physics
    test_rabi_frequencies()
    
    # Test 2: Single pulse (expected to fail)
    search_single_pulse_cz()
    
    # Test 3: Simple bang-bang
    config_3seg = search_bangbang_simple()
    
    # Test 4: Extended bang-bang
    config_5seg = search_bangbang_extended()
    
    # Test 5: Optimize best result
    best_config = config_5seg if config_5seg and config_5seg['fidelity'] > (config_3seg['fidelity'] if config_3seg else 0) else config_3seg
    optimized = optimize_bangbang(best_config)
    
    # Test 6: Check stored parameters
    test_stored_jp_params()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    if optimized and optimized['fidelity'] > 0.95:
        print(f"✓ Found working bang-bang parameters with fidelity {optimized['fidelity']:.4f}")
        print(f"  Ωτ = {optimized['omega_tau']:.4f}")
        print(f"  Update protocols.py with these values!")
    else:
        print("✗ Could not find high-fidelity bang-bang parameters in search")
        print("  The JP protocol may require more segments or different search strategy")
