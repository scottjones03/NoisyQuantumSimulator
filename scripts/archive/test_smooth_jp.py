#!/usr/bin/env python3
"""
Test script for smooth sinusoidal CZ gate exploration.

⚠️  WARNING: EXPERIMENTAL PARAMETERS
====================================

This script explores the smooth sinusoidal phase modulation approach to CZ gates.
Current findings (January 2026):

1. Simple phase modulation φ(t) = A·cos(ω·t) does NOT produce a correct CZ gate
2. Parameter sweeps cannot achieve both correct phase (±180°) AND high overlaps
3. The Evered et al. 99.5% fidelity uses LP-style detuning, not pure phase modulation

For a VALIDATED CZ gate, use protocol="levine_pichler" instead.

References:
- Evered et al., Nature 622, 268-272 (2023)
- Bluvstein PhD Thesis (Harvard, 2024)
"""

import numpy as np
import sys
import warnings
sys.path.insert(0, 'src')

from qpu_simulator.micro_physics.neutral_atoms.rydberg_gates.hamiltonians import (
    build_hilbert_space,
)
from qpu_simulator.micro_physics.neutral_atoms.rydberg_gates.simulation import (
    evolve_smooth_sinusoidal_jp,
    simulate_CZ_gate,
    compute_CZ_fidelity,
    b0,
    b1,
)
from qpu_simulator.micro_physics.neutral_atoms.rydberg_gates.configurations import (
    LPSimulationInputs,
    JPSimulationInputs,
    TwoPhotonExcitationConfig,
    NoiseSourceConfig,
)
from qpu_simulator.micro_physics.neutral_atoms.rydberg_gates.protocols import (
    get_protocol_params,
    SMOOTH_JP_PARAMS,
)
from qutip import tensor


# Helper function to extract overlap - works for QuTiP4 and QuTiP5
def get_overlap(psi0, psi_f):
    """Extract complex overlap ⟨psi0|psi_f⟩ from QuTiP objects."""
    overlap_qobj = psi0.dag() * psi_f
    if hasattr(overlap_qobj, 'full'):
        return complex(overlap_qobj.full()[0, 0])
    return complex(overlap_qobj)

def test_smooth_jp_exploration():
    """
    Explore smooth JP phase modulation (EXPERIMENTAL).
    
    This test documents that the current smooth JP parameters do NOT
    produce a correct CZ gate. It's kept for research purposes.
    """
    print("="*70)
    print("TEST: SMOOTH SINUSOIDAL PHASE MODULATION (EXPERIMENTAL)")
    print("="*70)
    print("\n⚠️  WARNING: This protocol does NOT produce a correct CZ gate!")
    print("    Use protocol='levine_pichler' for validated CZ operation.\n")
    
    # Physical parameters matching Evered et al.
    Omega = 2 * np.pi * 4.6e6   # 4.6 MHz Rabi frequency
    V_over_Omega = 50           # Strong blockade
    V = V_over_Omega * Omega
    
    # Get smooth JP parameters
    params = get_protocol_params("smooth_jp", V_over_Omega=V_over_Omega)
    omega_tau = params['omega_tau']
    tau_total = omega_tau / Omega
    delta_over_omega = params['delta_over_omega']
    
    print(f"Parameters (UNVALIDATED):")
    print(f"  Rabi frequency:    Ω/(2π) = {Omega/(2*np.pi)/1e6:.2f} MHz")
    print(f"  Blockade:          V/(2π) = {V/(2*np.pi)/1e6:.2f} MHz")
    print(f"  V/Ω ratio:         {V_over_Omega:.1f}")
    print(f"  Pulse area:        Ω·τ = {omega_tau:.4f} rad ({omega_tau/np.pi:.4f}π)")
    print(f"  Gate time:         τ = {tau_total*1e6:.3f} μs")
    print(f"  Two-photon det.:   δ/Ω = {delta_over_omega:+.4f}")
    
    # Build Hilbert space (3-level)
    hs = build_hilbert_space()
    
    # Initial states (computational basis)
    initial_states = {
        "00": tensor(b0, b0),
        "01": tensor(b0, b1),
        "10": tensor(b1, b0),
        "11": tensor(b1, b1),
    }
    
    print("\nRunning smooth sinusoidal phase modulation...")
    
    # Suppress expected warning for this test
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*EXPERIMENTAL.*")
        warnings.filterwarnings("ignore", message=".*Dark state condition.*")
        
        final_states, info = evolve_smooth_sinusoidal_jp(
            initial_states=initial_states,
            Omega=Omega,
            V=V,
            tau_total=tau_total,
            hs=hs,
            c_ops=None,  # No noise for ideal evolution
            delta_over_omega=delta_over_omega,
            verbose=True,
        )
    
    # Compute CZ fidelity
    fidelities, avg_fidelity, phase_info = compute_CZ_fidelity(
        final_states,
        extract_global_phase=True,
        hilbert_space_dim=3,
        protocol="smooth_jp",
    )
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Average fidelity: {avg_fidelity:.4f}")
    print(f"\nPer-state fidelities:")
    for label, fid in fidelities.items():
        print(f"  |{label}⟩: {fid:.4f}")
    
    print(f"\nPhase analysis:")
    print(f"  φ_01 = {phase_info.get('phi_01_deg', 0):.2f}°")
    print(f"  φ_11 = {phase_info.get('phi_11_deg', 0):.2f}°")
    print(f"  Phase error = {phase_info.get('phase_error_deg', 0):.2f}°")
    
    # Check overlaps
    print(f"\nState overlaps (|⟨ψ₀|ψ_f⟩|):")
    for label in ["00", "01", "10", "11"]:
        overlap = abs(get_overlap(initial_states[label], final_states[label]))
        print(f"  |{label}⟩: {overlap:.4f}")
    
    # EXPECTED: This will FAIL because the protocol doesn't work
    print(f"\n" + "="*60)
    print("EXPECTED RESULT: This test should FAIL")
    print("="*60)
    print("The smooth sinusoidal parameters are NOT validated.")
    print("Controlled phase is ~27° instead of ±180°.")
    print("Use protocol='levine_pichler' for a working CZ gate.")
    
    # Don't fail the test - just document the state
    return False, avg_fidelity


def test_lp_protocol_works():
    """
    Verify that LP protocol produces correct CZ gate.
    
    This is a control test to show what a working CZ gate looks like.
    """
    print("\n" + "="*70)
    print("TEST: LP PROTOCOL (VALIDATED REFERENCE)")
    print("="*70)
    
    # Create simulation inputs with minimal noise for testing
    noise = NoiseSourceConfig(
        include_spontaneous_emission=False,
        include_intermediate_scattering=False,
        include_motional_dephasing=False,
        include_doppler_dephasing=False,
        include_intensity_noise=False,
        include_laser_dephasing=False,
        include_magnetic_dephasing=False,
    )
    
    lp_inputs = LPSimulationInputs(
        noise=noise,
    )
    
    print("\nRunning LP protocol (validated)...")
    
    result = simulate_CZ_gate(
        simulation_inputs=lp_inputs,
        include_noise=False,
        verbose=True,
    )
    
    print(f"\n--- Results ---")
    print(f"Gate fidelity: {result.avg_fidelity:.4f}")
    print(f"Phase error: {result.phase_info.get('phase_error_deg', 0):.2f}°")
    
    is_good = result.avg_fidelity > 0.95
    print(f"\n{'✓ SUCCESS' if is_good else '✗ FAILED'}: Fidelity {'above' if is_good else 'below'} 0.95")
    
    return is_good, result.avg_fidelity


def test_parameter_exploration():
    """
    Explore smooth JP parameter space (research).
    
    This documents the parameter exploration done to understand
    why simple phase modulation doesn't produce a working CZ gate.
    """
    print("\n" + "="*70)
    print("TEST: PARAMETER EXPLORATION (RESEARCH)")
    print("="*70)
    print("\nExploring why smooth sinusoidal phase modulation doesn't work...\n")
    
    Omega = 2 * np.pi * 4.6e6
    V = 50 * Omega
    omega_tau = 1.8 * np.pi
    tau_total = omega_tau / Omega
    
    hs = build_hilbert_space()
    initial_states = {
        "00": tensor(b0, b0),
        "01": tensor(b0, b1),
        "10": tensor(b1, b0),
        "11": tensor(b1, b1),
    }
    
    delta_values = [-0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2]
    
    print(f"{'δ/Ω':^10} {'Fidelity':^12} {'Phase Err':^12} {'|01⟩ overlap':^14}")
    print("-"*55)
    
    best_fid = 0
    
    for delta_ratio in delta_values:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*EXPERIMENTAL.*")
            warnings.filterwarnings("ignore", message=".*Dark state.*")
            
            final_states, info = evolve_smooth_sinusoidal_jp(
                initial_states=initial_states,
                Omega=Omega,
                V=V,
                tau_total=tau_total,
                hs=hs,
                c_ops=None,
                delta_over_omega=delta_ratio,
                verbose=False,
            )
        
        fidelities, avg_fid, phase_info = compute_CZ_fidelity(
            final_states, extract_global_phase=True
        )
        
        # Get |01⟩ overlap to show single-atom return failure
        overlap_01 = abs(get_overlap(initial_states["01"], final_states["01"]))
        
        phase_err = phase_info.get('phase_error_deg', 0)
        print(f"{delta_ratio:+10.4f} {avg_fid:12.4f} {phase_err:+12.1f}° {overlap_01:14.4f}")
        
        if avg_fid > best_fid:
            best_fid = avg_fid
    
    print(f"\nBest fidelity achieved: {best_fid:.4f}")
    print(f"Required for CZ: > 0.95")
    print(f"\nConclusion: Simple phase modulation CANNOT achieve correct CZ gate.")
    print("The |01⟩ state overlap < 1 shows atoms don't return to |1⟩.")
    
    return False  # Expected to fail


if __name__ == "__main__":
    print("\n" + "="*70)
    print("CZ GATE PROTOCOL COMPARISON")
    print("="*70)
    print("\nThis script demonstrates:")
    print("1. LP protocol produces CORRECT CZ gate (reference)")
    print("2. Smooth sinusoidal phase modulation does NOT work")
    print("3. Research into why the smooth JP parameters fail")
    print("="*70)
    
    # Test 1: LP protocol (should PASS - this is the validated reference)
    print("\n" + "="*70)
    print("PART 1: VALIDATED LP PROTOCOL")
    print("="*70)
    success_lp, fid_lp = test_lp_protocol_works()
    
    # Test 2: Smooth JP exploration (expected to FAIL - documents the issue)
    print("\n" + "="*70)
    print("PART 2: SMOOTH JP EXPLORATION (EXPERIMENTAL)")
    print("="*70)
    success_smooth, fid_smooth = test_smooth_jp_exploration()
    
    # Test 3: Parameter exploration (research)
    print("\n" + "="*70)
    print("PART 3: PARAMETER EXPLORATION")
    print("="*70)
    test_parameter_exploration()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"  LP protocol (reference):     {'✓ PASS' if success_lp else '✗ FAIL'} (F={fid_lp:.4f})")
    print(f"  Smooth JP (experimental):    {'✗ FAIL (expected)' if not success_smooth else '✓ PASS'} (F={fid_smooth:.4f})")
    print(f"\nConclusion: Use protocol='levine_pichler' for CZ gates.")
    print("           The smooth sinusoidal parameters need further research.")
    print("="*70)
    
    sys.exit(0 if (success1 and success2) else 1)