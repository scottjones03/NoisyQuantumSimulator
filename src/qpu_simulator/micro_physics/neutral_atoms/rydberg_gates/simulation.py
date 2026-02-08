"""
CZ Gate Simulation
==================

This module provides the main simulation engine for Rydberg CZ gates.
It integrates all the physics components (Hamiltonians, noise, protocols)
into a comprehensive simulation framework.

High-Level Overview
-------------------
The CZ gate simulation proceeds in these stages:

1. **Setup**: Create atomic configuration and compute derived parameters
   - Rabi frequencies from laser powers
   - Blockade strength from inter-atom spacing
   - Trap-dependent noise rates

2. **Hamiltonian construction**: Build the time-dependent Hamiltonian
   - Laser coupling terms
   - Rydberg-Rydberg interaction
   - Detuning and phase modulation

3. **Noise model**: Construct Lindblad collapse operators
   - Rydberg decay
   - Laser and thermal dephasing
   - Atom loss channels

4. **Evolution**: Solve the master equation using QuTiP's mesolve
   - Propagate all 4 computational basis states
   - Track populations and coherences

5. **Analysis**: Compute gate fidelity and diagnostics
   - Compare output states to ideal CZ targets
   - Extract phase information
   - Build error budget

Why Use Master Equation?
------------------------
For open quantum systems with decoherence, we use the Lindblad master
equation instead of Schrödinger evolution:

    dρ/dt = -i[H, ρ] + Σ_k γ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})

where:
- ρ is the density matrix (9×9 for two 3-level atoms)
- H is the system Hamiltonian
- L_k are collapse operators (jump channels)
- γ_k are decay/dephasing rates

This captures both coherent dynamics AND incoherent noise.

Fidelity Calculation
--------------------
For a CZ gate, the ideal operation is:

    |00⟩ → |00⟩
    |01⟩ → e^{iφ} |01⟩
    |10⟩ → e^{iφ} |10⟩
    |11⟩ → e^{i(2φ-π)} |11⟩ = -e^{i2φ} |11⟩

The KEY CZ property: |11⟩ acquires an EXTRA π phase relative to
|01⟩ and |10⟩. This creates entanglement when applied to superposition
states.

Fidelity measures how well our actual evolution matches this ideal:

    F = (1/4) Σ |⟨ψ_ideal|ψ_actual⟩|²

For >99% fidelity (error correction threshold), we need F > 0.99.

References
----------
- Levine et al., PRL 123, 170503 (2019) - Two-pulse protocol
- Jandura & Pupillo, PRX Quantum 3, 010353 (2022) - Time-optimal
- Bluvstein PhD Thesis (Harvard, 2024) - Comprehensive error analysis

Author: Quantum Simulation Team
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Tuple, List, Optional, Union
from dataclasses import dataclass

# QuTiP imports (version 5 compatible)
try:
    from qutip import (
        Qobj, tensor, mesolve, basis, ket2dm, fidelity,
        Options
    )
except ImportError:
    raise ImportError(
        "QuTiP is required for simulation. Install with: pip install qutip"
    )

# Physical constants
from .constants import KB, HBAR

# Protocol parameters and constants
from .protocols import (
    get_protocol_params,
    get_lp_protocol,
    compute_phase_shift_xi,
    JP_SWITCHING_TIMES_DEFAULT,
    JP_PHASES_DEFAULT,
)

# Hamiltonians
from .hamiltonians import (
    build_hilbert_space, 
    build_full_hamiltonian, 
    build_phase_modulated_hamiltonian,
    HilbertSpace,
)

# Atom database
from .atom_database import (
    get_atom_properties, 
    get_ground_state_polarizability, 
    get_C6,
    get_default_intermediate_state,
)

# Laser physics
from .laser_physics import (
    laser_E0,
    single_photon_rabi,
    two_photon_rabi,
    rydberg_blockade,
)

# Trap physics
from .trap_physics import (
    tweezer_spacing,
    compute_trap_dependent_noise,
    calculate_zeeman_shift,
    calculate_stark_shift,
    get_polarizability_at_wavelength,
)

# Noise models
from .noise_models import (
    build_all_noise_operators,
    leakage_rate_to_adjacent_states,
    zeeman_dephasing_rate,
    mJ_mixing_rate,
    rydberg_zeeman_splitting,
)

# Configuration dataclasses
from .configurations import (
    AtomicConfiguration,
    LaserParameters,
    LPSimulationInputs,
    JPSimulationInputs,
    SmoothJPSimulationInputs,
    NoiseSourceConfig,
    TwoPhotonExcitationConfig,
)

# Pulse shaping
from .pulse_shaping import compute_leakage_detuning


# =============================================================================
# BASIS STATES
# =============================================================================
# These are the building blocks for our two-atom Hilbert space

# 3-level basis: |0⟩, |1⟩, |r⟩
b0 = basis(3, 0)  # Ground clock state |0⟩ = |F=1, mF=0⟩
b1 = basis(3, 1)  # Ground clock state |1⟩ = |F=2, mF=0⟩
br = basis(3, 2)  # Rydberg state |r⟩ = |nS_{1/2}, mJ=+1/2⟩

# 4-level basis: |0⟩, |1⟩, |r+⟩, |r-⟩ (includes Zeeman substates)
b0_4 = basis(4, 0)  # |0⟩
b1_4 = basis(4, 1)  # |1⟩
br_plus = basis(4, 2)   # |r+⟩ = |nS, mJ=+1/2⟩
br_minus = basis(4, 3)  # |r-⟩ = |nS, mJ=-1/2⟩


# =============================================================================
# FIDELITY CALCULATION
# =============================================================================

def compute_state_fidelity(
    psi_out: Qobj,
    psi_target: Qobj,
) -> float:
    """
    Compute fidelity between output and target states.
    
    For pure states:
        F = |⟨ψ_target|ψ_out⟩|²
        
    For mixed states (density matrices):
        F = [Tr(√(√ρ_target · ρ_out · √ρ_target))]²
    
    Parameters
    ----------
    psi_out : Qobj
        Output state (ket or density matrix)
    psi_target : Qobj
        Target state (ket or density matrix)
        
    Returns
    -------
    float
        Fidelity between 0 and 1
    """
    if psi_out.isket and psi_target.isket:
        # Pure state overlap
        overlap = psi_target.dag() * psi_out
        # Handle QuTiP version differences
        if hasattr(overlap, 'full'):
            overlap = overlap.full()[0, 0]
        return float(np.abs(overlap)**2)
    else:
        # Mixed state fidelity
        rho_out = ket2dm(psi_out) if psi_out.isket else psi_out
        rho_target = ket2dm(psi_target) if psi_target.isket else psi_target
        return float(fidelity(rho_out, rho_target)**2)


def compute_CZ_fidelity(
    results: Dict[str, Qobj],
    extract_global_phase: bool = True,
    hilbert_space_dim: int = 3,
) -> Tuple[Dict[str, float], float, Dict]:
    """
    Compute average gate fidelity for CZ operation.
    
    Works for BOTH Levine-Pichler (two-pulse) AND Jandura-Pupillo (one-pulse)
    protocols - it's protocol-agnostic and just analyzes whatever output
    states you give it.
    
    Background: Textbook CZ vs Physical Implementation
    ---------------------------------------------------
    You might know the TEXTBOOK CZ gate as:
    
        |00⟩ → |00⟩
        |01⟩ → |01⟩      (no phase!)
        |10⟩ → |10⟩      (no phase!)
        |11⟩ → -|11⟩     (π phase = minus sign)
    
    This is the "ideal" CZ with φ = 0. But here's the thing:
    
    **Physical implementations (both LP and JP) can't produce φ = 0!**
    
    When atoms interact with lasers, they accumulate "dynamical phases" -
    think of it like a spinning phase that builds up during the gate.
    What our protocols ACTUALLY produce is:
    
        |00⟩ → |00⟩                 (no laser coupling, no phase)
        |01⟩ → e^{iφ}|01⟩           (single atom does 2π rotation)
        |10⟩ → e^{iφ}|10⟩           (single atom does 2π rotation)  
        |11⟩ → e^{i(2φ-π)}|11⟩      (BOTH atoms rotate, but blockade adds -π!)
    
    where φ is some protocol-dependent phase (depends on Ω, Δ, τ).
    
    **Why is this still a valid CZ gate?**
    
    The KEY insight: what matters for entanglement is the RELATIVE phase
    between |11⟩ and the other states. Let's compute:
    
        Phase of |11⟩ relative to |01⟩ = (2φ - π) - φ = φ - π
        
    The extra "-π" is ALWAYS there regardless of φ! This is what creates
    entanglement. The φ part can be removed by single-qubit Z rotations
    (which are "free" in most quantum computing architectures).
    
    In circuit terms:
    
        [Physical CZ with φ] = [Z(φ) ⊗ Z(φ)] · [Textbook CZ] · [some Z gates]
    
    So our physical gate IS the textbook CZ, just dressed with some
    single-qubit phases that don't affect its entangling power.
    
    Why Extract Global Phase?
    -------------------------
    Since different parameter choices (Ω, Δ, τ) give different φ values,
    comparing our output to the textbook CZ (with φ=0) would unfairly
    penalize our gate even though it's working perfectly!
    
    **Solution: Phase extraction**
    
    We measure the actual φ from |01⟩ output, then check if |11⟩ has
    the correct RELATIVE phase (2φ - π). This measures the TRUE gate
    quality independent of the arbitrary single-qubit phases.
    
    Think of it like this: we're not asking "did you produce exactly
    the textbook gate?" but rather "did you produce a gate that's
    EQUIVALENT to CZ up to easy-to-fix single-qubit rotations?"
    
    The CZ Phase Condition
    ----------------------
    For a perfect CZ (with any φ), we require:
    
        φ₁₁ = 2φ₀₁ - π   (mod 2π)
    
    Why 2φ₀₁? Because |11⟩ has TWO atoms, each accumulating phase φ.
    Why -π? That's the blockade-induced entangling phase!
    
    This function checks this condition and reports any deviation as
    "phase_error" in the output.
    
    Parameters
    ----------
    results : dict
        Dictionary mapping "00", "01", "10", "11" to output Qobj states.
        These come from evolving initial computational basis states
        through your CZ protocol (LP or JP).
    extract_global_phase : bool
        If True (recommended): Extract φ from |01⟩ output and use
        phase-adjusted targets. This gives the "true" fidelity that
        accounts for arbitrary single-qubit phases.
        If False: Compare directly to textbook CZ (φ=0). Only use this
        if you've specifically designed your protocol to have φ=0.
    hilbert_space_dim : int
        Single-atom Hilbert space dimension:
        - 3: {|0⟩, |1⟩, |r⟩} basis
        - 4: {|0⟩, |1⟩, |r+⟩, |r-⟩} basis (includes Zeeman splitting)
        
    Returns
    -------
    fidelities : dict
        Per-state fidelities {"00": F00, "01": F01, "10": F10, "11": F11}
        Each is |⟨target|actual⟩|², ranging from 0 (orthogonal) to 1 (perfect).
    avg_fidelity : float
        Average fidelity = (F00 + F01 + F10 + F11) / 4.
        This is the main figure of merit. >0.99 is typically needed
        for fault-tolerant quantum computing.
    phase_info : dict
        Diagnostic information (only populated if extract_global_phase=True):
        - phi_01_rad/deg: Extracted single-atom phase
        - phi_11_rad/deg: Extracted two-atom phase  
        - expected_phi_11_rad: What φ₁₁ SHOULD be (= 2φ₀₁ - π)
        - phase_error_rad/deg: Deviation from ideal CZ condition
        - amp_01, amp_11: Overlap amplitudes (should be ~1 if no leakage)
        
    Notes
    -----
    **Protocol Independence:**
    
    This function works identically for:
    - Levine-Pichler (LP): Two pulses with phase shift ξ
    - Jandura-Pupillo (JP): Single pulse with bang-bang phase modulation
    
    Both protocols produce output states with the same structure
    (computational basis with protocol-dependent phases), so the
    same fidelity calculation applies.
    
    **What the amplitudes tell you:**
    
    amp_01 and amp_11 should be close to 1.0. If they're significantly
    less than 1, it means population leaked out of the computational
    subspace into |r⟩ states (bad!) or was lost to decay/dephasing.
    
    **Interpreting phase_error:**
    
    phase_error measures |φ₁₁ - (2φ₀₁ - π)|. 
    - 0°: Perfect CZ phase condition satisfied
    - Small (< 5°): Good gate, small coherent error
    - Large (> 10°): Something wrong with protocol parameters
    """
    # Select basis based on dimension
    if hilbert_space_dim == 3:
        _b0, _b1 = b0, b1
    else:
        _b0, _b1 = b0_4, b1_4
    
    # Computational basis states (two-atom)
    psi_00 = tensor(_b0, _b0)
    psi_01 = tensor(_b0, _b1)
    psi_10 = tensor(_b1, _b0)
    psi_11 = tensor(_b1, _b1)
    
    phase_info = {}
    
    # Check if outputs are density matrices (from mesolve with collapse ops)
    is_density_matrix = results["01"].isoper
    
    if extract_global_phase:
        if is_density_matrix:
            # For density matrices, we need to extract phase from coherences.
            # The key insight: even for mixed states, if the gate worked correctly,
            # the coherence ⟨target|ρ|target⟩ should be close to 1 (population in target),
            # and off-diagonal coherences encode the phase information.
            #
            # Strategy: We purify the density matrix to extract dominant phase,
            # then verify the CZ phase condition φ₁₁ = 2φ₀₁ - π
            
            rho_01 = results["01"]
            rho_11 = results["11"]
            rho_00 = results["00"]
            
            # Population in target state (how much stayed in computational subspace)
            proj_01 = psi_01 * psi_01.dag()
            proj_11 = psi_11 * psi_11.dag()
            proj_00 = psi_00 * psi_00.dag()
            
            pop_01 = np.real((proj_01 * rho_01).tr())
            pop_11 = np.real((proj_11 * rho_11).tr())
            pop_00 = np.real((proj_00 * rho_00).tr())
            
            amp_01 = np.sqrt(max(0, pop_01))
            amp_11 = np.sqrt(max(0, pop_11))
            
            # =====================================================================
            # MIXED STATE PHASE EXTRACTION (CRITICAL FOR CORRECT FIDELITY)
            # =====================================================================
            # For density matrices, we MUST extract the controlled phase to properly
            # assess gate quality. Without this, protocols that produce WRONG phase
            # (like JP bang-bang) would incorrectly appear to have high fidelity.
            #
            # Strategy: Extract dominant eigenvector from each output density matrix.
            # The dominant eigenvector approximates the "pure state part" of the
            # output and carries the phase information needed for CZ verification.
            #
            # The controlled phase φ_controlled = φ₁₁ - φ₀₁ - φ₁₀ + φ₀₀ must equal
            # ±π for a valid CZ gate. Deviations are penalized.
            # =====================================================================
            
            def get_dominant_phase(rho, target_ket):
                """Extract phase from dominant eigenvector of density matrix."""
                try:
                    evals, evecs = rho.eigenstates()
                    idx_max = np.argmax(evals)
                    dominant_vec = evecs[idx_max]
                    overlap = target_ket.dag() * dominant_vec
                    val = overlap.full()[0, 0] if hasattr(overlap, 'full') else complex(overlap)
                    return np.angle(val), np.abs(val)
                except Exception:
                    return 0.0, 0.0
            
            # Get phases from all four outputs
            rho_10 = results["10"]
            phi_00_dm, _ = get_dominant_phase(rho_00, psi_00)
            phi_01_dm, _ = get_dominant_phase(rho_01, psi_01)
            phi_10_dm, _ = get_dominant_phase(rho_10, psi_10)
            phi_11_dm, _ = get_dominant_phase(rho_11, psi_11)  # vs +|11⟩
            
            # Controlled phase calculation: φ₁₁ - φ₀₁ - φ₁₀ + φ₀₀
            controlled_phase_dm = phi_11_dm - phi_01_dm - phi_10_dm + phi_00_dm
            controlled_phase_dm = (controlled_phase_dm + np.pi) % (2*np.pi) - np.pi
            
            # Phase error from target (±π)
            phase_error_from_pi_dm = min(abs(controlled_phase_dm - np.pi), 
                                         abs(controlled_phase_dm + np.pi))
            
            # Phase fidelity factor
            cz_phase_fidelity_dm = np.cos(phase_error_from_pi_dm / 2)**2
            
            phase_info = {
                'phi_01_rad': phi_01_dm,
                'phi_01_deg': np.degrees(phi_01_dm),
                'phi_11_rad': phi_11_dm,
                'phi_11_deg': np.degrees(phi_11_dm),
                'expected_phi_11_rad': -np.pi,
                'controlled_phase_rad': controlled_phase_dm,
                'controlled_phase_deg': np.degrees(controlled_phase_dm),
                'phase_error_from_pi_rad': phase_error_from_pi_dm,
                'phase_error_from_pi_deg': np.degrees(phase_error_from_pi_dm),
                'cz_phase_fidelity': cz_phase_fidelity_dm,
                'amp_01': amp_01,
                'amp_11': amp_11,
                'pop_00': pop_00,
                'pop_01': pop_01,
                'pop_11': pop_11,
                'is_mixed_state': True,
                'note': 'Phase extracted from dominant eigenvector - penalty applied for CZ condition',
            }
            
            # For density matrix case, use ideal CZ targets (no phase adjustment)
            # The fidelity calculation will properly handle the mixed state comparison
            targets = {
                "00": psi_00,
                "01": psi_01,
                "10": psi_10,
                "11": -psi_11,  # The π phase
            }
        else:
            # =====================================================================
            # Pure state case: Extract global phase from |00⟩ and |01⟩
            # =====================================================================
            # The CZ gate has freedom in global phase. We extract this from the
            # |01⟩ output and use it to define phase-adjusted targets.
            #
            # IMPORTANT: We do NOT use the LP-specific formula φ_11 = 2φ_01 - π
            # because it doesn't hold for all protocols. Instead, we:
            # 1. Extract global phase from |01⟩ (single-atom phase accumulation)
            # 2. Use -|11⟩ as the target for |11⟩ (this is protocol-independent)
            # 3. Apply the same global phase to all targets
            #
            # This correctly handles both LP and JP protocols.
            
            overlap_01 = psi_01.dag() * results["01"]
            if hasattr(overlap_01, 'full'):
                overlap_01 = overlap_01.full()[0, 0]
            elif not isinstance(overlap_01, (complex, float)):
                overlap_01 = complex(overlap_01)
            phi_01 = np.angle(overlap_01)
            amp_01 = np.abs(overlap_01)
            
            # For |11⟩, compute overlap with BOTH +|11⟩ and -|11⟩
            # The correct CZ gate produces -|11⟩
            overlap_11_plus = psi_11.dag() * results["11"]
            overlap_11_minus = (-psi_11).dag() * results["11"]
            if hasattr(overlap_11_plus, 'full'):
                overlap_11_plus = overlap_11_plus.full()[0, 0]
                overlap_11_minus = overlap_11_minus.full()[0, 0]
            elif not isinstance(overlap_11_plus, (complex, float)):
                overlap_11_plus = complex(overlap_11_plus)
                overlap_11_minus = complex(overlap_11_minus)
            
            # Use the overlap with -|11⟩ for fidelity (this is what CZ should produce)
            phi_11 = np.angle(overlap_11_minus)  # Phase relative to -|11⟩
            amp_11 = np.abs(overlap_11_minus)     # Amplitude = fidelity^0.5
            
            # For diagnostic: also record phase relative to +|11⟩
            phi_11_plus = np.angle(overlap_11_plus)
            
            # The "phase error" is how far we are from -|11⟩
            # This is simply captured by amp_11 - if amp_11 ~ 1, state is close to -|11⟩
            phase_error = np.arccos(np.clip(amp_11, 0, 1))  # Small angle = good
            
            phase_info = {
                'phi_01_rad': phi_01,
                'phi_01_deg': np.degrees(phi_01),
                'phi_11_rad': phi_11,  # Phase relative to -|11⟩
                'phi_11_deg': np.degrees(phi_11),
                'phi_11_plus_rad': phi_11_plus,  # Phase relative to +|11⟩
                'phi_11_plus_deg': np.degrees(phi_11_plus),
                'phase_error_rad': phase_error,
                'phase_error_deg': np.degrees(phase_error),
                'amp_01': amp_01,
                'amp_11': amp_11,  # Overlap with -|11⟩
                'is_mixed_state': False,
            }
            
            # =====================================================================
            # PROTOCOL-INDEPENDENT TARGETS
            # =====================================================================
            # Use global phase from |01⟩ to align all targets, but always use -|11⟩
            # for the |11⟩ target. This works for both LP and JP protocols.
            #
            # Key insight: The CZ gate action is:
            #   |00⟩ → e^{iφ₀₀}|00⟩
            #   |01⟩ → e^{iφ₀₁}|01⟩
            #   |10⟩ → e^{iφ₁₀}|10⟩
            #   |11⟩ → e^{iφ₁₁}(-|11⟩)  ← Note the minus sign!
            #
            # The fidelity against -|11⟩ already captures whether the gate is correct.
            # We only need global phase adjustment for |00⟩, |01⟩, |10⟩.
            
            targets = {
                "00": psi_00,  # No phase adjustment needed (reference)
                "01": np.exp(1j * phi_01) * psi_01,
                "10": np.exp(1j * phi_01) * psi_10,  # Same as |01⟩ by symmetry
                "11": np.exp(1j * phi_11) * (-psi_11),  # Target is -|11⟩ with extracted phase
            }
    else:
        # Strict targets (no phase extraction)
        targets = {
            "00": psi_00,
            "01": psi_01,
            "10": psi_10,
            "11": -psi_11,  # The π phase!
        }
    
    # Compute per-state fidelities
    fidelities = {}
    for label in results:
        fidelities[label] = compute_state_fidelity(results[label], targets[label])
    
    # =========================================================================
    # CZ PHASE CONDITION CHECK
    # =========================================================================
    # The CZ gate requires a specific phase relationship:
    #   φ_controlled = φ₁₁ - φ₀₁ - φ₁₀ + φ₀₀ = π (mod 2π)
    #
    # This is the "controlled phase" that distinguishes CZ from identity.
    # The overlap |⟨-11|ψ⟩|² alone CANNOT check this because |⟨-11|+11⟩|² = 1.
    #
    # We compute the controlled phase and apply a penalty if it deviates from π.
    # This is critical for correctly assessing gate quality at weak blockade,
    # where the population stays in |11⟩ but the phase is wrong.
    
    # Apply CZ phase penalty for BOTH pure states AND mixed states
    # The controlled phase φ_controlled = φ₁₁ - φ₀₁ - φ₁₀ + φ₀₀ must equal ±π
    if phase_info:
        # For mixed states, cz_phase_fidelity was already computed above
        # For pure states, we need to compute it here
        if 'cz_phase_fidelity' not in phase_info:
            # Pure state case - extract phases
            def get_phase_amp(psi_target, psi_final):
                overlap = psi_target.dag() * psi_final
                val = overlap.full()[0,0] if hasattr(overlap, 'full') else complex(overlap)
                return np.angle(val), np.abs(val)
            
            phi_00, _ = get_phase_amp(psi_00, results["00"])
            phi_01_raw, _ = get_phase_amp(psi_01, results["01"])
            phi_10_raw, _ = get_phase_amp(psi_10, results["10"])
            phi_11_raw, _ = get_phase_amp(psi_11, results["11"])  # Against +|11⟩
            
            # Controlled phase: φ₁₁ - φ₀₁ - φ₁₀ + φ₀₀
            controlled_phase = phi_11_raw - phi_01_raw - phi_10_raw + phi_00
            controlled_phase = (controlled_phase + np.pi) % (2*np.pi) - np.pi
            
            phase_error_from_pi = min(abs(controlled_phase - np.pi), abs(controlled_phase + np.pi))
            cz_phase_fidelity = np.cos(phase_error_from_pi / 2)**2
            
            phase_info['controlled_phase_rad'] = controlled_phase
            phase_info['controlled_phase_deg'] = np.degrees(controlled_phase)
            phase_info['phase_error_from_pi_rad'] = phase_error_from_pi
            phase_info['phase_error_from_pi_deg'] = np.degrees(phase_error_from_pi)
            phase_info['cz_phase_fidelity'] = cz_phase_fidelity
        
        # Apply phase penalty to |11⟩ fidelity for ALL protocols (pure AND mixed)
        cz_phase_fidelity = phase_info.get('cz_phase_fidelity', 1.0)
        F11_population = fidelities["11"]
        F11_with_phase = F11_population * cz_phase_fidelity
        
        phase_info['F11_population'] = F11_population
        phase_info['F11_with_phase'] = F11_with_phase
        phase_info['cz_phase_condition_met'] = phase_info.get('phase_error_from_pi_rad', 0) < 0.2
        
        # Update the |11⟩ fidelity to include phase penalty
        fidelities["11"] = F11_with_phase
    
    avg_fidelity = np.mean(list(fidelities.values()))
    
    return fidelities, avg_fidelity, phase_info


# =============================================================================
# NOTE: Noise operators (build_all_noise_operators, build_decay_operators,
# build_dephasing_operators, build_loss_operators, build_scatter_operators)
# and compute_trap_dependent_noise are now in noise_models.py
# =============================================================================


# =============================================================================
# TIME EVOLUTION HELPERS
# =============================================================================

def evolve_state(
    H: Qobj,
    psi0: Qobj,
    tlist: np.ndarray,
    c_ops: List[Qobj] = None,
    options: dict = None,
) -> Qobj:
    """
    Evolve a quantum state under Hamiltonian H with optional noise.
    
    This is a thin wrapper around QuTiP's mesolve that handles
    common use cases for CZ gate simulation.
    
    Parameters
    ----------
    H : Qobj
        System Hamiltonian (time-independent)
    psi0 : Qobj
        Initial state (ket)
    tlist : array-like
        Time points for evolution
    c_ops : list of Qobj, optional
        Collapse operators for Lindblad evolution
    options : dict, optional
        Solver options {'atol', 'rtol', 'nsteps'}
        
    Returns
    -------
    Qobj
        Final state at tlist[-1]
        
    Notes
    -----
    For CZ gates, we typically evolve from t=0 to t=τ (pulse duration).
    The collapse operators encode all noise sources.
    """
    if c_ops is None:
        c_ops = []
    
    if options is None:
        options = {'atol': 1e-10, 'rtol': 1e-8, 'nsteps': 50000}
    
    result = mesolve(H, psi0, tlist, c_ops=c_ops, options=options)
    return result.states[-1]


def evolve_two_pulse(
    H1: Qobj,
    H2: Qobj,
    psi0: Qobj,
    tau_single: float,
    c_ops: List[Qobj] = None,
    n_steps: int = 100,
) -> Qobj:
    """
    Evolve through two-pulse Levine-Pichler protocol.
    
    The LP protocol applies two pulses:
        Pulse 1: H1 for time τ
        Pulse 2: H2 for time τ  (H2 includes phase shift ξ)
    
    Parameters
    ----------
    H1 : Qobj
        Hamiltonian for first pulse
    H2 : Qobj
        Hamiltonian for second pulse (with phase ξ)
    psi0 : Qobj
        Initial state
    tau_single : float
        Single pulse duration (total time = 2τ)
    c_ops : list of Qobj
        Collapse operators
    n_steps : int
        Time steps per pulse
        
    Returns
    -------
    Qobj
        Final state after both pulses
    """
    t1 = np.linspace(0, tau_single, n_steps)
    t2 = np.linspace(0, tau_single, n_steps)
    
    # First pulse
    psi_mid = evolve_state(H1, psi0, t1, c_ops)
    
    # Second pulse
    psi_final = evolve_state(H2, psi_mid, t2, c_ops)
    
    return psi_final


def evolve_two_pulse_lp(
    initial_states: Dict[str, Qobj],
    H1: Qobj,
    H2: Qobj,
    tau_single: float,
    c_ops: List[Qobj] = None,
) -> Dict[str, Qobj]:
    """
    Evolve all computational basis states through Levine-Pichler two-pulse sequence.
    
    This is a batch wrapper around evolve_two_pulse() that processes all 4
    computational basis states {|00⟩, |01⟩, |10⟩, |11⟩}.
    
    Parameters
    ----------
    initial_states : dict
        Initial states keyed by label ("00", "01", "10", "11")
    H1, H2 : Qobj
        First and second pulse Hamiltonians
    tau_single : float
        Single pulse duration (s)
    c_ops : list
        Collapse operators for noise
        
    Returns
    -------
    dict
        Final states after two-pulse evolution, keyed by input label
        
    See Also
    --------
    evolve_two_pulse : Single-state version of this function
    """
    final_states = {}
    for label, psi0 in initial_states.items():
        final_states[label] = evolve_two_pulse(H1, H2, psi0, tau_single, c_ops)
    return final_states
# =============================================================================
# BANG-BANG EVOLUTION (JANDURA-PUPILLO)
# =============================================================================
#
# BEGINNER'S GUIDE TO THE JANDURA-PUPILLO PROTOCOL
# ================================================
#
# If you're coming from the Levine-Pichler (LP) protocol, you might wonder:
# "Can we make the gate faster?" The JP protocol answers: YES, and here's HOW.
#
# THE PROBLEM: Making a CZ Gate as Fast as Possible
# -------------------------------------------------
# 
# Recall that a CZ gate needs |11⟩ to acquire a π phase relative to |01⟩/|10⟩.
# In LP, we do this with TWO pulses and a fixed detuning Δ/Ω ≈ 0.377.
# Total time: Ωτ_total ≈ 2 × 4.29 = 8.58 (in dimensionless units).
#
# But LP wasn't designed to be FAST - it was designed to be SIMPLE.
# JP asks: what's the absolute MINIMUM time to implement CZ?
#
# This is an "optimal control" problem from engineering:
#   - Given a system (two atoms + laser)
#   - Given constraints (max laser power → max Ω)
#   - Find the control sequence that achieves the goal in minimum time
#
# WHAT IS "BANG-BANG" CONTROL?
# ----------------------------
#
# "Bang-bang" is a term from control theory for a specific type of solution.
#
# Analogy: Imagine driving a car from A to B as fast as possible.
# You have constraints: max acceleration and max braking.
# The optimal strategy is NOT smooth driving - it's:
#   1. Floor the accelerator (BANG!) until halfway
#   2. Slam the brakes (BANG!) until you stop at B
#
# You spend all your time at the EXTREME values of your control.
# Never in between. That's "bang-bang" control.
#
# For the CZ gate, our "control knob" is the laser phase φ(t).
# The optimal solution jumps between extreme values:
#   φ(t) ∈ {-π/2, 0, +π/2}
# with sudden jumps (discontinuities) at specific "switching times".
#
# WHAT IS "LASER PHASE" PHYSICALLY?
# ---------------------------------
#
# A laser beam is an electromagnetic wave. Its electric field oscillates:
#
#   E(t) = E₀ cos(ωt + φ)
#        = E₀ cos(ωt) cos(φ) - E₀ sin(ωt) sin(φ)
#
# where:
#   - ω is the laser frequency (~10¹⁵ Hz for optical light)
#   - φ is the PHASE - it shifts WHERE in the oscillation cycle we are
#
# Visual: Think of two people on swings. If they're "in phase" (φ = 0),
# they swing together. If one is "90° out of phase" (φ = π/2), one is
# at maximum height when the other passes through center.
#
#   φ = 0:      E(t) = E₀ cos(ωt)        starts at maximum
#   φ = π/2:    E(t) = E₀ cos(ωt + π/2) = -E₀ sin(ωt)   starts at zero
#   φ = π:      E(t) = E₀ cos(ωt + π)   = -E₀ cos(ωt)   starts at minimum
#
# HOW DOES PHASE ENTER QUANTUM MECHANICS?
# ---------------------------------------
#
# When the laser couples the atom's ground state |1⟩ to Rydberg |r⟩,
# the coupling strength is a COMPLEX number:
#
#   Ω_complex = |Ω| × e^{iφ}
#
# where |Ω| is the Rabi frequency magnitude and φ is the laser phase.
#
# In the Hamiltonian:
#
#   H_coupling = (Ω/2) e^{iφ} |1⟩⟨r| + (Ω/2) e^{-iφ} |r⟩⟨1|
#              = (Ω/2) [cos(φ)(|1⟩⟨r| + |r⟩⟨1|) + i·sin(φ)(|1⟩⟨r| - |r⟩⟨1|)]
#              = (Ω/2) [cos(φ) σₓ + sin(φ) σᵧ]   (in |1⟩,|r⟩ subspace)
#
# So the phase φ determines the AXIS of rotation on the Bloch sphere:
#   - φ = 0:    Rotation around X-axis
#   - φ = π/2:  Rotation around Y-axis
#   - φ = π:    Rotation around -X-axis
#
# This is why phase control gives us control over the quantum evolution!
#
# HOW DO WE PHYSICALLY CONTROL LASER PHASE?
# -----------------------------------------
#
# Method 1: ACOUSTO-OPTIC MODULATOR (AOM)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# An AOM uses sound waves in a crystal to diffract light:
#
#   [Laser beam] → [Crystal with sound wave] → [Diffracted beam]
#
# The sound wave is generated by an RF signal. The PHASE of the RF
# directly sets the phase of the diffracted light!
#
#   RF signal: V(t) = V₀ cos(ω_RF·t + φ_RF)
#   Output light phase: φ_light = φ_RF
#
# To change φ, we just change φ_RF in our signal generator.
# Response time: ~10-100 nanoseconds (limited by sound speed in crystal)
#
# Method 2: ELECTRO-OPTIC MODULATOR (EOM)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# An EOM uses voltage to change the refractive index of a crystal:
#
#   [Laser] → [Crystal with voltage V] → [Phase-shifted light]
#
# The optical path length (and thus phase) depends on voltage:
#
#   φ_out = φ_in + (π/V_π) × V
#
# where V_π is the "half-wave voltage" (~1-10 kV for bulk, ~1-5 V for fiber).
#
# To change φ by π/2, we apply voltage V = V_π/2.
# Response time: ~picoseconds to nanoseconds (MUCH faster than AOM!)
#
# Method 3: ARBITRARY WAVEFORM GENERATOR (AWG)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Modern experiments use digital AWGs that directly synthesize:
#
#   V(t) = A(t) × cos(ω·t + φ(t))
#
# Both amplitude A(t) and phase φ(t) are programmed point-by-point.
# The AWG drives an AOM or EOM to transfer this control to the laser.
#
# PUTTING IT TOGETHER FOR BANG-BANG CONTROL
# -----------------------------------------
#
# In a JP protocol experiment:
#
# 1. Before the experiment: Program the AWG with the phase sequence
#    φ(t) = {φ₀ for t < t₁, φ₁ for t₁ < t < t₂, ...}
#
# 2. Trigger the pulse: AWG outputs the programmed waveform
#
# 3. AOM/EOM converts electrical phase → optical phase
#
# 4. Atom sees laser with time-dependent phase φ(t)
#
# The "discontinuous jumps" in φ are actually very fast (~ns) transitions,
# but since the gate takes ~μs, they appear instantaneous to the atom.
#
# =============================================================================
# WHAT IS DETUNING AND WHAT DOES IT DO?
# =============================================================================
#
# You've seen "detuning Δ" mentioned many times. Let's understand what it IS
# and what EFFECT it has on the atom.
#
# DEFINITION: WHAT IS DETUNING?
# -----------------------------
#
# When a laser tries to excite an atom from |1⟩ to |r⟩, the atom has a
# natural transition frequency ω_atom (the energy difference E_r - E_1
# divided by ℏ).
#
# Detuning is HOW FAR OFF the laser frequency is from this resonance:
#
#   Δ = ω_laser - ω_atom
#
#   Δ > 0 (blue detuned):  Laser frequency TOO HIGH
#   Δ = 0 (on resonance):  Laser frequency EXACTLY matches atom
#   Δ < 0 (red detuned):   Laser frequency TOO LOW
#
# Analogy: Imagine pushing a child on a swing. The swing has a natural
# frequency. If you push at exactly that frequency (Δ = 0), you transfer
# energy very efficiently. If you push too fast or too slow (Δ ≠ 0),
# some of your effort is "wasted."
#
# THE EFFECT OF DETUNING: THREE KEY CONSEQUENCES
# ----------------------------------------------
#
# EFFECT 1: Incomplete Population Transfer
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# With Δ = 0 (on resonance), a pulse of area Ωτ = π completely transfers
# population from |1⟩ to |r⟩:
#
#   |1⟩ ──[π pulse]──> |r⟩   (100% transfer)
#
# With Δ ≠ 0, the transfer is INCOMPLETE. The maximum excitation is:
#
#   P_max = Ω² / (Ω² + Δ²) = Ω² / Ω_gen²
#
# where Ω_gen = √(Ω² + Δ²) is the "generalized Rabi frequency."
#
# Example: If Δ = Ω, then P_max = 1/2. You can NEVER fully excite the atom!
#
#   Δ = 0:    |1⟩ ──────> |r⟩     (complete transfer possible)
#   Δ = Ω:    |1⟩ ──────> 50% |r⟩ (maximum, then oscillates back)
#   Δ >> Ω:   |1⟩ ──────> ~0% |r⟩ (almost no excitation)
#
# EFFECT 2: Faster Oscillations
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The Rabi oscillation frequency is NOT Ω when detuned - it's faster!
#
#   Oscillation frequency = Ω_gen = √(Ω² + Δ²)
#
# Think of it this way: the atom oscillates between |1⟩ and a PARTIAL
# excitation of |r⟩, and it does so FASTER than the on-resonance case.
#
# This is actually useful! We can make gates faster by using detuning,
# even though we don't fully excite to |r⟩.
#
# EFFECT 3: Phase Accumulation (THE KEY FOR CZ GATES!)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This is the most important effect for CZ gates.
#
# When Δ ≠ 0, the state accumulates a PHASE during evolution, even if
# it returns to its starting population!
#
# The Hamiltonian with detuning is:
#
#   H = (Ω/2)|1⟩⟨r| + (Ω/2)|r⟩⟨1| - Δ|r⟩⟨r|
#     = [ 0    Ω/2 ]    (in |1⟩, |r⟩ basis)
#       [ Ω/2  -Δ  ]
#
# The eigenvalues (energy levels) are:
#
#   E± = (-Δ ± Ω_gen) / 2
#
# After time τ, each eigenstate picks up phase e^{-iE±τ}. When you
# recombine them, you get a NET PHASE on the final state.
#
# For a 2π pulse (full Rabi cycle back to |1⟩):
#
#   |1⟩ → e^{iφ}|1⟩   where φ depends on Δ, Ω, and τ
#
# This is the "dynamical phase" that appears in our CZ gate!
#
# VISUALIZING DETUNING: THE BLOCH SPHERE
# --------------------------------------
#
# On the Bloch sphere, the atom state is a point on a sphere:
#   - North pole = |1⟩
#   - South pole = |r⟩
#   - Equator = superpositions
#
# The Hamiltonian causes rotation around an axis:
#   - Ω drives rotation around X-axis (population transfer)
#   - Δ drives rotation around Z-axis (phase accumulation)
#
# The TOTAL rotation axis depends on BOTH:
#
#          Z (phase)
#          ↑
#          |     ← Rotation axis
#          |   ↗
#          | θ/
#          |/
#    ──────+─────→ X (population)
#
#   tan(θ) = Ω / Δ
#
#   Δ = 0:  Rotation purely around X (complete population oscillation)
#   Δ >> Ω: Rotation almost around Z (mostly phase, little population change)
#   Δ = Ω:  Rotation at 45°, mix of both effects
#
# WHY DOES LP USE Δ/Ω ≈ 0.377?
# ----------------------------
#
# The Levine-Pichler protocol needs |1⟩ to:
#   1. Go to |r⟩ and back (population returns to |1⟩)
#   2. Accumulate a specific phase φ in the process
#
# The ratio Δ/Ω ≈ 0.377 is carefully chosen so that after the two-pulse
# sequence, the phases work out correctly for a CZ gate.
#
# Different ratios give different phases. The LP paper found 0.377 gives
# the right phase relationship between |01⟩ and |11⟩ states.
#
# WHY DOES JP USE Δ = 0?
# ----------------------
#
# The JP protocol achieves the same phase accumulation through a
# DIFFERENT mechanism: rapid phase modulation φ(t).
#
# A changing phase acts LIKE a detuning:
#
#   Effective Δ = -dφ/dt
#
# So JP gets the "detuning effect" (phase accumulation) without actual
# detuning, by modulating the laser phase instead. This turns out to be
# faster but requires more precise control.
#
# SUMMARY: DETUNING EFFECTS
# -------------------------
#
#   Effect              Δ = 0 (resonant)    Δ ≠ 0 (detuned)
#   ─────────────────   ────────────────    ────────────────
#   Max excitation      100%                Ω²/(Ω²+Δ²) < 100%
#   Oscillation rate    Ω                   √(Ω²+Δ²) > Ω
#   Phase accumulation  None (during Ω=0)   Yes! (key for CZ)
#   Bloch sphere axis   X-axis              Tilted toward Z
#
# =============================================================================
# WAIT - IF DETUNED IS FASTER, WHY IS JP (Δ=0) THE FASTER PROTOCOL?
# =============================================================================
#
#
# The oscillation rate √(Ω²+Δ²) tells you how fast the atom oscillates
# between |1⟩ and |r⟩. But that's NOT the same as the GATE TIME!
#
# Gate time is determined by: "How long until we achieve the CZ phase condition?"
#
# =============================================================================
# UNDERSTANDING GATE TIMING: Ωτ, CYCLES, AND WHY NOT JUST 1/Ω?
# =============================================================================
#
# FIRST: What is Ωτ and why use it instead of just τ?
# ---------------------------------------------------
#
# Ωτ is a DIMENSIONLESS number - it's the "pulse area" or "rotation angle"
# in radians. Using Ωτ instead of τ makes the physics UNIVERSAL.
#
# Example: These are all the SAME gate, just with different Ω:
#
#   Ω = 2π × 1 MHz,  τ = 1 μs    →  Ωτ = 2π × 1
#   Ω = 2π × 10 MHz, τ = 0.1 μs  →  Ωτ = 2π × 1
#   Ω = 2π × 100 MHz, τ = 0.01 μs → Ωτ = 2π × 1
#
# The physics is determined by Ωτ, not by Ω or τ separately.
# Higher Ω = same gate but faster. That's why we quote Ωτ.
#
# WHAT IS A "FULL CYCLE" vs "PARTIAL CYCLE"?
# ------------------------------------------
#
# On-resonance (Δ = 0), a full Rabi cycle is Ωτ = 2π:
#
#   |1⟩ → |r⟩ → |1⟩  (population goes up then comes back)
#
#   Population in |r⟩:
#   1 ─┐    ╱╲    ╱╲
#     │   ╱  ╲  ╱  ╲
#     │  ╱    ╲╱    ╲
#   0 ─┴──────────────→ Ωτ
#     0    π    2π    3π
#          ↑     ↑
#        half  full
#        cycle cycle
#
# But with DETUNING (Δ ≠ 0), things are different:
#   - Oscillations are FASTER: period = 2π/Ω_gen where Ω_gen = √(Ω² + Δ²)
#   - Oscillations are INCOMPLETE: never reaches 100% in |r⟩
#   - Population returns to |1⟩ at different Ωτ values
#
# So "partial cycle" means: Ωτ < 2π, but the state has undergone a
# useful evolution that, when combined with a phase-shifted second
# pulse, achieves the CZ gate.
#
# WHY ISN'T THE GATE TIME JUST 2π/Ω (ONE FULL CYCLE)?
# ---------------------------------------------------
#
# Great question! For a simple π rotation (like a NOT gate), you'd
# use Ωτ = π (half cycle). For a 2π rotation (identity + phase),
# you'd use Ωτ = 2π (full cycle).
#
# But a CZ gate is MORE COMPLEX than a simple rotation!
#
# A CZ gate must satisfy MULTIPLE constraints simultaneously:
#
#   1. |00⟩: No evolution (neither atom couples to laser)  ✓ automatic
#   2. |01⟩: Atom returns to |1⟩ with phase φ               needs tuning
#   3. |10⟩: Same as |01⟩ by symmetry                       ✓ automatic
#   4. |11⟩: Both atoms return with phase 2φ - π            needs tuning
#
# The challenge: |01⟩ and |11⟩ have DIFFERENT Hamiltonians!
#
#   |01⟩: Single atom, effective Rabi frequency = Ω
#   |11⟩: Both atoms, but blockade modifies dynamics
#         The symmetric state couples with Ω_eff = √2 · Ω
#
# These evolve at DIFFERENT RATES. A single "full cycle" of one
# is NOT a full cycle of the other!
#
# LP's solution: Use Ωτ ≈ 4.29 per pulse (NOT 2π!) because that's
# the value where BOTH |01⟩ AND |11⟩ satisfy their requirements
# (with the help of the phase shift ξ between pulses).
#
# IS τ THE SAME FOR BOTH LP PULSES?
# ---------------------------------
#
# YES! In the standard LP protocol, both pulses have the SAME duration:
#
#   Pulse 1: duration τ, phase 0
#   Pulse 2: duration τ, phase ξ (shifted)
#
# The only difference is the PHASE, not the duration.
# Total gate time = 2τ.
#
# IS τ THE SAME FOR LP AND JP?
# ----------------------------
#
# NO! Different protocols have different optimal timings:
#
#   LP (Levine-Pichler):
#     - Two pulses, each with Ωτ_single ≈ 4.29
#     - Total: Ωτ_total = 2 × 4.29 ≈ 8.58
#     - Physical time: τ_total = 8.58/Ω
#
#   JP (Jandura-Pupillo):
#     - One pulse with phase jumps
#     - Total: Ωτ_total ≈ 6.2    (2π)
#     - Physical time: τ_total = 6.2/Ω
#
# JP is ~28% faster because bang-bang control is more efficient.
#
# NUMERICAL EXAMPLE
# -----------------
#
# Let's say Ω = 2π × 5 MHz = 31.4 × 10⁶ rad/s
#
#   LP gate time: τ_total = 8.58 / (31.4 × 10⁶) = 0.27 μs
#   JP gate time: τ_total = 6.2 / (31.4 × 10⁶) = 0.20 μs
#
# =============================================================================
# WHAT DETERMINES GATE TIME?
# --------------------------
#
# For a CZ gate, we need:
#   1. Population to return to computational basis (no leftover |r⟩)
#   2. |11⟩ to have π more phase than |01⟩
#
# These are TWO separate requirements, and meeting BOTH takes time.
#
# WHY LP IS SLOWER (despite faster oscillations):
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# LP uses Δ/Ω ≈ 0.377, giving Ω_gen = √(1 + 0.377²)Ω ≈ 1.07Ω.
# That's only 7% faster oscillation - almost negligible!
#
# But here's the real issue: LP needs TWO FULL PULSES to work.
#
#   Pulse 1: Ωτ ≈ 4.29  (takes atom through ~0.68 of a 2π cycle)
#   Pulse 2: Ωτ ≈ 4.29  (with phase shift ξ to complete CZ)
#   ─────────────────────
#   Total:   Ωτ ≈ 8.58
#
# LP's constraint: The phase shift ξ between pulses can only take
# discrete values to ensure population returns to |1⟩. This forces
# the two-pulse structure.
#
# WHY DOES LP NEED THE PHASE SHIFT ξ?
# -----------------------------------
#
# On the Bloch sphere, the detuned pulse rotates the state around a
# TILTED axis. If you rotate long enough (a full 2π around the tilted
# axis), the population WILL return to |1⟩.
#
# So why not just use one long pulse?
#
# The issue is that |01⟩ and |11⟩ have DIFFERENT Hamiltonians:
#
#   |01⟩: Only ONE atom couples to laser → Ω_eff = Ω
#   |11⟩: BOTH atoms try to couple, but BLOCKADE interferes
#         The symmetric state |ψ_sym⟩ = (|1r⟩ + |r1⟩)/√2 couples
#         with Ω_eff = √2 · Ω (enhanced!), but |rr⟩ is blocked
#
# These two cases oscillate at DIFFERENT rates and accumulate
# DIFFERENT phases. For a CZ gate, we need a SPECIFIC relationship:
#
#   Phase of |11⟩ = 2 × (Phase of |01⟩) - π
#
# With a SINGLE pulse, you have only ONE adjustable parameter: τ.
# But you need to satisfy TWO conditions:
#   1. Population returns to computational basis (no |r⟩ left)
#   2. The phase relationship above is satisfied
#
# SOLUTION: Phase shift changes the rotation axis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The phase shift ξ rotates the AXIS of the second pulse:
#
#   Pulse 1: Rotation around axis n₁ = (cos(0), sin(0), ...) 
#   Pulse 2: Rotation around axis n₂ = (cos(ξ), sin(ξ), ...)
#
# By choosing ξ correctly, the second rotation "undoes" the
# population transfer while ADDING to the phase!
#
# Think of it like a ECHO sequence in NMR:
#   - First pulse rotates state one way
#   - Phase-shifted second pulse rotates it back
#   - But the phases ADD rather than cancel
#
# VISUAL: TWO ROTATIONS ON THE BLOCH SPHERE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#         Z (|1⟩)
#         ↑
#         |    ← After pulse 1: state moved HERE
#         | ↙
#    ─────●─────→ X          Pulse 1 rotates around axis₁
#        ↖|
#         |← After pulse 2: state back to |1⟩ (with phase!)
#         
# The KEY: By rotating around a DIFFERENT axis (phase-shifted),
# pulse 2 brings the state back to the Z-axis (|1⟩) while the
# TOTAL accumulated phase is controlled by ξ.
#
# WHY SPECIFIC "MAGIC" VALUES?
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Not any ξ works! We need:
#   1. Population returns exactly to |1⟩ (no leftover |r⟩)
#   2. The phase accumulated gives the CZ condition
#
# For |01⟩ (single atom), we need |01⟩ → e^{iφ}|01⟩.
# For |11⟩ (blocked pair), we need |11⟩ → e^{i(2φ-π)}|11⟩.
#
# The math works out that with Δ/Ω = 0.377, the phase shift must be:
#
#   ξ = exp(i × some_angle)  where some_angle ≈ 3.90 rad
#
# This is computed from the eigenvalue structure of the Hamiltonian.
# See protocols.py → compute_phase_shift_xi() for the calculation.
#
# INTUITION: WHY TWO PULSES, NOT ONE?
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Can't we just do ONE long pulse? Let's see what happens:
#
# Single pulse of duration 2τ:
#   - State rotates around ONE axis continuously
#   - Eventually returns to |1⟩ (at special times)
#   - BUT: the phases for |01⟩ and |11⟩ don't satisfy CZ condition!
#
# The problem is that |01⟩ and |11⟩ have DIFFERENT effective
# Hamiltonians (because of blockade), so they evolve at different
# rates. A single pulse can't satisfy BOTH requirements:
#   - |01⟩ returns to |01⟩ with phase φ
#   - |11⟩ returns to |11⟩ with phase 2φ - π
#
# The phase shift ξ gives us an EXTRA degree of freedom to tune.
# It lets us independently control the phase relationship.
#
# ANALOGY: Calibrating a balance scale
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#   - Single pulse: You have one knob (time τ)
#   - Two equations to satisfy (|01⟩ phase AND |11⟩ phase)
#   - One knob, two requirements = usually impossible!
#
#   - Two pulses with phase shift: You have TWO knobs (τ and ξ)
#   - Two equations to satisfy = solvable!
#
# That's why LP needs the two-pulse structure with phase shift.
#
# WHY JP IS FASTER (despite Δ=0):
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# JP doesn't rely on detuning for phase accumulation. Instead, it uses
# RAPID PHASE MODULATION to accumulate phase more efficiently.
#
# Key insight: JP's phase jumps act like "instantaneous kicks" that
# add phase WITHOUT waiting for slow oscillations to complete.
#
# Think of it this way:
#   - LP: "Let me slowly detune and wait for phase to build up..."
#   - JP: "BANG! Here's a chunk of phase. BANG! Here's more."
#
# The bang-bang approach is more DIRECT. It was mathematically proven
# (using Pontryagin's Maximum Principle) to be the TIME-OPTIMAL solution.
#
#   JP Total: Ωτ ≈ 6.2  (28% faster than LP!)
#
# ANALOGY: TWO WAYS TO CLIMB A HILL
# ---------------------------------
#
#   LP approach: Walk up a gentle slope (detuning = gentle tilt)
#                Takes longer but easier path.
#
#   JP approach: Take the stairs (phase jumps = discrete steps)
#                More abrupt but faster overall.
#
# The stairs (JP) get you to the top faster even though each step
# is "harder" (requires precise timing).
#
# BOTTOM LINE:
# ~~~~~~~~~~~~
# "Faster oscillation" ≠ "Faster gate"
#
# Gate speed depends on how EFFICIENTLY you can satisfy BOTH:
#   - Return population to |1⟩
#   - Accumulate the right phase
#
# JP's bang-bang control does both more efficiently than LP's
# smooth detuned approach, despite using Δ=0.
#
# =============================================================================
#
# WHY DOES PHASE CONTROL THE GATE?
# ---------------------------------
#
# Recall the laser-atom coupling Hamiltonian:
#
#   H = (Ω/2) e^{iφ} |1⟩⟨r| + h.c. + (detuning terms)
#
# The phase φ appears in the coupling! Changing φ rotates the "direction"
# of the Rabi oscillation on the Bloch sphere.
#
# Here's the key insight: a CHANGING phase φ(t) acts like an effective
# detuning! If φ changes at rate dφ/dt, it's equivalent to adding
# extra detuning Δ_eff = -dφ/dt.
#
# Think of it like a rotating reference frame:
#   - Constant φ: You're in a fixed frame, see the "true" detuning Δ
#   - Changing φ: You're in a rotating frame, see shifted detuning
#
# For bang-bang control:
#   - φ = constant during each segment → Δ_eff = 0 in that segment
#   - φ JUMPS at switching times → infinite instantaneous Δ_eff (a "kick")
#
# THE JP PULSE SEQUENCE
# ---------------------
#
# A typical JP protocol looks like this (time on x-axis):
#
#    φ(t):
#    +π/2 ─────┐              ┌─────
#              │              │
#       0      └──────────────┘
#              
#    -π/2 
#         ─────┬──────────────┬─────
#         0    t₁            t₂    τ_total
#              ↑              ↑
#         switching      switching
#           time           time
#
# The phase is CONSTANT within each segment (so we can use time-independent
# Hamiltonians), but JUMPS discontinuously at the switching times.
#
#
# WHY DOES THIS PRODUCE A CZ GATE?
# --------------------------------
#
# The math is complex, but the intuition is:
#
# 1. For |01⟩ (one atom in |1⟩, one in |0⟩):
#    - Only one atom couples to the laser
#    - The phase sequence drives it through a specific trajectory
#    - It returns to |1⟩ with some dynamical phase φ_single
#
# 2. For |11⟩ (both atoms in |1⟩):
#    - Both atoms want to couple to the laser
#    - But Rydberg blockade prevents |rr⟩ (double excitation)
#    - The system evolves in the BLOCKED subspace
#    - The phase sequence is carefully designed so that |11⟩ acquires
#      phase 2φ_single - π (twice single-atom phase MINUS π)
#
# The JP switching times and phases were found by numerical optimization
# (specifically, Pontryagin's Maximum Principle from optimal control theory).
# They're not intuitive - they're the OUTPUT of an optimization that
# minimizes gate time while achieving the CZ phase condition.
#
# COMPARISON: LP vs JP
# --------------------
#
#   Property              LP (Levine-Pichler)    JP (Jandura-Pupillo)
#   ────────────────────  ────────────────────   ────────────────────
#   Number of pulses      2                      1 (with phase jumps)
#   Control parameter     Pulse phases           Time-varying phase
#   Detuning Δ            Fixed (Δ/Ω ≈ 0.377)    Zero (Δ = 0)!
#   Gate time (Ωτ)        ~8.6                   ~6.2 (28% faster!)
#   Complexity            Simple                 Requires precise timing
#   Sensitivity           Moderate               More sensitive to errors
#
# The JP protocol achieves Δ = 0 by encoding the detuning effect into
# the phase modulation itself. This is elegant but requires more
# precise control hardware.
#
# =============================================================================


# =============================================================================
# SMOOTH SINUSOIDAL JP PROTOCOL (DARK STATE EVOLUTION)
# =============================================================================
#
# BLUVSTEIN/EVERED HIGH-FIDELITY CZ GATE
# ======================================
#
# This implements the smooth sinusoidal CZ gate that achieved 99.5% fidelity
# in the Harvard neutral atom quantum computer (Evered et al., Nature 2023).
#
# KEY PHYSICS INSIGHT - DARK STATE SUPPRESSES SCATTERING:
# -------------------------------------------------------
# For two atoms driven to Rydberg states, there are two collective states:
#   - Bright state |B⟩ = (|r1⟩ + |1r⟩)/√2 - couples to |11⟩ and intermediate |e⟩
#   - Dark state |D⟩ = (|r1⟩ - |1r⟩)/√2 - DECOUPLED from intermediate |e⟩
#
# By choosing OPPOSITE SIGNS for intermediate-state detuning (Δₑ) and 
# two-photon detuning (δ), population is preferentially transferred to |D⟩,
# which dramatically suppresses intermediate-state scattering.
#
# With Δₑ > 0 (blue-detuned from 6P₃/₂), we need δ < 0 (red-detuned two-photon).
#
# THE SMOOTH PROTOCOL:
# --------------------
# Instead of bang-bang phase jumps, the phase varies smoothly:
#
#   φ(t) = A·cos(ω_mod·t - φ_offset) + δ₀·t
#
# Where:
#   - A ≈ π/2: phase amplitude (same as bang-bang extreme values)
#   - ω_mod ≈ Ω: modulation frequency (resonant condition)
#   - φ_offset: phase offset (calibration parameter)
#   - δ₀: two-photon detuning (CRITICAL for dark state!)
#
# The δ₀·t term is the KEY INSIGHT: the phase slope corresponds to a 
# constant two-photon detuning Δ = δ₀, which must be NEGATIVE when Δₑ > 0.
#
# IMPROVEMENTS OVER BANG-BANG:
# ----------------------------
# 1. Smooth → reduced spectral leakage and pulse distortions
# 2. Dark state physics → suppressed intermediate-state scattering
# 3. Single pulse → no turn-on/turn-off transients between pulses
# 4. ~10% faster than Levine-Pichler two-pulse protocol
#
# FIDELITY ACHIEVED: 99.5% (from 97.5% with previous methods)
#
# Reference: Evered et al., Nature 622, 268-272 (2023)
#            Bluvstein PhD Thesis (Harvard, 2024), Section 2.5
# =============================================================================


def evolve_smooth_sinusoidal_jp(
    initial_states: Dict[str, Qobj],
    Omega: float,
    V: float,
    tau_total: float,
    hs,  # HilbertSpace object
    c_ops: List[Qobj] = None,
    A: float = None,
    omega_mod: float = None,
    phi_offset: float = 0.0,
    delta_over_omega: float = None,
    n_steps: int = 200,
    delta_zeeman: float = 0.0,
    delta_stark: float = 0.0,
    trap_laser_on: bool = True,
    intermediate_detuning_sign: str = "positive",
    verbose: bool = False,
) -> Tuple[Dict[str, Qobj], dict]:
    """
    Evolve quantum states through smooth sinusoidal phase-modulated CZ gate.
    
    ✓ VALIDATED: Achieves >99.9% fidelity for V/Ω in [10, 200]
    ============================================================
    
    From Bluvstein PhD Thesis (2024):
        "φ(t) = A cos(ωt − ϕ) + δ₀t"
    
    This implements the smooth sinusoidal CZ gate that achieved 99.5% fidelity
    in the Harvard neutral atom quantum computer (Evered et al., Nature 2023).
    
    The smooth JP protocol uses sinusoidal phase modulation with:
    - A ≈ 0.31π: phase amplitude (~56°)
    - ω ≈ 1.24Ω: modulation frequency (close to Rabi frequency)
    - ϕ ≈ 4.7 rad: phase offset (calibration parameter)
    - δ₀ ≈ 0.02Ω: two-photon detuning (from phase slope term)
    
    Parameters
    ----------
    initial_states : dict
        Dictionary of initial states: {"00": ψ₀₀, "01": ψ₀₁, "10": ψ₁₀, "11": ψ₁₁}
    Omega : float
        Two-photon Rabi frequency (rad/s).
    V : float
        Rydberg blockade interaction strength (rad/s). Should have V >> Ω.
    tau_total : float
        Total gate duration in seconds. Typically Ω·τ ≈ 1.8π (time-optimal).
    hs : HilbertSpace
        Hilbert space object from hamiltonians.py.
    c_ops : list of Qobj, optional
        Lindblad collapse operators. None for ideal evolution.
    A : float, optional
        Phase modulation amplitude (rad). Default: π/2.
    omega_mod : float, optional
        Modulation frequency (rad/s). Default: Ω (resonant condition).
    phi_offset : float, optional
        Phase offset in modulation (rad). Default: 0. Calibration parameter.
    delta_over_omega : float, optional
        Two-photon detuning ratio δ/Ω. Default: -0.08 (NEGATIVE for dark state
        when intermediate detuning is positive). CRITICAL FOR FIDELITY!
    n_steps : int, optional
        Number of time steps for evolution. Default: 200.
    delta_zeeman, delta_stark : float, optional
        Additional detuning shifts (rad/s).
    trap_laser_on : bool, optional
        Whether optical trap is on during gate.
    intermediate_detuning_sign : str, optional
        Sign of intermediate-state detuning: "positive" (blue-detuned, default)
        or "negative" (red-detuned). Used to validate dark state condition.
    verbose : bool, optional
        Print diagnostic information.
        
    Returns
    -------
    final_states : dict
        Final states after gate: {"00": ψ_final_00, ...}
    info : dict
        Diagnostic information including dark state parameters.
        
    Notes
    -----
    **DARK STATE PHYSICS (Critical for 99.5% fidelity):**
    
    The two-photon detuning δ (= Ω × delta_over_omega) MUST have OPPOSITE
    SIGN from the intermediate-state detuning Δₑ to maximize dark state
    population and suppress scattering:
    
        - Δₑ > 0 (blue-detuned intermediate) → δ < 0 (red-detuned two-photon)
        - Δₑ < 0 (red-detuned intermediate) → δ > 0 (blue-detuned two-photon)
    
    Default: delta_over_omega = -0.08 with Δₑ > 0 (Harvard configuration).
    
    **Calibration:**
    
    The parameters A, ω_mod, φ_offset, and δ can be individually scanned to
    maximize fidelity via randomized benchmarking (see Bluvstein Fig 2.14).
    
    Reference: Evered et al., Nature 622, 268-272 (2023)
               Bluvstein PhD Thesis (Harvard, 2024)
    """
    # ---------------------------------------------------------------------
    # SET DEFAULT PARAMETERS (Bluvstein-form validated values)
    # ---------------------------------------------------------------------
    # Validated parameters achieve >99.9% fidelity for V/Ω in [10, 200]
    # From Bluvstein: φ(t) = A·cos(ω·t - ϕ) + δ₀·t
    if A is None:
        A = 0.311 * np.pi  # Phase amplitude ≈ 56°
    
    if omega_mod is None:
        omega_mod = 1.242 * Omega  # ω_mod ≈ 1.24Ω (close to Ω as Bluvstein noted)
    
    if delta_over_omega is None:
        # Validated default: small positive value for Bluvstein-form gate
        # Note: For dark state physics with Δₑ > 0, use negative value
        delta_over_omega = 0.0205
    
    # Calculate two-photon detuning from ratio
    Delta = delta_over_omega * Omega
    
    # Calculate dimensionless parameters
    omega_tau = Omega * tau_total
    V_over_Omega = V / Omega
    
    # ---------------------------------------------------------------------
    # VALIDATE DARK STATE CONDITION (for scattering suppression)
    # ---------------------------------------------------------------------
    # Note: The validated smooth JP parameters work for CZ gate operation
    # regardless of dark state condition. Dark state physics provides
    # additional scattering suppression but is not required for basic CZ.
    dark_state_valid = False
    if intermediate_detuning_sign == "positive":
        # Δₑ > 0 requires δ < 0 for dark state
        dark_state_valid = (delta_over_omega < 0)
        expected_sign = "negative"
    else:
        # Δₑ < 0 requires δ > 0 for dark state
        dark_state_valid = (delta_over_omega > 0)
        expected_sign = "positive"
    
    if not dark_state_valid and delta_over_omega != 0:
        import warnings
        warnings.warn(
            f"Dark state condition violated! With {intermediate_detuning_sign} intermediate-state "
            f"detuning, two-photon detuning should be {expected_sign} but got "
            f"delta_over_omega = {delta_over_omega:+.4f}. This will increase scattering error.",
            UserWarning
        )
    
    if verbose:
        print(f"\n{'='*70}")
        print("BLUVSTEIN/EVERED SMOOTH SINUSOIDAL CZ GATE (DARK STATE)")
        print(f"{'='*70}")
        print(f"Rabi frequency:    Ω/(2π) = {Omega/(2*np.pi)/1e6:.2f} MHz")
        print(f"Blockade:          V/(2π) = {V/(2*np.pi)/1e6:.2f} MHz")
        print(f"V/Ω ratio:         {V_over_Omega:.1f}")
        print(f"Gate time:         τ = {tau_total*1e6:.3f} μs")
        print(f"Pulse area:        Ω·τ = {omega_tau:.4f} rad ({omega_tau/np.pi:.4f}π)")
        print(f"\n--- Phase Modulation φ(t) = A·cos(ω·t - φ) + δ₀·t ---")
        print(f"Phase amplitude:   A = {A:.4f} rad ({A/np.pi:.4f}π)")
        print(f"Modulation freq:   ω_mod/Ω = {omega_mod/Omega:.4f}")
        print(f"Phase offset:      φ_offset = {phi_offset:.4f} rad")
        print(f"\n--- Dark State Physics ---")
        print(f"Two-photon detuning: δ/Ω = {delta_over_omega:+.4f}")
        print(f"                     δ/(2π) = {Delta/(2*np.pi)/1e6:+.4f} MHz")
        print(f"Intermediate detuning sign: {intermediate_detuning_sign}")
        print(f"Dark state condition: {'✓ SATISFIED' if dark_state_valid else '✗ VIOLATED!'}")
        print(f"{'='*70}")
    
    # Check adiabaticity condition
    if V_over_Omega < 5:
        import warnings
        warnings.warn(
            f"V/Ω = {V_over_Omega:.1f} may be too weak for reliable CZ operation. "
            f"Recommend V/Ω > 10 for high-fidelity gates.",
            UserWarning
        )
    
    # ---------------------------------------------------------------------
    # DEFINE PHASE MODULATION FUNCTION (Bluvstein form)
    # ---------------------------------------------------------------------
    def phase_function(t):
        """
        Bluvstein-form phase modulation: φ(t) = A·cos(ω_mod·t - φ_offset) + δ₀·t
        
        The δ₀·t term creates a constant two-photon detuning via phase slope.
        This is mathematically equivalent to having a separate Δ term in
        the Hamiltonian, but the Bluvstein form makes calibration easier
        (all parameters in the phase profile).
        
        Note: We apply the δ₀·t term via the Delta parameter in the
        Hamiltonian for numerical stability.
        """
        return A * np.cos(omega_mod * t - phi_offset)
    
    # ---------------------------------------------------------------------
    # TIME-STEPPING EVOLUTION
    # ---------------------------------------------------------------------
    tlist_full = np.linspace(0, tau_total, n_steps + 1)
    dt = tau_total / n_steps
    
    final_states = {}
    phase_evolution = []
    
    options = {'atol': 1e-10, 'rtol': 1e-8, 'nsteps': 10000}
    
    for label, psi0 in initial_states.items():
        psi = psi0
        
        for i in range(n_steps):
            t_start = tlist_full[i]
            t_mid = t_start + dt/2
            
            phase = phase_function(t_mid)
            
            if label == "00" and i < 10:
                phase_evolution.append((t_mid * 1e6, np.degrees(phase)))
            
            # Build Hamiltonian with phase AND two-photon detuning
            H = build_phase_modulated_hamiltonian(
                Omega=Omega,
                V=V,
                phase=phase,
                hs=hs,
                Delta=Delta,  # CRITICAL: two-photon detuning for dark state!
                delta_zeeman=delta_zeeman,
                delta_stark=delta_stark,
                trap_laser_on=trap_laser_on,
            )
            
            tlist_step = np.array([0, dt])
            psi = evolve_state(H, psi, tlist_step, c_ops, options)
        
        final_states[label] = psi
    
    info = {
        'protocol': 'smooth_sinusoidal_jp',
        'A': A,
        'A_over_pi': A / np.pi,
        'omega_mod': omega_mod,
        'omega_mod_over_Omega': omega_mod / Omega,
        'phi_offset': phi_offset,
        'delta_over_omega': delta_over_omega,
        'Delta_Hz': Delta / (2 * np.pi),
        'omega_tau': omega_tau,
        'omega_tau_over_pi': omega_tau / np.pi,
        'tau_us': tau_total * 1e6,
        'V_over_Omega': V_over_Omega,
        'n_steps': n_steps,
        'dark_state_valid': dark_state_valid,
        'intermediate_detuning_sign': intermediate_detuning_sign,
        'phase_samples': phase_evolution[:10] if phase_evolution else [],
    }
    
    if verbose:
        print(f"\nEvolution complete!")
        print(f"Phase at t=0:   {np.degrees(phase_function(0)):.2f}°")
        print(f"Phase at t=τ/2: {np.degrees(phase_function(tau_total/2)):.2f}°")
        print(f"Phase at t=τ:   {np.degrees(phase_function(tau_total)):.2f}°")
    
    return final_states, info


# =============================================================================
# BANG-BANG JP EVOLUTION (PIECEWISE-CONSTANT PHASE CONTROL)
# =============================================================================
#
# JANDURA-PUPILLO TIME-OPTIMAL CZ GATE
# ======================================
#
# The JP bang-bang protocol uses a SINGLE continuous laser pulse where the
# phase φ(t) takes discrete values (typically ±π/2, 0) and switches
# instantaneously at optimized times. This is "bang-bang" optimal control:
# the control saturates its constraint |φ| ≤ π/2 almost everywhere.
#
# For a 5-segment gate with total pulse area Ωτ:
#
#   Phase φ(t):
#   π/2 ──┐           ┌── π/2
#         │           │
#     0 ──┤───┐   ┌───┤── 0
#         │   │   │   │
#  -π/2 ──┘   └───┘   └──
#         t₁  t₂  t₃  t₄        (dimensionless times Ωt)
#
# The Hamiltonian is piecewise-constant within each segment:
#   H_k = (Ω/2)(e^{iφ_k}|1⟩⟨r| + h.c.) + V|rr⟩⟨rr|
#
# No static detuning (Δ = 0) — the phase switching alone creates the
# controlled phase.
#
# Reference: Jandura & Pupillo, PRX Quantum 3, 010353 (2022)
# =============================================================================


def evolve_bangbang_jp(
    initial_states: Dict[str, Qobj],
    Omega: float,
    V: float,
    omega_tau: float,
    switching_times: List[float],
    phases: List[float],
    hs,  # HilbertSpace object
    c_ops: List[Qobj] = None,
    delta_zeeman: float = 0.0,
    delta_stark: float = 0.0,
    trap_laser_on: bool = True,
    n_steps_per_segment: int = 50,
    verbose: bool = False,
) -> Tuple[Dict[str, Qobj], dict]:
    """
    Evolve quantum states through bang-bang phase-modulated CZ gate.
    
    The JP bang-bang protocol applies a single pulse with piecewise-constant
    phase φ(t) that switches between discrete values at optimized times.
    
    Parameters
    ----------
    initial_states : dict
        Dictionary of initial states: {"00": ψ₀₀, "01": ψ₀₁, "10": ψ₁₀, "11": ψ₁₁}
    Omega : float
        Two-photon Rabi frequency (rad/s).
    V : float
        Rydberg blockade interaction strength (rad/s).
    omega_tau : float
        Total dimensionless pulse area Ω×τ.
    switching_times : list of float
        Dimensionless times (Ωt) where phase switches. For N segments,
        there are N-1 switching times. Must be sorted and within [0, omega_tau].
    phases : list of float
        Phase value (radians) for each segment. len(phases) = len(switching_times) + 1.
    hs : HilbertSpace
        Hilbert space object from hamiltonians.py.
    c_ops : list of Qobj, optional
        Lindblad collapse operators. None for ideal evolution.
    delta_zeeman : float, optional
        Differential Zeeman shift (rad/s). Default 0.
    delta_stark : float, optional
        Differential AC Stark shift (rad/s). Default 0.
    trap_laser_on : bool, optional
        Whether optical trap is on during gate.
    n_steps_per_segment : int, optional
        Number of time steps per phase segment for mesolve. Default: 50.
    verbose : bool, optional
        Print diagnostic information.
        
    Returns
    -------
    final_states : dict
        Final states after gate: {"00": ψ_final_00, ...}
    info : dict
        Diagnostic information.
    """
    tau_total = omega_tau / Omega
    V_over_Omega = V / Omega
    n_segments = len(phases)
    
    # Build segment boundaries in dimensionless time (Ωt)
    # switching_times are the interior boundaries; add 0 and omega_tau as endpoints
    boundaries_dimless = [0.0] + list(switching_times) + [omega_tau]
    
    # Convert to real time
    boundaries = [b / Omega for b in boundaries_dimless]
    
    if verbose:
        print(f"\n{'='*70}")
        print("JANDURA-PUPILLO BANG-BANG CZ GATE")
        print(f"{'='*70}")
        print(f"Rabi frequency:    Ω/(2π) = {Omega/(2*np.pi)/1e6:.2f} MHz")
        print(f"Blockade:          V/(2π) = {V/(2*np.pi)/1e6:.2f} MHz")
        print(f"V/Ω ratio:         {V_over_Omega:.1f}")
        print(f"Gate time:         τ = {tau_total*1e6:.3f} μs")
        print(f"Pulse area:        Ω·τ = {omega_tau:.4f} rad ({omega_tau/np.pi:.4f}π)")
        print(f"Segments:          {n_segments}")
        print(f"\n--- Phase Schedule ---")
        for i in range(n_segments):
            t_start = boundaries_dimless[i]
            t_end = boundaries_dimless[i+1]
            duration = t_end - t_start
            print(f"  Segment {i}: Ωt ∈ [{t_start:.4f}, {t_end:.4f}]  "
                  f"(ΔΩt = {duration:.4f})  φ = {phases[i]/np.pi:+.4f}π "
                  f"= {np.degrees(phases[i]):+.1f}°")
        print(f"{'='*70}")
    
    # Validate
    assert len(phases) == len(switching_times) + 1, (
        f"Need len(phases) = len(switching_times) + 1, "
        f"got {len(phases)} phases and {len(switching_times)} switching times"
    )
    
    options = {'atol': 1e-10, 'rtol': 1e-8, 'nsteps': 10000}
    
    final_states = {}
    
    for label, psi0 in initial_states.items():
        psi = psi0
        
        for seg_idx in range(n_segments):
            t_start = boundaries[seg_idx]
            t_end = boundaries[seg_idx + 1]
            dt_segment = t_end - t_start
            
            if dt_segment < 1e-18:
                continue  # Skip zero-duration segments
            
            phase = phases[seg_idx]
            
            # Build constant Hamiltonian for this segment
            # No static detuning (Delta=0) for bang-bang JP
            H = build_phase_modulated_hamiltonian(
                Omega=Omega,
                V=V,
                phase=phase,
                hs=hs,
                Delta=0.0,  # No two-photon detuning in bang-bang
                delta_zeeman=delta_zeeman,
                delta_stark=delta_stark,
                trap_laser_on=trap_laser_on,
            )
            
            # Evolve under constant H for the segment duration
            # Use multiple steps for accuracy with collapse operators
            n_sub = max(2, n_steps_per_segment)
            tlist_seg = np.linspace(0, dt_segment, n_sub + 1)
            psi = evolve_state(H, psi, tlist_seg, c_ops, options)
        
        final_states[label] = psi
    
    info = {
        'protocol': 'bangbang_jp',
        'omega_tau': omega_tau,
        'omega_tau_over_pi': omega_tau / np.pi,
        'tau_us': tau_total * 1e6,
        'V_over_Omega': V_over_Omega,
        'n_segments': n_segments,
        'switching_times': list(switching_times),
        'phases': list(phases),
        'phases_over_pi': [p / np.pi for p in phases],
    }
    
    if verbose:
        print(f"\nEvolution complete!")
    
    return final_states, info


# =============================================================================
# SHAPED PULSE EVOLUTION (LEVINE-PICHLER WITH PULSE SHAPING)
# =============================================================================
#
# BEGINNER'S GUIDE: PULSE SHAPES AND SPECTRAL LEAKAGE
# ====================================================
#
# Before diving into shaped pulses, we need to understand a fundamental
# concept from signal processing: the TIME-FREQUENCY TRADEOFF.
#
# WHAT IS A "SQUARE PULSE"?
# -------------------------
#
# A square pulse is the simplest laser pulse: turn ON suddenly, stay on
# for time τ, then turn OFF suddenly.
#
#   Amplitude Ω(t):
#   
#   Ω_max ─────────────────┐
#                          │
#     0  ─────┬────────────┴─────
#             0            τ     time
#
# Simple, right? But there's a hidden problem...
#
# THE FOURIER TRANSFORM: TIME ↔ FREQUENCY
# ----------------------------------------
#
# Any pulse in TIME can be decomposed into a sum of FREQUENCIES.
# This is the Fourier transform - a fundamental math tool.
#
# Key insight: The SHAPE of the pulse in time determines its
# FREQUENCY CONTENT.
#
# For a square pulse, the Fourier transform is a "sinc" function:
#
#   Frequency content:
#   
#        │    /\
#        │   /  \       Side lobes (LEAKAGE!)
#        │  /    \   /\    /\
#        │ /      \_/  \__/  \__
#   ─────┼──────────────────────→ frequency
#        │ ↑
#        main peak
#        (at laser freq)
#
# See those "side lobes"? That's SPECTRAL LEAKAGE!
#
# WHAT IS SPECTRAL LEAKAGE?
# -------------------------
#
# The laser is tuned to frequency ω_laser, targeting the |1⟩ → |r⟩
# transition. But the atom has OTHER transitions nearby:
#
#   Energy levels:        What we want:      What we DON'T want:
#   
#     |r⟩ ─────           |1⟩ → |r⟩         |1⟩ → |other⟩
#     |other⟩ ────        (our gate)        (ERRORS!)
#     |1⟩ ─────
#     |0⟩ ─────
#
# Spectral leakage means our pulse has frequency components that can
# ACCIDENTALLY excite these unwanted transitions!
#
# Example: If there's another Rydberg state |r'⟩ that's 10 MHz away,
# and our square pulse has side lobes at ±10 MHz, we'll partially
# excite |r'⟩ → LEAKAGE ERROR!
#
# WHY DO SQUARE PULSES HAVE WIDE SPECTRA?
# ---------------------------------------
#
# This is a fundamental physics principle: the TIME-BANDWIDTH PRODUCT.
#
#   Sharp edges in time ←→ Wide spectrum in frequency
#   Smooth edges in time ←→ Narrow spectrum in frequency
#
# A square pulse has INFINITELY sharp edges (instantaneous on/off).
# Mathematically, representing a sharp edge requires MANY frequencies.
#
# Think of it like sound:
#   - A pure tone (sine wave) has ONE frequency - sounds smooth
#   - A click (sharp pulse) has MANY frequencies - sounds harsh
#
# HOW DO SHAPED PULSES HELP?
# --------------------------
#
# Shaped pulses have SMOOTH turn-on and turn-off:
#
#   Square pulse:           Gaussian pulse:         Blackman pulse:
#   
#       ┌────────┐              ╱╲                    ╱‾‾╲
#       │        │             ╱  ╲                  ╱    ╲
#   ────┘        └────      ──╱    ╲──            ─╱      ╲─
#   
#   Sharp edges             Smooth (e^{-t²})      Very smooth
#   Wide spectrum           Narrower spectrum     Narrowest spectrum
#   More leakage            Less leakage          Least leakage
#
# By smoothing the edges, we concentrate the frequency content near
# the target frequency, reducing excitation of nearby transitions.
#
# THE TRADEOFF: SPEED vs PRECISION
# --------------------------------
#
# There's no free lunch! Shaped pulses are LONGER for the same effect.
#
# Why? The "area" under the pulse determines the rotation angle:
#
#   Rotation angle = ∫ Ω(t) dt = "pulse area"
#
# For a π pulse (180° rotation), we need area = π.
#
#   Square pulse:  Area = Ω_max × τ_square
#   Gaussian:      Area = Ω_max × τ_gauss × √(π/2) ≈ 1.25 × Ω_max × τ_gauss
#
# To get the SAME rotation with a Gaussian, we either:
#   1. Increase τ (longer pulse) - what we usually do
#   2. Increase Ω_max (more laser power) - often not possible
#
# Typical penalty: Gaussian takes ~20-40% longer than square for
# the same rotation, but has MUCH better frequency selectivity.
#
# WHEN TO USE SHAPED PULSES?
# --------------------------
#
# Use shaped pulses when:
#   - Nearby energy levels could cause leakage errors
#   - You need very high fidelity (>99.9%)
#   - You have time budget to spare
#
# Use square pulses when:
#   - Speed is critical and energy levels are well-separated
#   - Leakage errors are small compared to other noise
#   - Simpler control is preferred
#
# For Rydberg CZ gates, shaped pulses help because:
#   - Rydberg states have nearby Zeeman sublevels
#   - High fidelity requires avoiding ALL error sources
#   - Gate times (~1 μs) allow some overhead
#
# COMMON PULSE SHAPES
# -------------------
#
#   Shape       Formula                 Leakage    Length penalty
#   ─────────   ───────────────────     ────────   ──────────────
#   Square      Ω(t) = Ω₀              Worst       0% (baseline)
#   Gaussian    Ω(t) = Ω₀ e^{-t²/σ²}  Good        ~25%
#   Cosine      Ω(t) = Ω₀ sin²(πt/τ)  Better      ~33%
#   Blackman    (see formula)          Best        ~40%
#
# =============================================================================

def evolve_shaped_pulse(
    initial_states: Dict[str, Qobj],
    pulse_shape: str,
    Omega: float,
    Delta: float,
    V: float,
    xi: complex,
    tau_single: float,
    hs,  # HilbertSpace object
    c_ops: List[Qobj] = None,
    delta_zeeman: float = 0.0,
    delta_stark: float = 0.0,
    trap_laser_on: bool = True,
    drag_lambda: float = None,
    n_time_steps: int = 500,
    verbose: bool = False,
) -> Tuple[Dict[str, Qobj], dict]:
    """
    Evolve through Levine-Pichler protocol with shaped pulses.
    
    Shaped pulses (Gaussian, Blackman, etc.) reduce spectral leakage
    compared to square pulses, improving robustness at the cost of
    slightly longer gate times.
    
    Parameters
    ----------
    initial_states : dict
        Initial states
    pulse_shape : str
        Pulse shape: "square", "gaussian", "cosine", "blackman", "drag"
    Omega : float
        Peak Rabi frequency (rad/s)
    Delta : float
        Two-photon detuning (rad/s)
    V : float
        Blockade strength (rad/s)
    xi : complex
        Phase factor for second pulse
    tau_single : float
        Single pulse duration (s)
    hs : HilbertSpace
        Hilbert space object
    c_ops : list
        Collapse operators
    delta_zeeman, delta_stark : float
        Additional frequency shifts
    trap_laser_on : bool
        Whether trap is on during gate
    drag_lambda : float
        DRAG correction coefficient (for "drag" shape only)
    n_time_steps : int
        Number of time steps for evolution
    verbose : bool
        Print progress
        
    Returns
    -------
    final_states : dict
        Final states
    info : dict
        Pulse shaping information
    """
    if c_ops is None:
        c_ops = []
    
    from .pulse_shaping import (
        get_pulse_envelope, area_correction_factor, normalize_pulse_area
    )
    from .hamiltonians import build_laser_hamiltonian, build_full_hamiltonian
    
    # Compute area correction for non-square pulses
    correction = area_correction_factor(pulse_shape, tau_single)
    Omega_peak = Omega * correction  # Scale peak to maintain area
    
    if verbose:
        print(f"    Shaped pulse: {pulse_shape}")
        print(f"    Area correction: {correction:.3f}")
        print(f"    Peak Ω/(2π): {Omega_peak/(2*np.pi*1e6):.2f} MHz")
    
    # Time grid
    t_pulse = np.linspace(0, tau_single, n_time_steps)
    
    # Build time-dependent Hamiltonian function
    def H_func(t, args):
        """Time-dependent Hamiltonian with shaped envelope."""
        envelope = get_pulse_envelope(pulse_shape, t, tau_single)
        Omega_t = Omega_peak * envelope
        return build_full_hamiltonian(
            Omega=Omega_t,
            Delta=Delta,
            V=V,
            hs=hs,
        )
    
    final_states = {}
    options = {'atol': 1e-10, 'rtol': 1e-8, 'nsteps': 50000}
    
    for label, psi0 in initial_states.items():
        # First pulse with shape
        # For shaped pulses, need time-dependent solver
        # Here we use piecewise constant approximation
        psi = psi0
        dt = tau_single / n_time_steps
        
        for i in range(n_time_steps - 1):
            t_mid = (t_pulse[i] + t_pulse[i+1]) / 2
            envelope = get_pulse_envelope(pulse_shape, t_mid, tau_single)
            Omega_t = Omega_peak * envelope
            
            H_t = build_full_hamiltonian(Omega_t, Delta, V, hs, delta_stark=delta_stark, delta_zeeman=delta_zeeman, trap_laser_on=trap_laser_on)
            tlist_step = [0, dt]
            psi = evolve_state(H_t, psi, tlist_step, c_ops, options)
        
        # Second pulse with phase shift ξ
        for i in range(n_time_steps - 1):
            t_mid = (t_pulse[i] + t_pulse[i+1]) / 2
            envelope = get_pulse_envelope(pulse_shape, t_mid, tau_single)
            Omega_t = Omega_peak * envelope * xi  # Include phase shift
            
            H_t = build_full_hamiltonian(Omega_t, Delta, V, hs, delta_stark=delta_stark, delta_zeeman=delta_zeeman, trap_laser_on=trap_laser_on )
            tlist_step = [0, dt]
            psi = evolve_state(H_t, psi, tlist_step, c_ops, options)
        
        final_states[label] = psi
    
    info = {
        'pulse_shape': pulse_shape,
        'area_correction': correction,
        'peak_scaling': correction,
        'drag_lambda': drag_lambda,
    }
    
    return final_states, info


# =============================================================================
# MAIN SIMULATION FUNCTION
# =============================================================================

@dataclass
class SimulationResult:
    """
    Container for CZ gate simulation results.
    
    This dataclass organizes all simulation outputs for easy access
    and analysis. All fields match the dict returned by simulate_CZ_gate().
    
    Attributes
    ----------
    
    Gate Performance
    ~~~~~~~~~~~~~~~~
    avg_fidelity : float
        Average gate fidelity across all computational basis states
    fidelities : Dict[str, float]
        Per-state fidelities {"00": f00, "01": f01, "10": f10, "11": f11}
    phase_info : Dict
        Phase extraction details (global phase, per-state phases)
    
    Protocol Info
    ~~~~~~~~~~~~~
    protocol : str
        Protocol name ("Levine-Pichler" or "Jandura-Pupillo")
    n_pulses : int
        Number of pulses (2 for LP, 1 for JP)
    hilbert_space_dim : int
        Single-atom Hilbert space dimension (3 or 4)
    
    Rydberg Parameters
    ~~~~~~~~~~~~~~~~~~
    Omega : float
        Two-photon Rabi frequency (rad/s)
    V : float
        Blockade interaction strength (rad/s)
    Delta : float
        Static detuning (rad/s)
    V_over_Omega : float
        Blockade ratio V/Ω (dimensionless)
    Delta_over_Omega : float
        Detuning ratio Δ/Ω (dimensionless)
    
    Timing
    ~~~~~~
    tau_single : float
        Single pulse duration (s)
    tau_total : float
        Total gate duration (s)
    xi : complex
        Phase shift ξ = exp(-iφ) for LP protocol
    
    Geometry
    ~~~~~~~~
    R : float
        Inter-atom spacing (m)
    spacing_factor : float
        Spacing in units of λ/(2·NA)
    
    Trap Properties
    ~~~~~~~~~~~~~~~
    U0_mK : float
        Trap depth (mK)
    omega_r_kHz : float
        Radial trap frequency (kHz)
    sigma_r_nm : float
        Position uncertainty (nm)
    trap_wavelength_nm : float
        Tweezer wavelength (nm)
    magic_wavelength_analysis : Dict
        Magic wavelength details (alpha_ratio, enhancement, etc.)
    
    Noise Budget
    ~~~~~~~~~~~~
    noise_breakdown : Dict
        Full noise budget (rates, totals, metadata)
    include_noise : bool
        Whether noise was included in simulation
    include_motional_dephasing : bool
        Whether motional dephasing was included
    
    Pulse Info
    ~~~~~~~~~~
    pulse_info : Dict
        Pulse shaping details (shape, DRAG coefficient, etc.)
    
    Configuration
    ~~~~~~~~~~~~~
    config : AtomicConfiguration
        Full atomic configuration object
    species : str
        Atomic species ("Rb87" or "Cs133")
    n_rydberg : int
        Rydberg principal quantum number
    qubit_0, qubit_1 : Tuple[int, int]
        Qubit state quantum numbers (F, mF)
    
    Environment
    ~~~~~~~~~~~
    temperature_K : float
        Atom temperature (K)
    B_field_T : float
        Magnetic field (T)
    
    Raw Outputs
    ~~~~~~~~~~~
    results : Dict[str, Qobj]
        Final density matrices for each input state
    H1, H2 : Qobj
        Pulse Hamiltonians (None for JP protocol)
    c_ops : List[Qobj]
        Collapse operators used
    hs : HilbertSpace
        Hilbert space object with basis states
    """
    # Gate performance
    avg_fidelity: float
    fidelities: Dict[str, float]
    phase_info: Dict
    
    # Protocol info
    protocol: str
    n_pulses: int
    hilbert_space_dim: int
    
    # Rydberg parameters (all in rad/s unless noted)
    Omega: float
    V: float
    Delta: float
    V_over_Omega: float
    
    # Timing
    tau_single: float  # s
    tau_total: float   # s
    
    # Geometry
    R: float  # m
    
    # Fields with defaults MUST come after fields without defaults
    Delta_over_Omega: float = 0.0
    xi: complex = 1.0
    spacing_factor: float = 2.8
    
    # Trap properties
    U0_mK: float = 0.0
    omega_r_kHz: float = 0.0
    sigma_r_nm: float = 0.0
    trap_wavelength_nm: float = 1064.0
    magic_wavelength_analysis: Dict = None
    
    # Noise budget
    noise_breakdown: Dict = None
    include_noise: bool = True
    include_motional_dephasing: bool = True
    
    # Pulse info
    pulse_info: Dict = None
    
    # Configuration
    config: AtomicConfiguration = None
    species: str = "Rb87"
    n_rydberg: int = 70
    qubit_0: Tuple[int, int] = (1, 0)
    qubit_1: Tuple[int, int] = (2, 0)
    
    # Environment
    temperature_K: float = 2e-6
    B_field_T: float = 1e-4
    
    # Coherent frequency shifts
    delta_zeeman: float = 0.0  # Differential Zeeman shift (rad/s)
    delta_stark: float = 0.0   # Differential AC Stark shift (rad/s)
    trap_laser_on: bool = True # Whether trap was on during gate
    
    # Raw outputs
    results: Dict = None
    H1: Qobj = None
    H2: Qobj = None
    c_ops: List = None
    hs: HilbertSpace = None
    
    def __post_init__(self):
        """Initialize mutable default values."""
        if self.magic_wavelength_analysis is None:
            self.magic_wavelength_analysis = {}
        if self.noise_breakdown is None:
            self.noise_breakdown = {}
        if self.pulse_info is None:
            self.pulse_info = {}
        if self.results is None:
            self.results = {}
        if self.c_ops is None:
            self.c_ops = []
    
    @property
    def Omega_MHz(self) -> float:
        """Rabi frequency in MHz."""
        return self.Omega / (2 * np.pi * 1e6)
    
    @property
    def V_MHz(self) -> float:
        """Blockade interaction in MHz."""
        return self.V / (2 * np.pi * 1e6)
    
    @property
    def Delta_MHz(self) -> float:
        """Detuning in MHz."""
        return self.Delta / (2 * np.pi * 1e6)
    
    @property
    def gate_time_us(self) -> float:
        """Total gate time in μs."""
        return self.tau_total * 1e6
    
    @property
    def R_um(self) -> float:
        """Interatomic spacing in μm."""
        return self.R * 1e6
    
    @property
    def xi_rad(self) -> float:
        """Phase shift in radians."""
        return np.angle(self.xi)
    
    @property
    def xi_deg(self) -> float:
        """Phase shift in degrees."""
        return np.degrees(np.angle(self.xi))
    
    @property
    def temperature_uK(self) -> float:
        """Temperature in μK."""
        return self.temperature_K * 1e6
    
    @property
    def B_field_Gauss(self) -> float:
        """Magnetic field in Gauss."""
        return self.B_field_T * 1e4
    
    def print_summary(self):
        """Print a human-readable summary of results."""
        print("=" * 70)
        print("CZ GATE SIMULATION RESULTS")
        print("=" * 70)
        
        print(f"\n{'GATE PERFORMANCE':^70}")
        print("-" * 70)
        print(f"Average fidelity: {self.avg_fidelity:.6f} ({(1-self.avg_fidelity)*100:.4f}% error)")
        print(f"\nPer-state fidelities:")
        for state, fid in self.fidelities.items():
            print(f"  |{state}⟩ → {fid:.6f}")
        
        print(f"\n{'PROTOCOL':^70}")
        print("-" * 70)
        print(f"Protocol: {self.protocol}")
        print(f"Pulses: {self.n_pulses}")
        print(f"Hilbert space: {self.hilbert_space_dim}-level")
        if self.n_pulses == 2:
            print(f"Phase shift ξ: {self.xi_deg:.2f}°")
        
        print(f"\n{'RYDBERG PARAMETERS':^70}")
        print("-" * 70)
        print(f"Ω/(2π): {self.Omega_MHz:.3f} MHz")
        print(f"V/(2π): {self.V_MHz:.2f} MHz")
        print(f"V/Ω: {self.V_over_Omega:.2f}")
        if self.Delta_over_Omega != 0:
            print(f"Δ/(2π): {self.Delta_MHz:.3f} MHz (Δ/Ω = {self.Delta_over_Omega:.4f})")
        
        print(f"\n{'TIMING':^70}")
        print("-" * 70)
        print(f"Single pulse: {self.tau_single*1e6:.3f} μs")
        print(f"Total gate: {self.gate_time_us:.3f} μs")
        
        print(f"\n{'GEOMETRY & TRAP':^70}")
        print("-" * 70)
        print(f"Spacing R: {self.R_um:.2f} μm")
        print(f"Trap depth: {self.U0_mK:.2f} mK")
        print(f"Position σ: {self.sigma_r_nm:.1f} nm")
        print(f"Trap laser on during gate: {self.trap_laser_on}")
        
        print(f"\n{'COHERENT SHIFTS':^70}")
        print("-" * 70)
        print(f"Zeeman shift δ_z/(2π): {self.delta_zeeman/(2*np.pi):.1f} Hz")
        print(f"Stark shift δ_s/(2π): {self.delta_stark/(2*np.pi):.1f} Hz")
        
        if self.noise_breakdown:
            print(f"\n{'NOISE BUDGET':^70}")
            print("-" * 70)
            nb = self.noise_breakdown
            print(f"Decay rate: {nb.get('total_decay_rate', 0)/1e3:.2f} kHz")
            print(f"Dephasing rate: {nb.get('total_dephasing_rate', 0)/1e3:.2f} kHz")
            print(f"Loss rate: {nb.get('total_loss_rate', 0)/1e3:.2f} kHz")
            print(f"Collapse operators: {nb.get('n_collapse_ops', 0)}")
        
        print("=" * 70)


def simulate_CZ_gate(
    # --- Protocol-specific inputs (REQUIRED) ---
    simulation_inputs: Union[LPSimulationInputs, JPSimulationInputs, SmoothJPSimulationInputs],
    
    # --- Atomic configuration ---
    config: AtomicConfiguration = None,  # If None, created from species/n_rydberg
    species: str = "Rb87",
    n_rydberg: int = 70,
    qubit_0: Tuple[int, int] = (1, 0),   # |0⟩ = |F=1, mF=0⟩ (clock state)
    qubit_1: Tuple[int, int] = (2, 0),   # |1⟩ = |F=2, mF=0⟩ (clock state)
    hilbert_space_dim: int = 3,         # 3 for |0⟩,|1⟩,|r⟩ or 4 for |0⟩,|1⟩,|r+⟩,|r-⟩
    
    # --- Optical tweezer ---
    tweezer_power: float = 30e-3,       # Tweezer power (W)
    tweezer_waist: float = 1.0e-6,      # Tweezer beam waist (m)
    tweezer_wavelength_nm: float = None, # Wavelength (nm), None for species default
    
    # --- Environment ---
    temperature: float = 2e-6,          # Atom temperature (K)
    B_field: float = 1e-4,              # Magnetic field (T)
    NA: float = 0.5,                    # Numerical aperture
    spacing_factor: float = 2.8,        # Atom spacing in units of λ/(2·NA)
    
    # --- Noise control ---
    include_noise: bool = True,
    background_loss_rate_hz: float = None,
    
    # --- Trap laser control ---
    trap_laser_on: bool = True,         # Whether trap laser is on during gate
    
    # --- Output control ---
    verbose: bool = False,
    return_dataclass: bool = True,
) -> Union[SimulationResult, Dict]:
    """
    Simulate CZ gate with comprehensive physics-based noise modeling.
    
    PHYSICS PIPELINE OVERVIEW
    =========================
    
    This function implements a 12-step pipeline that computes CZ gate fidelity
    from first-principles physical parameters:
    
    **Step 0: Hilbert Space & Configuration**
        Build single-atom basis {|0⟩, |1⟩, |r⟩} (or 4-level with mJ states).
        Create AtomicConfiguration with species-specific properties.
    
    **Step 1: Trap Wavelength**
        Handle magic wavelength selection (1064 nm standard, or species-specific).
        Compute ground-state polarizability at chosen wavelength.
    
    **Step 2: Atom Spacing**
        R = spacing_factor × λ/(2·NA)
        Typical: 2.8 × 532nm = 3 μm for Rb87 with NA=0.5
    
    **Step 3: Rabi Frequencies**
        Two-photon Rydberg excitation: |1⟩ → |e⟩ → |r⟩
        Ω₁ = d·E₁/ℏ (780 nm leg), Ω₂ = d·E₂/ℏ (480 nm leg)
        Ω_eff = Ω₁·Ω₂/(2Δₑ) where Δₑ ~ 2π×5 GHz
    
    **Step 4: Blockade Interaction**
        V = C₆/R⁶ (van der Waals)
        At R=3 μm, n=70: V/2π ~ 30-100 MHz
    
    **Step 5: Protocol Parameters**
        V/Ω determines optimal Δ/Ω and Ωτ from lookup tables.
        LP: Δ/Ω ~ 0.37-0.38, Ωτ ~ 4.29 per pulse
        JP: Δ=0, uses bang-bang phase control
    
    **Step 6: Trap-Dependent Noise**
        compute_trap_dependent_noise() calculates:
        - Trap depth U₀ = α·I/(2ε₀c) ~ 1-3 mK
        - Trap frequency ω_r = √(4U₀/mw²) ~ 50-100 kHz
        - Position uncertainty σ_r = √(ℏ/mω_r) + thermal ~ 30-50 nm
        - Blockade fluctuation δV/V = 6σ_r/R ~ 1-5%
        - Magic wavelength enhancement factor
    
    **Step 7: Hamiltonians**
        H = (Ω/2)(|1⟩⟨r| + h.c.) - Δ|r⟩⟨r| + V|rr⟩⟨rr|
        For LP: H₁ (first pulse), H₂ = e^{iξ}H₁ (second pulse)
        For JP: Time-dependent phase φ(t) with bang-bang switching
    
    **Step 8: Collapse Operators**
        build_all_noise_operators() creates Lindblad operators:
        - Decay: |r⟩ → |1⟩ (γ_r ~ 10 kHz at n=70)
        - Dephasing: |r⟩⟨r| (laser linewidth, thermal, Zeeman)
        - Loss: atom escape from anti-trapping
        - Scattering: intermediate state |e⟩ contribution
    
    **Step 9: Initial States**
        Prepare computational basis: |00⟩, |01⟩, |10⟩, |11⟩
    
    **Step 10: Time Evolution**
        mesolve(H, ρ₀, tlist, c_ops) integrates Lindblad equation.
        LP: Two sequential pulses with phase shift.
        JP: Bang-bang segments with phase switching.
    
    **Step 11: Fidelity**
        F = |⟨ψ_ideal|ψ_final⟩|² averaged over basis states.
        Extract global phase from |00⟩ evolution.
    
    **Step 12: Package Results**
        Return SimulationResult dataclass with all physics outputs.
    
    Parameters
    ----------
    simulation_inputs : LPSimulationInputs, JPSimulationInputs, or SmoothJPSimulationInputs
        Protocol-specific simulation configuration. REQUIRED.
        
        Three protocol families are supported:
        
        **Levine-Pichler (LP)** — Two-pulse protocol with static detuning:
            LPSimulationInputs(
                excitation=TwoPhotonExcitationConfig(...),
                noise=NoiseSourceConfig(...),
                delta_over_omega=None,  # Use V/Ω lookup
                omega_tau=None,  # Use V/Ω lookup
                pulse_shape="square",
                drag_lambda=1.0,
            )
        
        **Smooth Jandura-Pupillo (recommended JP path)** — Continuous
        sinusoidal modulation of Ω(t) and φ(t), validated to ≥99% fidelity:
            SmoothJPSimulationInputs(
                excitation=TwoPhotonExcitationConfig(...),
                noise=NoiseSourceConfig(...),
                A=0.637,               # Modulation depth
                omega_mod_ratio=2.0,   # ω_mod/Ω ratio
                phi_offset=0.0,        # Phase offset (rad)
                delta_over_omega=None, # Use default lookup
                omega_tau=None,        # Use default lookup
            )
        
        **JPSimulationInputs (legacy bang-bang interface)** — Accepted for
        backward compatibility. Internally routed to the smooth JP solver
        which has superseded the piecewise-constant bang-bang code path.
        Users who pass JPSimulationInputs will get smooth JP evolution:
            JPSimulationInputs(
                excitation=TwoPhotonExcitationConfig(...),
                noise=NoiseSourceConfig(...),
                omega_tau=None,  # Use V/Ω lookup
            )
        
    config : AtomicConfiguration, optional
        Pre-configured atomic properties. If None, built from species/n_rydberg.
        
    species : str
        Atomic species ("Rb87" or "Cs133")
        
    n_rydberg : int
        Rydberg principal quantum number (typical: 50-100)
        
    qubit_0, qubit_1 : tuple
        Qubit state quantum numbers (F, mF). Default: clock states (1,0) and (2,0)
        
    hilbert_space_dim : int
        Single-atom Hilbert space dimension:
        - 3: |0⟩, |1⟩, |r⟩ (standard model)
        - 4: |0⟩, |1⟩, |r+⟩, |r-⟩ (includes mJ Zeeman substates)
        
    tweezer_power, tweezer_waist, tweezer_wavelength_nm : float
        Optical tweezer parameters for trap physics and noise
        
    temperature : float
        Atom temperature (K). Typical: 2-20 μK
        
    B_field : float
        Magnetic field (T). Typical: 0.1-10 Gauss
        
    NA : float
        Microscope numerical aperture (determines minimum spot size)
        
    spacing_factor : float
        Atom spacing in units of λ/(2·NA). Typical: 2.5-4.0
        
    include_noise : bool
        Whether to include decoherence (Lindblad collapse operators)
        
    trap_laser_on : bool
        Whether trap laser is on during gate execution
        
    verbose : bool
        Print progress information
        
    return_dataclass : bool
        If True, return SimulationResult. If False, return dict (legacy).
        
    Returns
    -------
    SimulationResult or dict
        Comprehensive results including gate fidelity, noise budget, 
        Rydberg parameters, timing, trap properties, and raw outputs.
        Use result.print_summary() for human-readable output.
        
    Examples
    --------
    >>> # Basic LP simulation
    >>> from qpu_simulator.micro_physics.neutral_atoms.rydberg_gates import (
    ...     simulate_CZ_gate, LPSimulationInputs
    ... )
    >>> result = simulate_CZ_gate(simulation_inputs=LPSimulationInputs())
    >>> result.print_summary()
    
    >>> # JP protocol with custom laser config
    >>> from qpu_simulator.micro_physics.neutral_atoms.rydberg_gates import (
    ...     JPSimulationInputs, TwoPhotonExcitationConfig, LaserParameters
    ... )
    >>> jp_inputs = JPSimulationInputs(
    ...     excitation=TwoPhotonExcitationConfig(
    ...         laser_1=LaserParameters(power=100e-6, waist=40e-6, linewidth_hz=500),
    ...         laser_2=LaserParameters(power=1.0, waist=40e-6, linewidth_hz=500),
    ...     )
    ... )
    >>> result = simulate_CZ_gate(simulation_inputs=jp_inputs, n_rydberg=70)
    >>> print(f"Fidelity: {result.avg_fidelity:.5f}")
    
    >>> # Compare LP vs JP protocols
    >>> lp_result = simulate_CZ_gate(simulation_inputs=LPSimulationInputs())
    >>> jp_result = simulate_CZ_gate(simulation_inputs=JPSimulationInputs())
    >>> print(f"LP: {lp_result.avg_fidelity:.4f}, JP: {jp_result.avg_fidelity:.4f}")
    
    See Also
    --------
    SimulationResult : Dataclass containing all simulation outputs
    compute_trap_dependent_noise : Unified trap physics + noise rates
    build_all_noise_operators : Lindblad collapse operator construction
    """
    # =========================================================================
    # STEP 0: Extract parameters from simulation_inputs
    # =========================================================================
    # The simulation_inputs dataclass provides a unified way to specify
    # protocol-specific parameters including laser config, noise sources,
    # and pulse shaping.
    
    # Determine protocol from simulation_inputs type
    if isinstance(simulation_inputs, LPSimulationInputs):
        protocol = "levine_pichler"
        pulse_shape = simulation_inputs.pulse_shape
        drag_lambda = simulation_inputs.drag_lambda
        delta_over_omega = simulation_inputs.delta_over_omega  # May be None (use V/Ω lookup)
        omega_tau = simulation_inputs.omega_tau  # May be None (use V/Ω lookup)
    elif isinstance(simulation_inputs, SmoothJPSimulationInputs):
        protocol = "smooth_jp"
        pulse_shape = "smooth_sinusoidal"
        delta_over_omega = simulation_inputs.delta_over_omega  # May be None (use defaults)
        omega_tau = simulation_inputs.omega_tau  # May be None (use defaults)
        drag_lambda = 1.0  # Not used for smooth JP
    elif isinstance(simulation_inputs, JPSimulationInputs):
        # JP bang-bang: piecewise-constant phase control
        # Preserves switching_times/phases and routes to evolve_bangbang_jp()
        protocol = "jandura_pupillo"
        pulse_shape = "bangbang"
        delta_over_omega = 0.0  # JP bang-bang has no static detuning
        omega_tau = simulation_inputs.omega_tau  # May be None (use V/Ω lookup)
        drag_lambda = 1.0  # Not used for JP
    else:
        raise TypeError(
            f"simulation_inputs must be LPSimulationInputs, JPSimulationInputs, "
            f"or SmoothJPSimulationInputs, got {type(simulation_inputs).__name__}"
        )
    
    # Extract laser configuration
    exc = simulation_inputs.excitation
    laser_1 = exc.laser_1
    laser_2 = exc.laser_2
    Delta_e = exc.Delta_e
    counter_propagating = exc.counter_propagating
    
    # Extract noise configuration
    noise_cfg = simulation_inputs.noise
    include_spontaneous_emission = noise_cfg.include_spontaneous_emission
    include_intermediate_scattering = noise_cfg.include_intermediate_scattering
    include_motional_dephasing = noise_cfg.include_motional_dephasing
    include_doppler_dephasing = noise_cfg.include_doppler_dephasing
    include_intensity_noise = noise_cfg.include_intensity_noise
    intensity_noise_frac = noise_cfg.intensity_noise_frac
    include_laser_dephasing = noise_cfg.include_laser_dephasing
    include_magnetic_dephasing = noise_cfg.include_magnetic_dephasing
    
    # =========================================================================
    # STEP 1: Select Hilbert space and create AtomicConfiguration
    # =========================================================================
    # We model each atom as a 3-level or 4-level system:
    #   3-level: |0⟩ (F=1,mF=0), |1⟩ (F=2,mF=0), |r⟩ (nS₁/₂,mJ=+1/2)
    #   4-level: adds |r-⟩ (nS₁/₂,mJ=-1/2) for Zeeman substate mixing
    # The two-atom Hilbert space is dim² = 9 or 16 dimensional.
    hs = build_hilbert_space(hilbert_space_dim)
    
    # AtomicConfiguration stores all species-specific properties:
    # mass, hyperfine structure, dipole moments, C6 coefficient, etc.
    if config is None:
        config = AtomicConfiguration(
            species=species,
            qubit_0=qubit_0,
            qubit_1=qubit_1,
            n_rydberg=n_rydberg,
            L_rydberg="S",
        )
    
    # Get full atom properties dictionary from database
    atom = get_atom_properties(config.species)
    
    # =========================================================================
    # STEP 2: Extract laser parameters from simulation_inputs
    # =========================================================================
    # laser_1 and laser_2 are extracted from simulation_inputs.excitation above
    rydberg_power_1 = laser_1.power
    rydberg_waist_1 = laser_1.waist
    pol_1 = laser_1.polarization
    laser_linewidth_1 = laser_1.linewidth_hz
    polarization_purity_1 = laser_1.polarization_purity

    rydberg_power_2 = laser_2.power
    rydberg_waist_2 = laser_2.waist
    pol_2 = laser_2.polarization
    laser_linewidth_2 = laser_2.linewidth_hz
    polarization_purity_2 = laser_2.polarization_purity
    
    # Combine laser linewidths (add in quadrature for independent noise)
    if laser_linewidth_1 is not None and laser_linewidth_2 is not None:
        laser_linewidth_hz = np.sqrt(laser_linewidth_1**2 + laser_linewidth_2**2)
    elif laser_linewidth_1 is not None:
        laser_linewidth_hz = laser_linewidth_1
    elif laser_linewidth_2 is not None:
        laser_linewidth_hz = laser_linewidth_2
    else:
        laser_linewidth_hz = 1000.0  # Default 1 kHz
    
    # =========================================================================
    # STEP 3: Handle trap wavelength
    # =========================================================================
    # The tweezer wavelength determines the AC Stark shift (trap depth) for
    # different atomic states. At a "magic" wavelength, ground and Rydberg
    # states see the same shift → reduced anti-trapping loss.
    # Standard: 1064 nm (Nd:YAG), 852 nm (near Cs D2), or species-specific.
    if tweezer_wavelength_nm is not None:
        trap_wavelength = tweezer_wavelength_nm * 1e-9
        alpha_ground_at_wavelength = get_polarizability_at_wavelength(
            config.species, "ground", tweezer_wavelength_nm
        )
    else:
        trap_wavelength = atom["trap_wavelength"]
        alpha_ground_at_wavelength = atom["alpha_ground"]
    
    wavelength_nm = trap_wavelength * 1e9
    
    # =========================================================================
    # STEP 2: Compute spacing from optics
    # =========================================================================
    # Atom spacing R = spacing_factor × λ/(2·NA)
    # The diffraction limit sets minimum spot size w₀ ~ λ/(π·NA).
    # Typical: spacing_factor = 2.5-4, NA = 0.5-0.7 → R ~ 3-5 μm
    # Larger R → weaker blockade V ∝ R⁻⁶, but less crosstalk.
    R = tweezer_spacing(trap_wavelength, NA, spacing_factor)
    
    # =========================================================================
    # STEP 3: Compute Rabi frequencies from Rydberg lasers
    # =========================================================================
    # Two-photon Rydberg excitation: |1⟩ --Ω₁--> |e⟩ --Ω₂--> |r⟩
    #   Leg 1: 780 nm (Rb) or 852 nm (Cs), detuned Δₑ from intermediate |e⟩
    #   Leg 2: 480 nm (Rb) or 510 nm (Cs) to Rydberg state
    # Effective Rabi: Ω = Ω₁·Ω₂/(2Δₑ) when Δₑ >> Ω₁, Ω₂
    E0_1 = laser_E0(rydberg_power_1, rydberg_waist_1)
    E0_2 = laser_E0(rydberg_power_2, rydberg_waist_2)
    
    # Get dipole moments from atom database
    # Leg 1: ground → intermediate (species-dependent: 5P3/2 for Rb, 6P3/2 for Cs)
    intermediate_state = get_default_intermediate_state(config.species)
    dipole_1e = atom["intermediate_states"][intermediate_state]["dipole_from_ground"]
    # Leg 2: intermediate → Rydberg (needs n-dependent scaling)
    # Reference dipole is for n_ref, scales as n^(-3/2)
    n_ref = atom["n_ref"]
    dipole_er_ref = atom["dipole_intermediate_to_rydberg_ref"]
    dipole_er = dipole_er_ref * (n_rydberg / n_ref)**(-1.5)
    
    # Single-photon Rabi frequencies from electric field and dipole moments
    Omega1 = single_photon_rabi(dipole_1e, E0_1)
    Omega2 = single_photon_rabi(dipole_er, E0_2)
    Omega = two_photon_rabi(Omega1, Omega2, Delta_e)
    
    # =========================================================================
    # STEP 3.5: Validate Rabi frequency is physically reasonable
    # =========================================================================
    # Maximum achievable two-photon Rabi frequency is limited by:
    # 1. Available laser power (typically <1W focused to ~50μm)
    # 2. Intermediate state scattering (Ω ≪ Δ_e to stay adiabatic)
    # 3. Practical laser systems (ultra-stable cavities, etc.)
    # Typical experimental range: 2π × 0.5-50 MHz
    # Exceptional: up to 2π × 100 MHz with high power
    
    OMEGA_MAX_PHYSICAL = 2 * np.pi * 100e6  # 100 MHz - extreme upper limit
    OMEGA_MAX_TYPICAL = 2 * np.pi * 30e6    # 30 MHz - typical high-power
    OMEGA_MIN_PRACTICAL = 2 * np.pi * 0.1e6 # 0.1 MHz - below this is impractical
    
    Omega_MHz = Omega / (2 * np.pi * 1e6)
    
    if Omega > OMEGA_MAX_PHYSICAL:
        import warnings
        warnings.warn(
            f"Ω/2π = {Omega_MHz:.1f} MHz exceeds physical limit (~100 MHz). "
            f"Results may be unphysical. Check laser powers.",
            UserWarning
        )
    elif Omega > OMEGA_MAX_TYPICAL and verbose:
        print(f"  ⚠ Note: Ω/2π = {Omega_MHz:.1f} MHz is unusually high (typical <30 MHz)")
    
    if Omega < OMEGA_MIN_PRACTICAL:
        import warnings
        warnings.warn(
            f"Ω/2π = {Omega_MHz*1e3:.1f} kHz is very low. Gate will be very slow "
            f"and susceptible to decoherence.",
            UserWarning
        )
    
    # =========================================================================
    # STEP 4: Compute blockade interaction
    # =========================================================================
    # Van der Waals interaction between two Rydberg atoms:
    #   V = C₆/R⁶ where C₆ ∝ n¹¹ (very strong n-dependence!)
    # At R = 3 μm, n = 70: V/2π ~ 30-100 MHz
    # The blockade regime requires V >> Ω (typically V/Ω > 5)
    #
    # Note: get_C6() returns C₆ in (rad/s)·m⁶ units, so rydberg_blockade()
    # returns V directly in rad/s - no HBAR conversion needed.
    C6 = get_C6(n_rydberg, config.species)
    V = rydberg_blockade(C6, R)  # Returns V in rad/s
    
    # =========================================================================
    # STEP 5: Get V/Ω-ADAPTIVE protocol parameters and compute timing
    # =========================================================================
    # The optimal gate parameters depend on the ratio V/Ω.
    # At V/Ω → ∞ (perfect blockade): Δ/Ω → 0.377, Ωτ → 4.29
    # At finite V/Ω: parameters shift to maintain high fidelity
    # See Levine et al. PRL 2019 & Jandura-Pupillo PRX Quantum 2022
    V_over_Omega = V / Omega if Omega > 0 else float('inf')
    
    # =========================================================================
    # Blockade regime info (for diagnostics only - no artificial penalties)
    # =========================================================================
    # The CZ gate mechanism REQUIRES strong blockade (V >> Ω).
    # With proper V units (rad/s), the Hamiltonian evolution will naturally
    # produce wrong dynamics (and thus low fidelity) when blockade is weak.
    #
    # Thresholds for reference:
    #   V/Ω > 100: Perfect blockade (negligible double excitation)
    #   V/Ω ~ 10-100: Good blockade (small coherent error)
    #   V/Ω ~ 5-10: Marginal blockade (noticeable error)
    #   V/Ω < 5: Weak blockade (significant physics failure)
    #   V/Ω < 1: No blockade (atoms evolve independently)
    
    if V_over_Omega < 1 and verbose:
        print(f"  ⚠ V/Ω = {V_over_Omega:.2f} < 1: NO BLOCKADE - atoms will evolve independently")
    elif V_over_Omega < 10 and verbose:
        print(f"  ⚠ V/Ω = {V_over_Omega:.1f} < 10: Weak blockade regime")
    
    protocol_params = get_protocol_params(protocol, V_over_Omega=V_over_Omega)
    is_jp_bangbang = protocol.lower() in ["jandura_pupillo", "jp", "single_pulse", "time_optimal"]
    is_smooth_jp = protocol.lower() in ["smooth_jp", "dark_state", "sinusoidal_jp"]
    is_jp_protocol = is_jp_bangbang or is_smooth_jp
    
    if verbose:
        print(f"  V/Ω = {V_over_Omega:.1f} → using adapted protocol parameters")
    
    if protocol.lower() in ["levine_pichler", "lp", "two_pulse"]:
        # Levine-Pichler: Two pulses with phase shift ξ between them
        # Each pulse: H = (Ω/2)(|1⟩⟨r| + h.c.) - Δ|r⟩⟨r| + V|rr⟩⟨rr|
        # Second pulse has additional phase: Ω → Ω·e^{iξ}
        _delta_over_omega = delta_over_omega if delta_over_omega is not None else protocol_params['delta_over_omega']
        _omega_tau = omega_tau if omega_tau is not None else protocol_params['omega_tau']
        tau_single = _omega_tau / Omega
        tau_total = 2 * tau_single
        Delta_gate = _delta_over_omega * Omega
        n_pulses = 2
        
        if verbose:
            override_note = " (override)" if (delta_over_omega is not None or omega_tau is not None) else ""
            print(f"  LP protocol: Δ/Ω = {_delta_over_omega:.4f}, Ωτ = {_omega_tau:.4f}{override_note}")
            
    elif is_jp_bangbang:
        # JP bang-bang: piecewise-constant phase control
        # φ(t) ∈ {±π/2, 0} switches at optimized times, NO static detuning
        # Uses switching_times/phases from simulation_inputs or protocol defaults
        _si = simulation_inputs  # shorthand
        _omega_tau = omega_tau if omega_tau is not None else protocol_params.get('omega_tau', 22.08)
        tau_single = _omega_tau / Omega
        tau_total = tau_single
        Delta_gate = 0  # No static detuning for bang-bang JP
        _delta_over_omega = 0.0
        n_pulses = 1
        
        # Extract switching_times and phases from simulation_inputs or protocol defaults
        _bb_switching_times = getattr(_si, 'switching_times', None) or protocol_params.get(
            'switching_times', [2.214, 8.823, 13.258, 19.867])
        _bb_phases = getattr(_si, 'phases', None) or protocol_params.get(
            'phases', [np.pi/2, 0, -np.pi/2, 0, np.pi/2])
        
        if verbose:
            n_seg = len(_bb_phases)
            override_note = " (override)" if omega_tau is not None else ""
            print(f"  JP bang-bang protocol: Ωτ = {_omega_tau:.4f}{override_note}")
            print(f"  Segments: {n_seg}, switching times: {[f'{t:.4f}' for t in _bb_switching_times]}")
            print(f"  Phases: {[f'{p/np.pi:+.3f}π' for p in _bb_phases]}")
            
    elif is_smooth_jp:
        # Smooth sinusoidal JP (Bluvstein-form): VALIDATED protocol
        # Uses continuous phase modulation: φ(t) = A·cos(ω·t - ϕ) + δ₀·t
        # Achieves >99.9% fidelity across V/Ω from 10-200
        _omega_tau = omega_tau if omega_tau is not None else protocol_params.get('omega_tau', 10.09)
        tau_single = _omega_tau / Omega
        tau_total = tau_single
        Delta_gate = 0  # No static detuning (effective detuning via phase slope)
        _delta_over_omega = delta_over_omega if delta_over_omega is not None else protocol_params.get('delta_over_omega', 0.0205)
        n_pulses = 1
        
        if verbose:
            A = protocol_params.get('A', 0.311 * np.pi)
            omega_mod_ratio = protocol_params.get('omega_mod_ratio', 1.242)
            override_note = " (override)" if omega_tau is not None else ""
            print(f"  Smooth JP protocol (Bluvstein-form): Ωτ = {_omega_tau:.4f}{override_note}")
            print(f"  Phase modulation: A = {A/np.pi:.3f}π, ω_mod/Ω = {omega_mod_ratio:.3f}")
            
    else:
        raise ValueError(f"Unknown protocol: {protocol}")
    
    # =========================================================================
    # STEP 6: Compute trap-dependent noise using unified function
    # =========================================================================
    # This is the KEY connection between trap physics and noise rates.
    # compute_trap_dependent_noise (from trap_physics.py) calculates:
    #   - Trap depth U₀ = α·I/(2ε₀c) from tweezer power/waist/polarizability
    #   - Trap frequency ω_r from harmonic approximation at trap bottom
    #   - Position uncertainty σ_r from quantum + thermal contributions
    #   - Blockade fluctuation δV/V = 6·σ_r/R from position uncertainty
    #   - Thermal dephasing rate from δV/V and V/Ω ratio
    #   - Doppler dephasing from thermal velocity
    #   - Intensity noise dephasing from trap fluctuations
    #   - Magic wavelength analysis (polarizability ratio, anti-trap loss)
    
    # Get Rydberg laser wavelengths from atomic configuration
    ryd_wl_1 = config.excitation_wavelength_1_nm  # Ground → intermediate (e.g., 780 nm for Rb)
    ryd_wl_2 = config.excitation_wavelength_2_nm  # Intermediate → Rydberg (e.g., 480 nm for Rb)
    
    trap_noise = compute_trap_dependent_noise(
        species=config.species,
        tweezer_power=tweezer_power,
        tweezer_waist=tweezer_waist,
        temperature=temperature,
        spacing=R,
        gate_time=tau_total,
        n_rydberg=config.n_rydberg,
        gamma_phi_laser=np.pi * laser_linewidth_hz,  # Convert linewidth to dephasing rate
        Omega_1=Omega1,
        Delta_e=Delta_e,
        intermediate_state=config.intermediate_state,
        Omega_eff=Omega,
        tweezer_wavelength_nm=wavelength_nm,
        # Noise parameters from simulation_inputs
        include_doppler=include_doppler_dephasing,
        include_intensity_noise=include_intensity_noise,
        intensity_noise_frac=intensity_noise_frac,
        rydberg_wavelength_1_nm=ryd_wl_1,
        rydberg_wavelength_2_nm=ryd_wl_2,
        counter_propagating=counter_propagating,
    )
    
    U0 = trap_noise['trap_depth_uK'] * KB / 1e6
    omega_r = trap_noise['trap_freq_radial_kHz'] * 2 * np.pi * 1e3
    sigma_r = trap_noise['position_uncertainty_nm'] * 1e-9
    
    magic_analysis = {
        'alpha_ratio': trap_noise['alpha_ratio'],
        'alpha_ground_au': trap_noise['alpha_ground_au'],
        'alpha_rydberg_au': trap_noise['alpha_rydberg_au'],
        'gamma_antitrap_Hz': trap_noise['gamma_loss_antitrap'],
        'differential_shift_Hz': trap_noise['differential_shift_Hz'],
        'magic_enhancement': trap_noise['magic_enhancement'],
        'wavelength_nm': trap_noise['wavelength_nm'],
    }
    
    # =========================================================================
    # STEP 6b: Calculate Zeeman and Stark shifts (coherent frequency shifts)
    # =========================================================================
    # These are COHERENT shifts that modify the effective detuning in the
    # Hamiltonian. They are distinct from the INCOHERENT dephasing rates
    # computed in the noise model.
    #
    # Zeeman shift: From magnetic field B
    #   - Clock states (mF=0): Only quadratic Zeeman ~ 575 Hz/G² (Rb87)
    #   - Non-clock states: Linear Zeeman ~ 700 kHz/G (much larger!)
    #
    # Stark shift: From trap laser AC Stark effect
    #   - Depends on differential polarizability Δα = |α_r - α_g|
    #   - At magic wavelength: Δα ≈ 0, so δ_stark ≈ 0
    #   - At 1064 nm: Can be significant (kHz-MHz range)
    
    delta_zeeman = calculate_zeeman_shift(
        B_field=B_field,
        qubit_0=config.qubit_0,
        qubit_1=config.qubit_1,
        species=config.species
    )
    
    # Qubit state differential Stark shift
    # This is the shift between |0⟩ (F=1) and |1⟩ (F=2) hyperfine states,
    # NOT the ground-to-Rydberg shift (which is huge but pre-compensated).
    #
    # Physics: The hyperfine states have slightly different polarizabilities
    # - For Rb87 at 1064 nm: Δα ≈ 2.4 a.u. (~0.35% of α_ground)
    # - At 1 mK trap depth: δf ≈ 70 kHz
    #
    # This shift affects the qubit frequency and contributes to dephasing
    # if trap intensity fluctuates.
    if trap_laser_on:
        from .trap_physics import calculate_qubit_stark_shift
        
        # Get trap depth if available
        trap_depth_mK = trap_noise.get('trap_depth_uK', 0) / 1000  # Convert μK to mK
        
        delta_stark = calculate_qubit_stark_shift(
            tweezer_power=tweezer_power,
            tweezer_waist=tweezer_waist,
            species=config.species,
            trap_depth_mK=trap_depth_mK if trap_depth_mK > 0 else None,
        )
    else:
        delta_stark = 0.0
    
    if verbose:
        print(f"  Zeeman shift: δ_z/(2π) = {delta_zeeman/(2*np.pi):.1f} Hz")
        print(f"  Stark shift:  δ_s/(2π) = {delta_stark/(2*np.pi):.1f} Hz")
        print(f"  Trap laser during gate: {trap_laser_on}")
    
    # =========================================================================
    # STEP 7: Build Hamiltonians using UNIFIED BUILDERS
    # =========================================================================
    # The two-atom Hamiltonian in the rotating frame:
    #   H = Σᵢ [(Ω/2)(|1⟩ᵢ⟨r| + h.c.) - (Δ + δ_z + δ_s)|r⟩ᵢ⟨r|] + V|rr⟩⟨rr|
    # where δ_z = Zeeman shift, δ_s = Stark shift
    # For LP protocol, H2 has phase shift: Ω → Ω·e^{iξ}
    # For JP protocol, we use time-dependent phase φ(t) with bang-bang switching
    if verbose:
        print(f"Building Hamiltonians using unified builders (dim={hs.dim})")
        print(f"  Protocol: {protocol_params['name']}")
    
    if not is_jp_protocol:
        # Build H1 for first pulse (no phase shift)
        H1 = build_full_hamiltonian(
            Omega=Omega,
            Delta=Delta_gate,
            V=V,
            hs=hs,
            dim=hilbert_space_dim,
            delta_zeeman=delta_zeeman,
            delta_stark=delta_stark,
            trap_laser_on=trap_laser_on,
        )
        
        # Compute phase shift ξ for second pulse
        xi = compute_phase_shift_xi(Delta_gate, Omega, tau_single)
        
        # Build H2 for second pulse (with phase shift ξ)
        # The phase appears on Ω: Ω → Ω·e^{iξ}
        H2 = build_full_hamiltonian(
            Omega=Omega * xi,  # Include phase shift
            Delta=Delta_gate,
            V=V,
            hs=hs,
            dim=hilbert_space_dim,
            delta_zeeman=delta_zeeman,
            delta_stark=delta_stark,
            trap_laser_on=trap_laser_on,
        )
    else:
        # JP/smooth-JP protocols build Hamiltonians during time evolution
        H1 = None
        H2 = None
        xi = 1.0
    
    # =========================================================================
    # STEP 8: Build collapse operators using UNIFIED NOISE MODEL
    # =========================================================================
    # The Lindblad master equation adds decoherence via collapse operators:
    #   dρ/dt = -i[H,ρ] + Σⱼ γⱼ(LⱼρLⱼ† - ½{Lⱼ†Lⱼ, ρ})
    # We build Lⱼ operators for each noise source using build_all_noise_operators()
    # from noise_models.py, which takes the rates from compute_trap_dependent_noise().
    c_ops = []
    noise_breakdown = {
        'total_decay_rate': 0.0,
        'total_dephasing_rate': 0.0,
        'total_loss_rate': 0.0,
        'n_collapse_ops': 0,
        'motional_dephasing_included': include_motional_dephasing,
        'gamma_scatter_intermediate': trap_noise['gamma_scatter_intermediate'],
        'Omega1_MHz': Omega1 / (2 * np.pi * 1e6),
    }
    
    if include_noise:
        # Laser dephasing: γ_φ = π × linewidth for Lorentzian lineshape
        # laser_linewidth_hz is extracted from simulation_inputs above
        gamma_phi = np.pi * laser_linewidth_hz
        
        # Background gas collision loss (typically negligible at UHV)
        if background_loss_rate_hz is not None:
            gamma_loss_background = background_loss_rate_hz
        else:
            gamma_loss_background = trap_noise['gamma_loss_background']
        
        # Motional dephasing from blockade fluctuations δV/V
        # This is often the DOMINANT error source at high temperature!
        if include_motional_dephasing:
            gamma_blockade_fluct = trap_noise['gamma_phi_thermal']
            gamma_motional = gamma_blockade_fluct
        else:
            gamma_motional = 0.0
            gamma_blockade_fluct = 0.0
        
        # Doppler dephasing from thermal atomic velocity
        # This is a key temperature-dependent noise source!
        # γ_doppler = k_eff × v_thermal where k_eff = |k1 - k2| for counter-propagating
        gamma_doppler = trap_noise.get('gamma_phi_doppler', 0.0)
        
        # Intensity noise dephasing from trap laser fluctuations  
        # This is a key tweezer-power-dependent noise source!
        # γ_intensity = (∂ω/∂I) × σ_I = (U₀/ℏ) × (ΔI/I)
        gamma_intensity = trap_noise.get('gamma_phi_intensity', 0.0)
        
        # Zeeman dephasing from B-field noise (small for clock states)
        # Use zeeman_dephasing_rate from noise_models for consistency
        B_rms_gauss = max(0.01 * B_field * 1e4, 1e-3)  # Convert T to Gauss, min 1 mG
        qubit_type = "clock" if config.is_clock_transition else "stretched"
        K_quad = 575.0 if config.species == "Rb87" else 427.0  # Hz/G²
        gamma_zeeman = zeeman_dephasing_rate(B_rms_gauss, qubit_type, K_quad)
        
        # Anti-trap loss: Rydberg state is repelled from tweezer
        # Scale by Rydberg population fraction and gate time dependence
        rydberg_fraction = 0.3
        t_reference = 1e-6
        time_factor = min(1.0, (tau_total / t_reference)**2)
        gamma_antitrap_effective = trap_noise['gamma_loss_antitrap'] * rydberg_fraction * time_factor
        
        # Spectral leakage: off-resonant excitation to |n±1⟩ or |nP⟩ states
        # This is INCOHERENT leakage: population leaks, decays, doesn't return
        Delta_leak = compute_leakage_detuning(config.species, config.n_rydberg)
        gamma_leakage = leakage_rate_to_adjacent_states(
            Omega=Omega, 
            Delta_leak=Delta_leak, 
            pulse_shape=pulse_shape, 
            tau=tau_single,
            gamma_rydberg=trap_noise['gamma_r'],  # Pass Rydberg decay rate
        )
        
        # Combine all thermal/motional dephasing sources
        # These add in quadrature for uncorrelated noise:
        # γ_total² = γ_blockade² + γ_doppler² + γ_intensity²
        # But for Markovian approximation, we add rates linearly (conservative)
        gamma_thermal_total = gamma_motional + gamma_doppler + gamma_intensity
        
        # =====================================================================
        # mJ leakage rate from polarization impurity
        # =====================================================================
        # Impure laser polarization can drive transitions to wrong mJ states:
        #   σ⁺ → Δm = +1, σ⁻ → Δm = -1, π → Δm = 0
        # If σ⁺ polarization has ε_pol fraction of σ⁻, population leaks to |r-⟩.
        # Rate: γ_mJ = ε_pol² × Ω² / Δ_Zeeman  (off-resonant driving)
        # 
        # For 4-level Hilbert space (|0⟩, |1⟩, |r+⟩, |r-⟩), this is a key error.
        if hilbert_space_dim == 4:
            # Calculate Zeeman splitting between |r+⟩ and |r-⟩
            # For S₁/₂ state: ΔE_Zeeman = g_J × μ_B × B
            Delta_zeeman_rydberg = rydberg_zeeman_splitting(B_field, L=0, J=0.5)
            
            # Combined polarization purity (both lasers contribute)
            # For two-photon: effective impurity adds
            combined_polarization_purity = min(polarization_purity_1, polarization_purity_2)
            
            # Calculate mJ leakage rate
            gamma_mJ = mJ_mixing_rate(
                Omega_eff=Omega,
                polarization_purity=combined_polarization_purity,
                Delta_zeeman=Delta_zeeman_rydberg
            )
        else:
            # 3-level model doesn't track mJ substates
            gamma_mJ = 0.0
            Delta_zeeman_rydberg = 0.0
            combined_polarization_purity = 1.0
        
        # Build collapse operators
        c_ops, noise_dict = build_all_noise_operators(
            hs=hs,
            gamma_r=trap_noise['gamma_r'],
            gamma_bbr=trap_noise.get('gamma_bbr', 0),
            gamma_phi_laser=gamma_phi,
            gamma_phi_thermal=gamma_thermal_total,  # Combined motional dephasing
            gamma_phi_zeeman=gamma_zeeman,
            gamma_loss_antitrap=gamma_antitrap_effective,
            gamma_loss_background=gamma_loss_background,
            gamma_scatter_intermediate=trap_noise['gamma_scatter_intermediate'],
            gamma_leakage=gamma_leakage,
            mJ_leakage_rate=gamma_mJ,
        )
        
        noise_breakdown.update(noise_dict)
        noise_breakdown['gamma_blockade_fluct'] = gamma_blockade_fluct
        noise_breakdown['gamma_doppler'] = gamma_doppler
        noise_breakdown['gamma_intensity_noise'] = gamma_intensity
        noise_breakdown['gamma_thermal_total'] = gamma_thermal_total
        noise_breakdown['delta_V_over_V_percent'] = trap_noise['blockade_fluctuation_percent']
        noise_breakdown['anti_trap_time_factor'] = time_factor
        noise_breakdown['magic_enhancement'] = trap_noise['magic_enhancement']
        noise_breakdown['alpha_ratio'] = trap_noise['alpha_ratio']
        # Add Doppler physics info for debugging
        noise_breakdown['k_eff_rad_per_m'] = trap_noise.get('k_eff_rad_per_m', 0)
        noise_breakdown['v_thermal_m_per_s'] = trap_noise.get('v_thermal_m_per_s', 0)
        # Add polarization-related noise info
        noise_breakdown['gamma_mJ_leakage'] = gamma_mJ
        noise_breakdown['polarization_1'] = pol_1
        noise_breakdown['polarization_2'] = pol_2
        noise_breakdown['polarization_purity_1'] = polarization_purity_1
        noise_breakdown['polarization_purity_2'] = polarization_purity_2
        noise_breakdown['combined_polarization_purity'] = combined_polarization_purity
        noise_breakdown['Delta_zeeman_rydberg_Hz'] = Delta_zeeman_rydberg / (2 * np.pi)
    
    # =========================================================================
    # STEP 9: Setup initial states
    # =========================================================================
    # We simulate all four computational basis states |00⟩, |01⟩, |10⟩, |11⟩
    # to compute the full process fidelity. The CZ gate action is:
    #   |00⟩ → |00⟩, |01⟩ → |01⟩, |10⟩ → |10⟩, |11⟩ → -|11⟩
    # The -1 phase on |11⟩ comes from the Rydberg blockade mechanism.
    if hilbert_space_dim == 3:
        initial_states = {
            "00": tensor(b0, b0),
            "01": tensor(b0, b1),
            "10": tensor(b1, b0),
            "11": tensor(b1, b1),
        }
    else:
        initial_states = {
            "00": tensor(b0_4, b0_4),
            "01": tensor(b0_4, b1_4),
            "10": tensor(b1_4, b0_4),
            "11": tensor(b1_4, b1_4),
        }
    
    # =========================================================================
    # STEP 10: Run simulation
    # =========================================================================
    # We use QuTiP's mesolve() to integrate the Lindblad master equation:
    #   dρ/dt = -i[H,ρ] + Σⱼ γⱼ(LⱼρLⱼ† - ½{Lⱼ†Lⱼ, ρ})
    # Different protocols use different time-evolution strategies:
    #   - LP square: Two constant-H pulses via mesolve
    #   - LP shaped: Time-dependent H(t) with Gaussian/DRAG envelope
    #   - JP bang-bang: Piecewise-constant H with phase jumps at switching times
    #   - Smooth JP: Time-stepping with smooth sinusoidal phase modulation
    results_output = {}
    
    pulse_info = {
        'shape': pulse_shape if not is_jp_protocol else ('smooth_sinusoidal' if is_smooth_jp else 'jp_phase_modulated'),
        'implementation': 'time_dependent_hamiltonian' if is_jp_protocol else 'constant_hamiltonian',
        'delta_zeeman': delta_zeeman,
        'delta_stark': delta_stark,
        'trap_laser_on': trap_laser_on,
    }
    
    if is_jp_bangbang:
        # =====================================================================
        # JANDURA-PUPILLO BANG-BANG CZ GATE
        # =====================================================================
        # Piecewise-constant phase modulation: φ(t) switches between 
        # ±π/2, 0 at optimized switching times. No static detuning.
        # Reference: Jandura & Pupillo, PRX Quantum 3, 010353 (2022)
        
        if verbose:
            print(f"  JP bang-bang → piecewise-constant phase CZ gate")
        
        results_output, bb_info = evolve_bangbang_jp(
            initial_states=initial_states,
            Omega=Omega,
            V=V,
            omega_tau=_omega_tau,
            switching_times=_bb_switching_times,
            phases=_bb_phases,
            hs=hs,
            c_ops=c_ops,
            delta_zeeman=delta_zeeman,
            delta_stark=delta_stark,
            trap_laser_on=trap_laser_on,
            n_steps_per_segment=50,
            verbose=verbose,
        )
        
        # Update pulse_info with bang-bang specifics
        pulse_info['shape'] = 'bangbang'
        pulse_info['implementation'] = 'piecewise_constant_hamiltonian'
        pulse_info['protocol_variant'] = 'jandura_pupillo_bangbang'
        pulse_info['switching_times'] = list(_bb_switching_times)
        pulse_info['phases'] = list(_bb_phases)
        pulse_info['n_segments'] = len(_bb_phases)
        pulse_info['omega_tau'] = _omega_tau
        
        # tau_total already set in Step 5
        
    elif is_smooth_jp:
        # =====================================================================
        # BLUVSTEIN/EVERED SMOOTH SINUSOIDAL CZ (DARK STATE PHYSICS)
        # =====================================================================
        # This is the HIGH-FIDELITY (99.5%) CZ gate from Evered et al., Nature 2023.
        #
        # Key physics:
        # - Phase modulation: φ(t) = A·cos(ω·t - φ_offset) + δ₀·t
        # - Dark state: OPPOSITE signs for Δₑ (intermediate) and δ (two-photon)
        # - Suppresses intermediate-state scattering by maximizing dark state pop
        #
        # The two-photon detuning δ is CRITICAL for fidelity!
        
        if verbose:
            print(f"  Smooth JP → Bluvstein/Evered dark state CZ gate")
        
        # Get smooth JP parameters: prefer values from simulation_inputs,
        # fall back to protocol lookup table defaults
        smooth_params = get_protocol_params("smooth_jp", V_over_Omega=V_over_Omega)
        
        # Extract parameters - simulation_inputs fields override defaults.
        # Use getattr() so this works for both SmoothJPSimulationInputs
        # (which has A, omega_mod_ratio, etc.) and JPSimulationInputs
        # (which only has omega_tau, switching_times, phases).
        _si = simulation_inputs  # shorthand
        A = getattr(_si, 'A', None) or smooth_params.get('A', 0.311 * np.pi)
        omega_mod_ratio = getattr(_si, 'omega_mod_ratio', None) or smooth_params.get('omega_mod_ratio', 1.242)
        phi_offset = getattr(_si, 'phi_offset', None) or smooth_params.get('phi_offset', 4.696)
        _raw_delta = getattr(_si, 'delta_over_omega', None)
        smooth_delta_over_omega_magnitude = abs(_raw_delta if _raw_delta is not None else smooth_params.get('delta_over_omega', 0.0205))
        _raw_omega_tau = getattr(_si, 'omega_tau', None)
        smooth_omega_tau = _raw_omega_tau if _raw_omega_tau is not None else smooth_params.get('omega_tau', 10.09)
        
        # Calculate gate time from smooth JP parameters
        smooth_tau_total = smooth_omega_tau / Omega
        
        # CRITICAL: Dark state physics requires opposite signs for Δₑ and δ
        # The validated parameters assume a specific sign convention.
        # For positive Δₑ (blue detuning): δ must be NEGATIVE
        # For negative Δₑ (red detuning): δ must be POSITIVE
        intermediate_detuning_sign = "positive" if Delta_e > 0 else "negative"
        if Delta_e > 0:
            # Blue intermediate detuning → need negative two-photon detuning
            smooth_delta_over_omega = -smooth_delta_over_omega_magnitude
        else:
            # Red intermediate detuning → need positive two-photon detuning  
            smooth_delta_over_omega = +smooth_delta_over_omega_magnitude
        
        results_output, smooth_info = evolve_smooth_sinusoidal_jp(
            initial_states=initial_states,
            Omega=Omega,
            V=V,
            tau_total=smooth_tau_total,
            hs=hs,
            c_ops=c_ops,
            A=A,
            omega_mod=omega_mod_ratio * Omega,
            phi_offset=phi_offset,
            delta_over_omega=smooth_delta_over_omega,
            n_steps=300,
            delta_zeeman=delta_zeeman,
            delta_stark=delta_stark,
            trap_laser_on=trap_laser_on,
            intermediate_detuning_sign=intermediate_detuning_sign,
            verbose=verbose,
        )
        
        # Update pulse_info with smooth JP specifics
        pulse_info['protocol_variant'] = 'bluvstein_evered_dark_state'
        pulse_info['A'] = A
        pulse_info['omega_mod_ratio'] = omega_mod_ratio
        pulse_info['phi_offset'] = phi_offset
        pulse_info['delta_over_omega'] = smooth_delta_over_omega
        pulse_info['dark_state_valid'] = smooth_info.get('dark_state_valid', True)
        
        # Override tau_total for fidelity calculations
        tau_total = smooth_tau_total
        tau_single = tau_total  # Single pulse protocol
        
    elif pulse_shape != "square":
        # =====================================================================
        # LEVINE-PICHLER WITH SHAPED PULSES
        # =====================================================================
        # LP with shaped pulses: Gaussian, cosine, blackman, or DRAG envelope
        # reduces spectral leakage to non-computational states.
        # DRAG adds derivative correction: Ω(t) → Ω(t) - i·λ·dΩ/dt
        #
        # NOTE: Pulse shaping is ONLY supported for LP protocol.
        # JP protocol always uses bang-bang phase control (above).
        results_output, sp_info = evolve_shaped_pulse(
            initial_states=initial_states,
            pulse_shape=pulse_shape,
            Omega=Omega,
            Delta=Delta_gate,
            V=V,
            xi=xi,
            tau_single=tau_single,
            hs=hs,
            c_ops=c_ops,
            delta_zeeman=delta_zeeman,
            delta_stark=delta_stark,
            trap_laser_on=trap_laser_on,
            drag_lambda=drag_lambda if pulse_shape == 'drag' else None,
            verbose=verbose,
        )
        
        pulse_info['drag_lambda'] = sp_info.get('drag_lambda')
        
    else:
        # =====================================================================
        # LEVINE-PICHLER WITH SQUARE PULSES
        # =====================================================================
        # LP with square pulses: simplest case, constant H for each pulse
        # Two pulses with phase jump ξ between them:
        #   Pulse 1: H = (Ω/2)(|1⟩⟨r| + h.c.) - Δ|r⟩⟨r| + V|rr⟩⟨rr|
        #   Pulse 2: Same but Ω → Ω·e^{iξ}
        results_output = evolve_two_pulse_lp(
            initial_states=initial_states,
            H1=H1,
            H2=H2,
            tau_single=tau_single,
            c_ops=c_ops,
        )
    
    # =========================================================================
    # STEP 11: Compute fidelity
    # =========================================================================
    # CZ fidelity is computed by comparing final states to ideal CZ action.
    # We extract global phase (e^{iφ}) since it's not physically observable.
    # Average fidelity F_avg = (1/4)Σ|⟨ψ_ideal|ψ_final⟩|² for pure states,
    # or F = Tr(ρ_ideal · ρ_final) for mixed states.
    fidelities, avg_fidelity, phase_info = compute_CZ_fidelity(
        results_output,
        extract_global_phase=True,
        hilbert_space_dim=hilbert_space_dim,
    )
    
    # Fidelity is computed purely from quantum state comparison.
    # No artificial penalties - the Hamiltonian evolution naturally
    # produces low fidelity when blockade is weak.
    
    # =========================================================================
    # STEP 12: Package results
    # =========================================================================
    # We return a comprehensive SimulationResult containing:
    #   - Gate performance metrics (fidelity, phases)
    #   - Physical parameters (Ω, V, Δ, R, τ)
    #   - Trap properties (U₀, ωᵣ, σᵣ, magic wavelength)
    #   - Noise budget breakdown (all γ rates)
    #   - Configuration and environment settings
    
    # Build result dictionary (for both dict and dataclass return)
    result_dict = {
        # Gate performance
        'avg_fidelity': avg_fidelity,
        'fidelities': fidelities,
        'phase_info': phase_info,
        
        # Protocol info
        'protocol': protocol_params['name'],
        'n_pulses': n_pulses,
        'hilbert_space_dim': hilbert_space_dim,
        
        # Rydberg parameters
        'Omega': Omega,
        'V': V,
        'Delta': Delta_gate,
        'V_over_Omega': V / Omega if Omega > 0 else float('inf'),
        'Delta_over_Omega': _delta_over_omega,
        
        # Timing
        'tau_single': tau_single,
        'tau_total': tau_total,
        'xi': xi,
        
        # Geometry
        'R': R,
        'spacing_factor': spacing_factor,
        
        # Trap properties
        'U0_mK': U0 / KB * 1e3,
        'omega_r_kHz': omega_r / (2 * np.pi * 1e3),
        'sigma_r_nm': sigma_r * 1e9,
        'trap_wavelength_nm': wavelength_nm,
        'magic_wavelength_analysis': magic_analysis,
        
        # Noise budget
        'noise_breakdown': noise_breakdown,
        'include_noise': include_noise,
        'include_motional_dephasing': include_motional_dephasing,
        
        # Pulse shaping
        'pulse_info': pulse_info,
        
        # Configuration
        'config': config,
        'species': config.species,
        'n_rydberg': config.n_rydberg,
        'qubit_0': config.qubit_0,
        'qubit_1': config.qubit_1,
        
        # Environment
        'temperature_K': temperature,
        'B_field_T': B_field,
        
        # Coherent frequency shifts
        'delta_zeeman': delta_zeeman,
        'delta_stark': delta_stark,
        'trap_laser_on': trap_laser_on,
        
        # Raw outputs
        'results': results_output,
        'H1': H1,
        'H2': H2,
        'c_ops': c_ops,
        'hs': hs,
    }
    
    if return_dataclass:
        return SimulationResult(**result_dict)
    else:
        # Legacy dict return with extra convenience keys
        result_dict.update({
            'Omega_rad_per_s': Omega,
            'Omega_MHz': Omega / (2 * np.pi * 1e6),
            'V_rad_per_s': V,
            'V_MHz': V / (2 * np.pi * 1e6),
            'Delta_rad_per_s': Delta_gate,
            'Delta_MHz': Delta_gate / (2 * np.pi * 1e6),
            'xi_rad': np.angle(xi) if n_pulses == 2 else 0.0,
            'xi_deg': np.degrees(np.angle(xi)) if n_pulses == 2 else 0.0,
            'tau_single_us': tau_single * 1e6,
            'tau_total_us': tau_total * 1e6,
            'gate_time_us': tau_total * 1e6,
            'R_meters': R,
            'R_um': R * 1e6,
            'temperature_uK': temperature * 1e6,
            'B_field_Gauss': B_field * 1e4,
        })
        return result_dict



# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Basis states
    "b0", "b1", "br",
    "b0_4", "b1_4", "br_plus", "br_minus",
    
    # Fidelity computation
    "compute_state_fidelity",
    "compute_CZ_fidelity",
    
    # Evolution helpers
    "evolve_state",
    "evolve_two_pulse",
    "evolve_two_pulse_lp",
    "evolve_smooth_sinusoidal_jp",  # Bluvstein/Evered dark state CZ gate
    "evolve_bangbang_jp",           # Jandura-Pupillo bang-bang phase CZ gate
    "evolve_shaped_pulse",
    
    # Main simulation
    "SimulationResult",
    "simulate_CZ_gate",
]
