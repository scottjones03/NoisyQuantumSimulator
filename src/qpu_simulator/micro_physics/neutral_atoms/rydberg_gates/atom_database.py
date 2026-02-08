"""
Atomic Species Database for Rydberg Gate Simulations
=====================================================

This module provides a comprehensive database of atomic properties for Rb87 and Cs133,
the two most common species used in neutral atom quantum computing.

WHAT'S IN THIS DATABASE?
------------------------

1. **Fundamental properties**: Mass, nuclear spin, ionization energy
2. **Quantum defects**: Corrections to hydrogen-like energy levels (δ_S, δ_P, δ_D, δ_F)
3. **Hyperfine structure**: Qubit states live in the ground state hyperfine manifold
4. **Rydberg properties**: C₆ coefficients, lifetimes, polarizabilities
5. **Scaling laws**: How properties change with principal quantum number n

WHY DO WE NEED ALL THIS DATA?
-----------------------------

Accurate gate simulation requires precise atomic data. Let me give you some examples:

**Example 1: Computing Rabi frequency**
    The Rabi frequency (how fast we drive transitions) depends on the dipole matrix
    element: Ω = d·E₀/ℏ. If we get the dipole moment wrong by 10%, our Rabi frequency
    is wrong by 10%, and our gate time is wrong by 10%!

**Example 2: Blockade strength**
    The Rydberg blockade V = C₆/R⁶ determines whether two atoms can both be excited.
    If C₆ is wrong, we might think we have a good blockade when we don't, leading
    to gate errors. C₆ scales as n¹¹, so getting n slightly wrong is a big deal!

**Example 3: Noise estimates**
    Spontaneous emission rate depends on the Rydberg lifetime: γ_r = 1/τ.
    If τ is wrong by 2×, our fidelity estimates are completely off.

QUANTUM DEFECT THEORY
---------------------

For hydrogen, energy levels are E_n = -Ry/n². For Rb and Cs, the valence electron
sometimes penetrates the inner electron core, seeing more than +1e charge. We
correct for this with the "quantum defect" δ_L:

    E_n = -Ry/(n - δ_L)² = -Ry/n*²

where n* = n - δ_L is the "effective principal quantum number".

Different orbital angular momentum states (S, P, D, F) have different quantum defects:
- S states (L=0): Largest defect (~3.1 for Rb) - most core penetration
- P states (L=1): Medium defect (~2.7 for Rb)
- D states (L=2): Smaller defect (~1.3 for Rb)
- F states (L=3): Nearly zero (~0.02 for Rb) - almost hydrogenic

**Physical picture**: Think of the electron orbiting on an ellipse. S orbitals are
"radially" oriented and plunge deep into the core. F orbitals are more "circular"
and stay far from the core.

SCALING LAWS
------------

Rydberg properties scale as power laws in n*:

| Property        | Scaling | Why?                                    |
|-----------------|---------|------------------------------------------|
| Orbital radius  | n*²     | Bohr model: r = n²a₀                    |
| Binding energy  | n*⁻²    | E = -Ry/n*²                             |
| Lifetime (0K)   | n*³     | τ ∝ ω⁻³|d|² ∝ n³                        |
| C₆ coefficient  | n*¹¹    | C₆ ∝ α² and α ∝ n*⁷ (roughly)          |
| Polarizability  | n*⁷     | α ∝ ⟨r²⟩ × (dipole)² ~ n⁴ × n³        |
| Dipole (P→nS)   | n*⁻³/²  | Wavefunction overlap decreases with n  |

**Why does C₆ scale so steeply?**
Van der Waals interactions arise from fluctuating dipoles. The polarizability
of a Rydberg atom scales as n⁷, and C₆ ∝ α² would give n¹⁴. The actual n¹¹
comes from details of the dispersion sum rules.

PRACTICAL CONSEQUENCE:
At n=53 (Bluvstein), C₆ ≈ 34 GHz·μm⁶
At n=70 (common choice), C₆ ≈ 860 GHz·μm⁶
That's 25× stronger interaction for going from n=53 to n=70!

References
----------
[1] Li et al., Phys. Rev. A 67, 052502 (2003) - Quantum defects
[2] Saffman et al., Rev. Mod. Phys. 82, 2313 (2010) - Rydberg physics review
[3] Steck, "Rubidium 87 D Line Data" (2021) - Rb87 atomic data
[4] Beterov et al., Phys. Rev. A 79, 052504 (2009) - Rydberg lifetimes
[5] Zhang et al., Phys. Rev. A 84, 043408 (2011) - Rydberg polarizabilities
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import numpy as np

from .constants import (
    HBAR, EPS0, C, E_CHARGE, A0, KB, MU_B,
    RY_JOULES, G_I_RB87, G_I_CS133, G_E
)


# =============================================================================
# THE MAIN ATOMIC DATABASE
# =============================================================================

ATOM_DB = {
    # =========================================================================
    # RUBIDIUM-87
    # =========================================================================
    # Rb87 is THE workhorse of cold atom experiments:
    # - Laser cooling on D2 line (780 nm) is technically mature
    # - Large C₆ coefficients give strong Rydberg blockade
    # - Nuclear spin I=3/2 gives simple hyperfine structure
    # - Extensive literature data available
    
    "Rb87": {
        # ---------------------------------------------------------------------
        # FUNDAMENTAL PROPERTIES
        # ---------------------------------------------------------------------
        "mass": 1.443160648e-25,    # kg (= 86.909 atomic mass units)
        "nuclear_spin": 1.5,         # I = 3/2
        "g_I": G_I_RB87,             # Nuclear g-factor (for Zeeman shifts)
        
        # Ionization energy: minimum energy to remove the valence electron
        # All Rydberg state energies are measured relative to this
        "E_ionization": 4.177128 * E_CHARGE,  # J (= 4.177 eV)
        
        # ---------------------------------------------------------------------
        # QUANTUM DEFECTS
        # ---------------------------------------------------------------------
        # These determine Rydberg state energies: E_n = -Ry/(n - δ_L)²
        # 
        # Physical meaning: S states have δ ≈ 3.1, meaning the n=70 S-state
        # has the same energy as hydrogen's n=66.9 state. The effective
        # principal quantum number is n* = n - δ = 66.9.
        #
        # These values are from precision microwave spectroscopy:
        # Li et al., Phys. Rev. A 67, 052502 (2003)
        
        "quantum_defects": {
            "S": 3.1311807,   # nS₁/₂ states - largest defect (most core penetration)
            "P": 2.6548849,   # nP states
            "D": 1.3480917,   # nD states  
            "F": 0.0165192,   # nF states - nearly hydrogenic (minimal core overlap)
        },
        
        # ---------------------------------------------------------------------
        # HYPERFINE STRUCTURE (where qubits live)
        # ---------------------------------------------------------------------
        # The 5S₁/₂ ground state splits into F=1 and F=2 manifolds
        # due to coupling between electron spin (S=1/2) and nuclear spin (I=3/2).
        #
        # Total angular momentum: F = I + J, so F = 1 or 2 (|I-J| to I+J)
        # Each F level has 2F+1 magnetic sublevels: mF = -F, -F+1, ..., +F
        #
        # COMMON QUBIT ENCODINGS:
        # 1. "Clock states": |0⟩ = |F=1, mF=0⟩, |1⟩ = |F=2, mF=0⟩
        #    - First-order insensitive to B-field (only quadratic Zeeman)
        #    - Separation: 6.835 GHz (microwave transition)
        #
        # 2. "Stretched states": |0⟩ = |F=2, mF=-2⟩, |1⟩ = |F=2, mF=-1⟩
        #    - Allows cycling transition for detection
        #    - More sensitive to B-field fluctuations
        
        "hyperfine_ground": {
            # F=1 manifold (lower energy)
            # Energy relative to hyperfine center of gravity
            (1, -1): {
                "energy_B0": -4.271676631815181e9 * HBAR * 2*np.pi,
                "label": "|5S₁/₂, F=1, mF=-1⟩"
            },
            (1,  0): {
                "energy_B0": -4.271676631815181e9 * HBAR * 2*np.pi,
                "label": "|5S₁/₂, F=1, mF=0⟩"
            },
            (1, +1): {
                "energy_B0": -4.271676631815181e9 * HBAR * 2*np.pi,
                "label": "|5S₁/₂, F=1, mF=+1⟩"
            },
            # F=2 manifold (higher energy by 6.835 GHz)
            (2, -2): {
                "energy_B0": 2.563005979089109e9 * HBAR * 2*np.pi,
                "label": "|5S₁/₂, F=2, mF=-2⟩"
            },
            (2, -1): {
                "energy_B0": 2.563005979089109e9 * HBAR * 2*np.pi,
                "label": "|5S₁/₂, F=2, mF=-1⟩"
            },
            (2,  0): {
                "energy_B0": 2.563005979089109e9 * HBAR * 2*np.pi,
                "label": "|5S₁/₂, F=2, mF=0⟩"
            },
            (2, +1): {
                "energy_B0": 2.563005979089109e9 * HBAR * 2*np.pi,
                "label": "|5S₁/₂, F=2, mF=+1⟩"
            },
            (2, +2): {
                "energy_B0": 2.563005979089109e9 * HBAR * 2*np.pi,
                "label": "|5S₁/₂, F=2, mF=+2⟩"
            },
        },
        
        # Landé g-factors: determine Zeeman shift strength
        # Zeeman shift: ΔE = g_F × μ_B × B × mF
        "g_F": {
            1: -0.5,   # F=1: electron and nucleus spins anti-aligned
            2: +0.5,   # F=2: electron and nucleus spins aligned
        },
        
        # Quadratic Zeeman coefficient for clock transition
        # For |F=1,mF=0⟩ ↔ |F=2,mF=0⟩: ΔE = 575 Hz × B²(Gauss)
        "K_quad_clock": 575.0,  # Hz/G²
        
        # ---------------------------------------------------------------------
        # INTERMEDIATE EXCITED STATES (for two-photon excitation)
        # ---------------------------------------------------------------------
        # To excite to Rydberg states, we use two lasers:
        # Ground (5S₁/₂) → Intermediate (5P) → Rydberg (nS or nD)
        #
        # The intermediate state is DETUNED to avoid scattering.
        # Common choice: 5P₃/₂ (D2 line at 780 nm) + blue laser to nS
        
        "intermediate_states": {
            "5P1/2": {  # D1 line (795 nm)
                "energy": 377.107385690e12 * HBAR * 2*np.pi,  # J
                "linewidth": 2*np.pi * 5.746e6,  # Hz (natural linewidth Γ)
                "dipole_from_ground": 2.99 * E_CHARGE * A0,  # reduced dipole
                "g_J": 2/3,  # Landé g-factor
            },
            "5P3/2": {  # D2 line (780 nm) - MOST COMMON CHOICE
                "energy": 384.230484468e12 * HBAR * 2*np.pi,  # J
                "linewidth": 2*np.pi * 6.065e6,  # Hz (= 6.07 MHz)
                "dipole_from_ground": 4.23 * E_CHARGE * A0,  # reduced dipole
                "g_J": 4/3,  # Landé g-factor
            },
        },
        
        # ---------------------------------------------------------------------
        # RYDBERG REFERENCE VALUES (at n_ref = 70)
        # ---------------------------------------------------------------------
        # All n-dependent properties are scaled from these reference values.
        # We store measured/calculated values at n=70, then use scaling laws.
        #
        # Why n=70? It's a "sweet spot":
        # - C₆ ~ 860 GHz·μm⁶: strong enough for fast gates
        # - τ ~ 140 μs: long enough for gate operation
        # - Not too sensitive to stray electric fields
        
        "n_ref": 70,
        
        # C₆ coefficient: determines Rydberg-Rydberg interaction strength
        # V(R) = C₆/R⁶ is the van der Waals interaction
        # At R = 4 μm: V = 860 GHz·μm⁶ / (4μm)⁶ = 210 MHz = STRONG BLOCKADE!
        # Source: Saffman et al., Rev. Mod. Phys. 82, 2313 (2010), Table IV
        "C6_ref": 2 * np.pi * 862.69e9 * (1e-6)**6,  # J·m⁶
        
        # Rydberg state lifetime
        # Limited by: (1) spontaneous emission to lower states
        #             (2) blackbody radiation-induced transitions
        # At T=0K: only spontaneous emission matters
        # At T=300K: BBR adds significant decay (roughly doubles rate)
        "tau_ref": 140e-6,     # seconds at n=70, T=300K
        "tau_0K_ref": 280e-6,  # seconds at n=70, T=0K (no BBR)
        
        # Polarizability at 1064 nm trap wavelength
        # Ground state: pulled TOWARD intensity maxima (attractive trap)
        # Rydberg state: pushed AWAY from intensity maxima (anti-trapping!)
        # This mismatch causes problems - atom can escape during gate
        "alpha_ground": 687.3 * 4 * np.pi * EPS0 * A0**3,  # SI units
        "alpha_rydberg_ref": -200000 * 4 * np.pi * EPS0 * A0**3,  # SI, NEGATIVE!
        
        # Per-F-level polarizabilities (for differential AC Stark shift)
        "alpha_hyperfine": {
            1: 686.1 * 4 * np.pi * EPS0 * A0**3,  # F=1
            2: 688.5 * 4 * np.pi * EPS0 * A0**3,  # F=2 (0.34% larger)
        },
        
        # ---------------------------------------------------------------------
        # DIPOLE: INTERMEDIATE → RYDBERG TRANSITION
        # ---------------------------------------------------------------------
        # This is the dipole matrix element for the SECOND leg of two-photon excitation:
        #   Ground (5S₁/₂) → Intermediate (5P₃/₂) → Rydberg (nS₁/₂)
        #                    └── first leg ──┘    └── second leg ──┘
        #
        # For Rb87: ⟨nS₁/₂|er|5P₃/₂⟩ - transition from 5P to Rydberg nS
        # The 5P state is the D2 line intermediate at 780 nm
        #
        # This scales as n*^(-3/2) due to decreasing wavefunction overlap
        # at higher n (Rydberg electron spreads out more)
        "dipole_intermediate_to_rydberg_ref": 0.014 * E_CHARGE * A0,  # ⟨70S|er|5P⟩
        
        # ---------------------------------------------------------------------
        # SCALING EXPONENTS
        # ---------------------------------------------------------------------
        # Property(n) = Property(n_ref) × (n*/n*_ref)^exponent
        # where n* = n - δ is the effective principal quantum number
        
        "scaling_exponents": {
            "C6": 11,                 # C₆ ∝ n*¹¹ (STEEP!)
            "lifetime_0K": 3,         # τ(0K) ∝ n*³
            "lifetime_BBR": 2,        # BBR contribution ∝ n*²
            "polarizability": 7,      # α_rydberg ∝ n*⁷
            "dipole_to_rydberg": -1.5,  # ⟨nS|r|5P⟩ ∝ n*⁻³/²
        },
        
        # ---------------------------------------------------------------------
        # TRANSITION FREQUENCIES
        # ---------------------------------------------------------------------
        "transitions": {
            "ground_to_5P3/2": 384.230484468e12,  # Hz (D2 line, 780 nm)
            "ground_to_5P1/2": 377.107385690e12,  # Hz (D1 line, 795 nm)
        },
        
        # Default trap wavelength
        "trap_wavelength": 1064e-9,  # m (Nd:YAG laser)
        
        # ---------------------------------------------------------------------
        # MAGIC WAVELENGTHS
        # ---------------------------------------------------------------------
        "magic_wavelengths": {
            "hyperfine": {
                "scalar_magic_nm": 790.0,  # Between D1 and D2
                "1064nm_differential_Hz_per_mK": 70e3,  # Differential shift
            },
            "ground_rydberg": {
                "near_magic_nm": 1004,  # Approximate magic for n~50-60
            },
        },
    },
    
    # =========================================================================
    # CESIUM-133
    # =========================================================================
    # Cs133 is used in some experiments:
    # - Larger C₆ coefficients (stronger interactions)
    # - Larger mass (slower thermal motion)
    # - More complex hyperfine structure (I=7/2)
    
    "Cs133": {
        "mass": 2.20694657e-25,  # kg (= 132.905 amu)
        "nuclear_spin": 3.5,     # I = 7/2
        "g_I": G_I_CS133,
        "E_ionization": 3.8939 * E_CHARGE,  # J
        
        "quantum_defects": {
            "S": 4.0493532,  # Larger than Rb (bigger atomic core)
            "P": 3.5915871,
            "D": 2.4754562,
            "F": 0.0334,
        },
        
        "hyperfine_ground": {
            # F=3 manifold (lower energy)
            (3, -3): {"energy_B0": -4.021776399375e9 * HBAR * 2*np.pi, 
                      "label": "|6S₁/₂, F=3, mF=-3⟩"},
            (3, -2): {"energy_B0": -4.021776399375e9 * HBAR * 2*np.pi,
                      "label": "|6S₁/₂, F=3, mF=-2⟩"},
            (3, -1): {"energy_B0": -4.021776399375e9 * HBAR * 2*np.pi,
                      "label": "|6S₁/₂, F=3, mF=-1⟩"},
            (3,  0): {"energy_B0": -4.021776399375e9 * HBAR * 2*np.pi,
                      "label": "|6S₁/₂, F=3, mF=0⟩"},
            (3, +1): {"energy_B0": -4.021776399375e9 * HBAR * 2*np.pi,
                      "label": "|6S₁/₂, F=3, mF=+1⟩"},
            (3, +2): {"energy_B0": -4.021776399375e9 * HBAR * 2*np.pi,
                      "label": "|6S₁/₂, F=3, mF=+2⟩"},
            (3, +3): {"energy_B0": -4.021776399375e9 * HBAR * 2*np.pi,
                      "label": "|6S₁/₂, F=3, mF=+3⟩"},
            # F=4 manifold (higher energy by 9.193 GHz)
            (4, -4): {"energy_B0": 5.170855370625e9 * HBAR * 2*np.pi,
                      "label": "|6S₁/₂, F=4, mF=-4⟩"},
            (4, -3): {"energy_B0": 5.170855370625e9 * HBAR * 2*np.pi,
                      "label": "|6S₁/₂, F=4, mF=-3⟩"},
            (4, -2): {"energy_B0": 5.170855370625e9 * HBAR * 2*np.pi,
                      "label": "|6S₁/₂, F=4, mF=-2⟩"},
            (4, -1): {"energy_B0": 5.170855370625e9 * HBAR * 2*np.pi,
                      "label": "|6S₁/₂, F=4, mF=-1⟩"},
            (4,  0): {"energy_B0": 5.170855370625e9 * HBAR * 2*np.pi,
                      "label": "|6S₁/₂, F=4, mF=0⟩"},
            (4, +1): {"energy_B0": 5.170855370625e9 * HBAR * 2*np.pi,
                      "label": "|6S₁/₂, F=4, mF=+1⟩"},
            (4, +2): {"energy_B0": 5.170855370625e9 * HBAR * 2*np.pi,
                      "label": "|6S₁/₂, F=4, mF=+2⟩"},
            (4, +3): {"energy_B0": 5.170855370625e9 * HBAR * 2*np.pi,
                      "label": "|6S₁/₂, F=4, mF=+3⟩"},
            (4, +4): {"energy_B0": 5.170855370625e9 * HBAR * 2*np.pi,
                      "label": "|6S₁/₂, F=4, mF=+4⟩"},
        },
        
        "g_F": {
            3: -0.25,  # |g_F| smaller than Rb due to larger I
            4: +0.25,
        },
        "K_quad_clock": 427.0,  # Hz/G²
        
        "intermediate_states": {
            "6P1/2": {  # D1 line (894 nm)
                "energy": 335.116048807e12 * HBAR * 2*np.pi,
                "linewidth": 2*np.pi * 4.575e6,
                "dipole_from_ground": 3.18 * E_CHARGE * A0,
                "g_J": 2/3,
            },
            "6P3/2": {  # D2 line (852 nm)
                "energy": 351.725718509e12 * HBAR * 2*np.pi,
                "linewidth": 2*np.pi * 5.234e6,
                "dipole_from_ground": 4.49 * E_CHARGE * A0,
                "g_J": 4/3,
            },
        },
        
        "alpha_ground": 1000 * 4 * np.pi * EPS0 * A0**3,  # at 1064nm
        "alpha_rydberg_ref": -300000 * 4 * np.pi * EPS0 * A0**3,
        
        "alpha_hyperfine": {
            3: 998 * 4 * np.pi * EPS0 * A0**3,
            4: 1002 * 4 * np.pi * EPS0 * A0**3,
        },
        
        # ---------------------------------------------------------------------
        # DIPOLE: INTERMEDIATE → RYDBERG TRANSITION
        # ---------------------------------------------------------------------
        # For Cs133: ⟨nS₁/₂|er|6P₃/₂⟩ - transition from 6P to Rydberg nS
        # The 6P state is the D2 line intermediate at 852 nm
        #
        # Note: Cs uses 6P (not 5P like Rb) because Cs has one more electron shell
        # Ground state is 6S₁/₂, so first excited P state is 6P
        "dipole_intermediate_to_rydberg_ref": 0.012 * E_CHARGE * A0,  # ⟨70S|er|6P⟩
        
        "n_ref": 70,
        "C6_ref": 2 * np.pi * 1400e9 * (1e-6)**6,  # ~1.6× larger than Rb
        "tau_ref": 160e-6,
        "tau_0K_ref": 320e-6,
        
        "scaling_exponents": {
            "C6": 11,
            "lifetime_0K": 3,
            "lifetime_BBR": 2,
            "polarizability": 7,
            "dipole_to_rydberg": -1.5,
        },
        
        "transitions": {
            "ground_to_6P3/2": 351.725718509e12,  # D2 line (852 nm)
            "ground_to_6P1/2": 335.116048807e12,  # D1 line (894 nm)
        },
        
        "trap_wavelength": 1064e-9,
        
        "magic_wavelengths": {
            "hyperfine": {
                "scalar_magic_nm": 866.0,
                "1064nm_differential_Hz_per_mK": 50e3,
            },
            "ground_rydberg": {
                "near_magic_nm": 1064,
            },
        },
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_default_intermediate_state(species: str) -> str:
    """
    Get the default intermediate state for two-photon Rydberg excitation.
    
    For two-photon excitation to Rydberg states, we use:
        Ground → Intermediate (P state) → Rydberg (nS)
    
    The intermediate state depends on the atomic species:
    - Rb87: Uses 5P₃/₂ (D2 line at 780 nm)
    - Cs133: Uses 6P₃/₂ (D2 line at 852 nm)
    
    Parameters
    ----------
    species : str
        Atomic species ("Rb87" or "Cs133")
        
    Returns
    -------
    str
        Intermediate state label (e.g., "5P3/2" or "6P3/2")
        
    Raises
    ------
    ValueError
        If species not recognized
        
    Example
    -------
    >>> get_default_intermediate_state("Rb87")
    '5P3/2'
    >>> get_default_intermediate_state("Cs133")
    '6P3/2'
    """
    state_map = {
        "Rb87": "5P3/2",   # D2 line at 780 nm
        "Cs133": "6P3/2",  # D2 line at 852 nm
    }
    if species not in state_map:
        raise ValueError(f"Unknown species: {species}. Available: {list(state_map.keys())}")
    return state_map[species]


def get_atom_properties(species: str) -> dict:
    """
    Get the full property dictionary for an atomic species.
    
    Parameters
    ----------
    species : str
        Atomic species name: "Rb87" or "Cs133"
        
    Returns
    -------
    dict
        All atomic properties for the species
        
    Raises
    ------
    ValueError
        If species is not in database
        
    Example
    -------
    >>> props = get_atom_properties("Rb87")
    >>> props["mass"]  # kg
    1.443160648e-25
    >>> props["quantum_defects"]["S"]
    3.1311807
    """
    if species not in ATOM_DB:
        raise ValueError(f"Unknown species: {species}. "
                         f"Available: {list(ATOM_DB.keys())}")
    return ATOM_DB[species]


def effective_n(n: int, species: str, orbital: str = "S") -> float:
    """
    Calculate effective principal quantum number n*.
    
    The effective quantum number accounts for the quantum defect:
        n* = n - δ_L
    
    where δ_L is the quantum defect for orbital angular momentum L.
    
    **Physical meaning**: A Rb 70S state has n* = 70 - 3.13 = 66.87.
    This means it has the same binding energy as a hydrogen n=66.87 state.
    
    Parameters
    ----------
    n : int
        Principal quantum number (e.g., 70)
    species : str
        Atomic species ("Rb87" or "Cs133")
    orbital : str
        Orbital type: "S", "P", "D", or "F"
        
    Returns
    -------
    float
        Effective principal quantum number n*
        
    Example
    -------
    >>> effective_n(70, "Rb87", "S")
    66.8688193  # 70 - 3.1311807
    
    >>> effective_n(70, "Cs133", "S")
    65.9506468  # 70 - 4.0493532 (larger defect for Cs)
    """
    delta = ATOM_DB[species]["quantum_defects"].get(orbital, 0)
    return n - delta


def get_quantum_defect(species: str, orbital: str = "S") -> float:
    """
    Get the quantum defect for a given species and orbital type.
    
    **What is a quantum defect?**
    
    In hydrogen, the binding energy is exactly E_n = -Ry/n². But in 
    multi-electron atoms like Rb and Cs, the valence electron sometimes
    "sees" the inner electrons, experiencing more than +1e charge.
    
    The quantum defect δ corrects the energy formula:
        E_n = -Ry/(n - δ)² = -Ry/n*²
    
    **Why does δ depend on orbital type?**
    
    - S orbitals (L=0): Spherically symmetric, penetrate deep into core
      → Largest quantum defect (δ ≈ 3.1 for Rb S-states)
      
    - P orbitals (L=1): Node at nucleus, less core penetration
      → Medium defect (δ ≈ 2.7 for Rb)
      
    - D orbitals (L=2): Two nodes, even less penetration
      → Smaller defect (δ ≈ 1.3 for Rb)
      
    - F orbitals (L=3): Three nodes, almost no core penetration
      → Nearly hydrogenic (δ ≈ 0.02 for Rb)
    
    Parameters
    ----------
    species : str
        Atomic species ("Rb87" or "Cs133")
    orbital : str
        Orbital type: "S", "P", "D", or "F"
        
    Returns
    -------
    float
        Quantum defect δ
        
    Example
    -------
    >>> get_quantum_defect("Rb87", "S")
    3.1311807
    """
    return ATOM_DB[species]["quantum_defects"].get(orbital, 0)


def get_rydberg_energy(n: int, species: str, orbital: str = "S") -> float:
    """
    Calculate the binding energy of a Rydberg state.
    
    The energy is measured from the ionization threshold (E = 0 at ionization):
        E_n = -Ry / n*²
    
    where n* = n - δ is the effective principal quantum number.
    
    **Physical picture**: A Rydberg atom is like hydrogen with one important
    correction: the valence electron occasionally penetrates the inner
    electron core. The quantum defect accounts for this.
    
    Parameters
    ----------
    n : int
        Principal quantum number (typically 50-100 for Rydberg states)
    species : str
        Atomic species ("Rb87" or "Cs133")
    orbital : str
        Orbital type: "S", "P", "D", or "F"
        
    Returns
    -------
    float
        Binding energy in Joules (NEGATIVE value, since bound state)
        
    Example
    -------
    >>> E_70S = get_rydberg_energy(70, "Rb87", "S")
    >>> E_70S / E_CHARGE  # Convert to eV
    -0.00305  # About -3.05 meV, very weakly bound!
    
    Compare to ground state: -4.18 eV (1400× more tightly bound)
    """
    n_star = effective_n(n, species, orbital)
    return -RY_JOULES / n_star**2


def get_C6(n: int, species: str) -> float:
    """
    Calculate the C₆ coefficient for Rydberg-Rydberg interactions.
    
    **What is C₆?**
    
    When two Rydberg atoms are separated by distance R, they interact via
    the van der Waals potential:
    
        V(R) = C₆ / R⁶
    
    This interaction is CRUCIAL for quantum gates! If V >> Ω (Rabi frequency),
    the atoms "block" each other from both being excited. This is called
    the "Rydberg blockade" and is the basis of the CZ gate.
    
    **Why does C₆ scale as n¹¹?**
    
    Van der Waals interactions arise from fluctuating electric dipoles.
    For Rydberg atoms:
    - Polarizability α ∝ ⟨r²⟩ × (dipole moments)² ~ n⁷
    - C₆ ∝ α² would give n¹⁴
    - But the frequency denominators reduce this to ~n¹¹
    
    **Practical numbers**:
    - n=50: C₆ ≈ 25 GHz·μm⁶
    - n=70: C₆ ≈ 860 GHz·μm⁶ (35× larger!)
    - n=100: C₆ ≈ 25,000 GHz·μm⁶
    
    Parameters
    ----------
    n : int
        Principal quantum number
    species : str
        Atomic species ("Rb87" or "Cs133")
        
    Returns
    -------
    float
        C₆ coefficient in J·m⁶
        
    Example
    -------
    >>> C6_70 = get_C6(70, "Rb87")
    >>> C6_70 / (2 * np.pi * 1e9 * (1e-6)**6)  # Convert to GHz·μm⁶
    862.69
    
    >>> # Blockade at R = 4 μm
    >>> V = C6_70 / (4e-6)**6
    >>> V / (2 * np.pi * 1e6)  # Convert to MHz
    ~210  # V/2π ≈ 210 MHz = strong blockade!
    """
    props = ATOM_DB[species]
    n_star = effective_n(n, species, "S")
    n_star_ref = effective_n(props["n_ref"], species, "S")
    
    # Scale from reference value
    scaling_exp = props["scaling_exponents"]["C6"]
    return props["C6_ref"] * (n_star / n_star_ref)**scaling_exp


def get_rydberg_lifetime(n: int, species: str, temperature: float = 300.0) -> float:
    """
    Calculate Rydberg state lifetime including blackbody radiation.
    
    **What limits Rydberg lifetime?**
    
    1. **Spontaneous emission** (dominates at T=0):
       - The Rydberg atom can spontaneously emit a photon and decay to a 
         lower state. Rate γ_sp ∝ n⁻³ (longer lifetime at higher n!)
       
    2. **Blackbody radiation** (BBR, dominates at room temperature):
       - Thermal photons from the environment can stimulate transitions
         to nearby Rydberg states, effectively "ionizing" the atom.
       - Rate γ_BBR ∝ T⁴ × n⁻² (worse at higher T, better at higher n)
    
    **Total lifetime**: 1/τ = 1/τ_sp + 1/τ_BBR
    
    **Practical numbers for Rb87 at n=70**:
    - T = 0 K: τ ≈ 280 μs (spontaneous only)
    - T = 300 K: τ ≈ 140 μs (BBR doubles decay rate!)
    - T = 4 K (cryogenic): τ ≈ 270 μs (almost no BBR)
    
    Parameters
    ----------
    n : int
        Principal quantum number
    species : str
        Atomic species ("Rb87" or "Cs133")
    temperature : float
        Temperature in Kelvin (default 300 K = room temperature)
        
    Returns
    -------
    float
        Lifetime in seconds
        
    Example
    -------
    >>> tau_300K = get_rydberg_lifetime(70, "Rb87", 300)
    >>> tau_300K * 1e6  # Convert to μs
    ~140
    
    >>> tau_4K = get_rydberg_lifetime(70, "Rb87", 4)
    >>> tau_4K * 1e6  # Convert to μs
    ~270  # Much longer at cryogenic temperatures!
    """
    props = ATOM_DB[species]
    n_star = effective_n(n, species, "S")
    n_star_ref = effective_n(props["n_ref"], species, "S")
    
    # Spontaneous emission lifetime (scales as n³)
    tau_0K = props["tau_0K_ref"] * (n_star / n_star_ref)**props["scaling_exponents"]["lifetime_0K"]
    
    if temperature < 1:
        # Effectively zero temperature
        return tau_0K
    
    # BBR lifetime (scales as n² and depends on temperature)
    # τ_BBR ∝ n² / T⁴ for T >> T_rydberg, but we use empirical fit
    T_ref = 300.0
    tau_BBR_ref = props["tau_ref"] * props["tau_0K_ref"] / (props["tau_0K_ref"] - props["tau_ref"])
    
    # Scale BBR contribution
    tau_BBR = tau_BBR_ref * (n_star / n_star_ref)**props["scaling_exponents"]["lifetime_BBR"]
    tau_BBR *= (T_ref / temperature)**4  # Temperature scaling
    
    # Combine: 1/τ = 1/τ_sp + 1/τ_BBR
    return 1.0 / (1.0/tau_0K + 1.0/tau_BBR)


def get_rydberg_polarizability(n: int, species: str) -> float:
    """
    Calculate Rydberg state polarizability at trap wavelength.
    
    **What is polarizability?**
    
    When an atom is placed in an electric field E, its electron cloud 
    distorts, creating an induced dipole moment:
        d_induced = α × E
    
    where α is the polarizability. The energy shift in the field is:
        U = -α × E² / 2 = -α × I / (2ε₀c)
    
    where I is the laser intensity.
    
    **Why does this matter for trapping?**
    
    For GROUND STATE atoms: α > 0 at 1064 nm (red-detuned from transitions)
    → Atoms are attracted to intensity MAXIMA (center of tweezer)
    → This is how optical tweezers TRAP atoms!
    
    For RYDBERG STATE atoms: α < 0 at 1064 nm (giant negative polarizability)
    → Atoms are REPELLED from intensity maxima
    → During gate operation, atoms feel an ANTI-TRAPPING force!
    
    **Why is Rydberg polarizability so large and negative?**
    
    The electron orbit radius scales as n², and polarizability depends on
    how easily the electron cloud can be distorted. Result: α ∝ n⁷.
    
    At n=70: |α_rydberg| ~ 200,000 atomic units
    Ground state: α_ground ~ 700 atomic units
    Ratio: ~300× larger polarizability for Rydberg state!
    
    The NEGATIVE sign comes from the AC Stark effect: at 1064 nm, the
    Rydberg state is shifted UP in energy (blue shift) because the laser
    frequency is above the Rydberg-to-continuum transition frequencies.
    
    Parameters
    ----------
    n : int
        Principal quantum number
    species : str
        Atomic species ("Rb87" or "Cs133")
        
    Returns
    -------
    float
        Polarizability in SI units (C²·m²/J)
        
    Example
    -------
    >>> alpha_r = get_rydberg_polarizability(70, "Rb87")
    >>> alpha_ground = ATOM_DB["Rb87"]["alpha_ground"]
    >>> abs(alpha_r) / alpha_ground  # Ratio
    ~290  # Rydberg is ~300× more polarizable (and opposite sign!)
    """
    props = ATOM_DB[species]
    n_star = effective_n(n, species, "S")
    n_star_ref = effective_n(props["n_ref"], species, "S")
    
    scaling_exp = props["scaling_exponents"]["polarizability"]
    return props["alpha_rydberg_ref"] * (n_star / n_star_ref)**scaling_exp


def get_dipole_to_rydberg(n: int, species: str) -> float:
    """
    Get the dipole matrix element from intermediate (5P/6P) to Rydberg (nS).
    
    **What is a dipole matrix element?**
    
    When a laser drives an atomic transition, the coupling strength depends
    on how much the electron "overlaps" with the electric field. This is
    quantified by the dipole matrix element:
    
        d = ⟨final|e·r|initial⟩
    
    The Rabi frequency (how fast we drive the transition) is:
        Ω = d · E₀ / ℏ
    
    where E₀ is the laser electric field amplitude.
    
    **Why does d decrease with n?**
    
    The Rydberg wavefunction spreads out over a larger volume as n increases:
    - Orbital radius: r ~ n² × a₀
    - Wavefunction amplitude: |ψ|² ~ 1/r³ ~ n⁻⁶
    
    The overlap with the localized intermediate P state decreases:
        d ∝ ∫ ψ_P × r × ψ_nS dr ∝ n⁻³/²
    
    **Practical consequence**: To maintain the same Rabi frequency at 
    higher n, you need MORE laser power (since Ω ∝ d × E₀ ∝ d × √P).
    
    **Species-specific transitions**:
    - Rb87: 5P₃/₂ → nS₁/₂ (D2 line at 780 nm as intermediate)
    - Cs133: 6P₃/₂ → nS₁/₂ (D2 line at 852 nm as intermediate)
    
    Parameters
    ----------
    n : int
        Principal quantum number of Rydberg state
    species : str
        Atomic species ("Rb87" or "Cs133")
        
    Returns
    -------
    float
        Reduced dipole matrix element in SI units (C·m)
        
    Example
    -------
    >>> d_70 = get_dipole_to_rydberg(70, "Rb87")
    >>> d_50 = get_dipole_to_rydberg(50, "Rb87")
    >>> d_50 / d_70  # Ratio
    ~1.65  # Stronger coupling at lower n (closer to core)
    """
    props = ATOM_DB[species]
    n_star = effective_n(n, species, "S")
    n_star_ref = effective_n(props["n_ref"], species, "S")
    
    scaling_exp = props["scaling_exponents"]["dipole_to_rydberg"]
    return props["dipole_intermediate_to_rydberg_ref"] * (n_star / n_star_ref)**scaling_exp


def get_intermediate_state_linewidth(species: str, 
                                      intermediate_state: str = "5P3/2") -> float:
    """
    Get the natural linewidth of an intermediate excited state.
    
    **What is natural linewidth?**
    
    An excited state has a finite lifetime τ due to spontaneous emission.
    The uncertainty principle gives a minimum energy width:
        ΔE × τ ≥ ℏ/2
    
    This translates to a frequency width Γ = 1/τ, called the "natural linewidth".
    
    For the Rb87 D2 line (5P₃/₂):
    - Lifetime τ = 26.2 ns
    - Linewidth Γ/2π = 6.07 MHz
    
    **Why does linewidth matter for two-photon excitation?**
    
    When we excite Ground → Intermediate → Rydberg, we DETUNE from the
    intermediate state by Δₑ to avoid scattering. But if we're too close
    (Δₑ < few × Γ), we still scatter significant photons.
    
    The off-resonant scattering rate is:
        γ_scatter ≈ Ω₁² × Γ / (4Δₑ²)
    
    where Ω₁ is the first laser Rabi frequency. This sets the minimum Δₑ.
    
    Parameters
    ----------
    species : str
        Atomic species ("Rb87" or "Cs133")
    intermediate_state : str
        State label (e.g., "5P3/2", "5P1/2", "6P3/2", "6P1/2")
        
    Returns
    -------
    float
        Natural linewidth Γ in rad/s (NOT Hz!)
        To convert to Hz: Γ_Hz = Γ / (2π)
        
    Example
    -------
    >>> Gamma = get_intermediate_state_linewidth("Rb87", "5P3/2")
    >>> Gamma / (2 * np.pi * 1e6)  # Convert to MHz
    6.065  # The D2 line has ~6 MHz natural linewidth
    """
    # Handle different naming conventions
    if species == "Rb87":
        key = intermediate_state.replace("5P", "5P").replace("1/2", "1/2").replace("3/2", "3/2")
    else:  # Cs133
        key = intermediate_state.replace("5P", "6P").replace("6P", "6P")
        if "5P" in intermediate_state:
            key = intermediate_state.replace("5P", "6P")
    
    return ATOM_DB[species]["intermediate_states"][key]["linewidth"]


def get_hyperfine_splitting(species: str) -> float:
    """
    Get the ground state hyperfine splitting frequency.
    
    **What is hyperfine splitting?**
    
    The electron spin and nuclear spin interact magnetically, splitting
    the ground state into two manifolds with different total angular 
    momentum F = I + J.
    
    For Rb87 (I=3/2, J=1/2): F = 1 or F = 2
    Splitting: 6.834682611 GHz
    
    For Cs133 (I=7/2, J=1/2): F = 3 or F = 4  
    Splitting: 9.192631770 GHz (defines the SI second!)
    
    **Why does this matter for qubits?**
    
    The qubit states |0⟩ and |1⟩ are typically in different hyperfine
    manifolds. The splitting frequency is the "qubit frequency" that
    must be addressed with microwaves for single-qubit gates.
    
    Parameters
    ----------
    species : str
        Atomic species ("Rb87" or "Cs133")
        
    Returns
    -------
    float
        Hyperfine splitting in Hz
        
    Example
    -------
    >>> f_hf = get_hyperfine_splitting("Rb87")
    >>> f_hf / 1e9  # Convert to GHz
    6.834682611
    """
    if species == "Rb87":
        return 6.834682610904e9  # Hz
    elif species == "Cs133":
        return 9.192631770e9  # Hz (this defines the SI second!)
    else:
        raise ValueError(f"Unknown species: {species}")


# =============================================================================
# CONVENIENCE FUNCTIONS FOR COMMON QUERIES
# =============================================================================

def list_available_species() -> List[str]:
    """Return list of available atomic species."""
    return list(ATOM_DB.keys())


def get_mass(species: str) -> float:
    """Get atomic mass in kg."""
    return ATOM_DB[species]["mass"]


def get_ionization_energy(species: str) -> float:
    """Get ionization energy in Joules."""
    return ATOM_DB[species]["E_ionization"]


def get_ground_state_polarizability(species: str, F: Optional[int] = None) -> float:
    """
    Get ground state polarizability at trap wavelength.
    
    Parameters
    ----------
    species : str
        Atomic species
    F : int, optional
        Hyperfine level. If None, returns average polarizability.
        
    Returns
    -------
    float
        Polarizability in SI units (C²·m²/J)
    """
    if F is not None and "alpha_hyperfine" in ATOM_DB[species]:
        return ATOM_DB[species]["alpha_hyperfine"][F]
    return ATOM_DB[species]["alpha_ground"]


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # The database
    "ATOM_DB",
    
    # Main accessors
    "get_atom_properties",
    "get_quantum_defect",
    "effective_n",
    "get_rydberg_energy",
    
    # Rydberg scaling functions
    "get_C6",
    "get_rydberg_lifetime", 
    "get_rydberg_polarizability",
    "get_dipole_to_rydberg",
    
    # Intermediate state
    "get_intermediate_state_linewidth",
    
    # Ground state
    "get_hyperfine_splitting",
    "get_ground_state_polarizability",
    "get_mass",
    "get_ionization_energy",
    
    # Utilities
    "list_available_species",
]
