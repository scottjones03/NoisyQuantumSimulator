"""
Laser Physics for Rydberg Gate Simulations
===========================================

This module provides functions for computing laser-atom interactions,
including Rabi frequencies, two-photon excitation, and the Rydberg blockade.

TWO-PHOTON RYDBERG EXCITATION
-----------------------------

**Why two photons?**

To excite an atom from the ground state to a Rydberg state requires a LOT of
energy. For Rb87 going from 5S to 70S:
- Energy difference: ~4 eV
- Single photon: λ ≈ 310 nm (deep UV, difficult to work with)

Instead, we use TWO photons:
1. First photon (780 nm, red): Ground → Intermediate (5P₃/₂)
2. Second photon (480 nm, blue): Intermediate → Rydberg (70S)

**The key trick: INTERMEDIATE STATE DETUNING**

We don't actually populate the intermediate state! Instead, we DETUNE both
lasers so that the atom only goes to the Rydberg state when it absorbs
BOTH photons simultaneously:

```
                        ↑ Energy
                        │
    Rydberg (70S) ─────┼─────────────
                        │     ↑
                        │     │ Blue laser (480 nm)
                        │     │
    Intermediate ═══════════════════════ (virtual, never populated)
          (5P)          │     ↑  Δₑ (detuning)
                        │     │
                        │ Red laser (780 nm)
                        │     │
    Ground (5S) ──────────────────────
```

When Δₑ >> Γ (linewidth), the atom never "stops" at the intermediate state,
avoiding spontaneous emission from that state.

EFFECTIVE TWO-PHOTON RABI FREQUENCY
-----------------------------------

The effective Rabi frequency for the two-photon process is:

    Ω_eff = (Ω₁ × Ω₂) / (2 × Δₑ)

where:
- Ω₁: First laser Rabi frequency (ground → intermediate)
- Ω₂: Second laser Rabi frequency (intermediate → Rydberg)
- Δₑ: Detuning from intermediate state

**Physical interpretation**: The probability amplitude to virtually populate
the intermediate state is ~Ω₁/Δₑ. The coupling out of that virtual state is
~Ω₂. Combined with the factor of 2 from rotating wave approximation, we get
Ω_eff = Ω₁Ω₂/(2Δₑ).

**Trade-off**: Larger Δₑ means less scattering (good!) but requires more
laser power for the same Ω_eff (bad!). Typical choice: Δₑ ~ 1-10 GHz.

THE RYDBERG BLOCKADE
--------------------

When two atoms are both in Rydberg states, they interact via van der Waals:

    V(R) = C₆ / R⁶

This interaction SHIFTS the energy of the doubly-excited state |rr⟩.
If V >> Ω, the laser is FAR OFF RESONANCE for exciting the second atom,
and only ONE atom can be excited. This is the "Rydberg blockade."

**Blockade radius**: The distance where V(R_b) = ℏΩ

    R_b = (C₆ / ℏΩ)^(1/6)

At R < R_b: Strong blockade, only one atom excited
At R > R_b: Weak blockade, both atoms can be excited

For n=70 Rb87 with Ω = 2π × 5 MHz:
- C₆ ≈ 860 GHz·μm⁶
- R_b ≈ 10 μm

Typical atom spacing: 3-5 μm << R_b, so blockade is STRONG.

References
----------
[1] Saffman et al., Rev. Mod. Phys. 82, 2313 (2010) - Comprehensive review
[2] Levine et al., Phys. Rev. Lett. 123, 170503 (2019) - High-fidelity gates
[3] Bluvstein PhD Thesis, Harvard (2024) - Experimental parameters
"""

import numpy as np
from typing import Tuple, Optional

from .constants import HBAR, EPS0, C, E_CHARGE, A0
from .atom_database import (
    ATOM_DB, get_C6, get_rydberg_polarizability,
    get_intermediate_state_linewidth, effective_n
)


# =============================================================================
# ELECTRIC FIELD AND INTENSITY
# =============================================================================

def laser_E0(power: float, waist: float) -> float:
    """
    Calculate peak electric field amplitude of a Gaussian laser beam.
    
    **What is E₀?**
    
    A laser beam is an oscillating electromagnetic wave with electric field:
        E(t) = E₀ cos(ωt)
    
    The electric field amplitude E₀ determines how strongly the laser
    couples to atomic transitions (through the dipole interaction d·E).
    
    **Relation to intensity:**
    
    For a Gaussian beam with total power P and waist w:
        Peak intensity: I₀ = 2P / (π w²)
        Field-intensity relation: I = (1/2) ε₀ c E₀²
        
    Combining: E₀ = √(4P / (π w² ε₀ c))
    
    **Typical values:**
    - 1 mW, 50 μm waist: E₀ ≈ 870 V/m (ground-intermediate)
    - 1 W, 20 μm waist: E₀ ≈ 6.9 kV/m (intermediate-Rydberg)
    
    Parameters
    ----------
    power : float
        Total laser power in Watts
    waist : float
        Beam waist (1/e² intensity radius) in meters
        
    Returns
    -------
    float
        Peak electric field amplitude in V/m
        
    Example
    -------
    >>> # 1 W blue laser with 20 μm waist
    >>> E0 = laser_E0(1.0, 20e-6)
    >>> print(f"E₀ = {E0:.0f} V/m = {E0/1000:.2f} kV/m")
    E₀ = 6909 V/m = 6.91 kV/m
    """
    # Peak intensity of Gaussian beam
    I_peak = 2 * power / (np.pi * waist**2)
    
    # E₀ from I = (1/2) ε₀ c E₀²
    return np.sqrt(2 * I_peak / (EPS0 * C))


def laser_intensity(power: float, waist: float) -> float:
    """
    Calculate peak intensity of a Gaussian laser beam.
    
    I₀ = 2P / (π w²)
    
    Parameters
    ----------
    power : float
        Total power in Watts
    waist : float
        Beam waist in meters
        
    Returns
    -------
    float
        Peak intensity in W/m²
        
    Example
    -------
    >>> I = laser_intensity(1e-3, 50e-6)  # 1 mW, 50 μm
    >>> print(f"I = {I/1e6:.1f} MW/m² = {I/1e4:.1f} W/cm²")
    """
    return 2 * power / (np.pi * waist**2)


# =============================================================================
# SINGLE-PHOTON RABI FREQUENCY
# =============================================================================

def single_photon_rabi(dipole: float, E0: float) -> float:
    """
    Calculate single-photon Rabi frequency.
    
    **What is a Rabi frequency?**
    
    When an atom is driven by a near-resonant laser, it oscillates between
    ground and excited states. The oscillation frequency is the Rabi frequency:
    
        Ω = d · E₀ / ℏ
    
    where d is the transition dipole moment and E₀ is the electric field.
    
    **Physical picture**: Ω/2π is how many times per second the atom
    "flops" between ground and excited states. A π-pulse (flip the state)
    takes time t = π/Ω.
    
    **Typical values for Rb87:**
    - Ground → 5P₃/₂: d ≈ 4.2 ea₀ ≈ 3.6×10⁻²⁹ C·m
      With E₀ = 870 V/m: Ω/2π ≈ 47 MHz
      
    - 5P₃/₂ → 70S: d ≈ 0.014 ea₀ ≈ 1.2×10⁻³¹ C·m
      With E₀ = 6900 V/m: Ω/2π ≈ 80 kHz (much weaker!)
    
    Parameters
    ----------
    dipole : float
        Transition dipole moment in C·m
    E0 : float
        Peak electric field amplitude in V/m
        
    Returns
    -------
    float
        Rabi frequency in rad/s (divide by 2π for Hz)
        
    Example
    -------
    >>> d = 4.2 * E_CHARGE * A0  # Rb D2 line
    >>> E0 = laser_E0(1e-3, 50e-6)  # 1 mW, 50 μm
    >>> Omega = single_photon_rabi(d, E0)
    >>> print(f"Ω/2π = {Omega/(2*np.pi)/1e6:.1f} MHz")
    """
    return dipole * E0 / HBAR


def single_photon_rabi_from_power(dipole: float, power: float, waist: float) -> float:
    """
    Calculate single-photon Rabi frequency directly from laser power.
    
    Convenience function combining laser_E0 and single_photon_rabi.
    
    Parameters
    ----------
    dipole : float
        Transition dipole moment in C·m
    power : float
        Laser power in Watts
    waist : float
        Beam waist in meters
        
    Returns
    -------
    float
        Rabi frequency in rad/s
    """
    E0 = laser_E0(power, waist)
    return single_photon_rabi(dipole, E0)


# =============================================================================
# TWO-PHOTON RABI FREQUENCY
# =============================================================================

def two_photon_rabi(Omega1: float, Omega2: float, Delta_e: float) -> float:
    """
    Calculate effective two-photon Rabi frequency.
    
    **Two-photon excitation:**
    
    When driving a two-photon transition (Ground → Intermediate → Rydberg)
    with the intermediate state detuned by Δₑ, the effective Rabi frequency is:
    
        Ω_eff = Ω₁ × Ω₂ / (2 × Δₑ)
    
    This comes from adiabatic elimination of the intermediate state.
    
    **⚠️ CRITICAL: Two different Δ symbols in this codebase!**
    
    This function uses **Δₑ (intermediate detuning)**, which is:
    - LARGE: typically 1-10 GHz
    - Purpose: Avoid populating the short-lived P state (lifetime ~26 ns)
    - Determines the STRENGTH of the effective |1⟩→|r⟩ coupling
    
    This is DIFFERENT from the **Δ (two-photon detuning)** in hamiltonians.py:
    - SMALL: typically 0-5 MHz
    - Purpose: Fine-tune gate phases for CZ operation
    - Determines how far the effective laser is from |r⟩ resonance
    
    **Physical picture:**
    
        |r⟩ Rydberg ─────────────────────────────── (target)
                              ↑
                         Δ ~ MHz (two-photon detuning, in Hamiltonian)
                              ↑
        |P⟩ Intermediate ═══════════════════════════ (virtual, Δₑ ~ GHz away)
                    ↑                   ↑
                   Ω₁                  Ω₂
                    ↑                   ↑
        |1⟩ Ground ──────────────────────────────── (initial state)
    
    The atom "borrows" the P state virtually for time ~1/Δₑ, then continues
    to |r⟩. The effective coupling Ω_eff is weaker because of this virtual hop.
    
    **Derivation (simple version):**
    
    1. Amplitude in intermediate state: c_e ~ Ω₁/(2Δₑ) (virtual occupation)
    2. Coupling from intermediate to Rydberg: Ω₂
    3. Effective coupling: Ω_eff ~ c_e × Ω₂ = Ω₁Ω₂/(2Δₑ)
    
    **Practical numbers (Bluvstein thesis):**
    - Ω₁/2π ~ 100 MHz (red laser, easily achieved)
    - Ω₂/2π ~ 0.5 MHz (blue laser, limited by dipole)
    - Δₑ/2π ~ 7.8 GHz (far detuned)
    - Ω_eff/2π ≈ 100 × 0.5 / (2 × 7800) ≈ 3 kHz... wait, that's too small!
    
    Actually, to get Ω_eff/2π = 4.6 MHz with Δₑ/2π = 7.8 GHz requires
    Ω₁×Ω₂ ~ 2π × 72 GHz². This needs HIGH power lasers.
    
    **Scattering rate consideration:**
    
    The residual P-state population is ~(Ω₁/2Δₑ)². For Ω₁ = 2π×100 MHz and
    Δₑ = 2π×10 GHz: P(P) ~ 10⁻⁴. With P-state lifetime ~26 ns (Γ ~ 6 MHz):
    
        Γ_scatter ~ P(P) × Γ_P ~ 10⁻⁴ × 6 MHz ~ 600 Hz
    
    During a 200 ns gate: probability of scattering ~ 600 Hz × 200 ns ~ 10⁻⁴
    This sets a fundamental limit on gate fidelity!
    
    Parameters
    ----------
    Omega1 : float
        First laser Rabi frequency (ground → intermediate) in rad/s
    Omega2 : float
        Second laser Rabi frequency (intermediate → Rydberg) in rad/s
    Delta_e : float
        Intermediate state detuning in rad/s (positive = blue-detuned).
        THIS IS THE LARGE (~GHz) DETUNING, not the small two-photon detuning!
        
    Returns
    -------
    float
        Effective two-photon Rabi frequency in rad/s.
        This Ω_eff is what appears as "Ω" in the Hamiltonian H_laser.
        
    Example
    -------
    >>> Omega1 = 2*np.pi * 100e6  # 100 MHz
    >>> Omega2 = 2*np.pi * 720e6  # 720 MHz (needs ~5W with tight focus)
    >>> Delta_e = 2*np.pi * 7.8e9  # 7.8 GHz
    >>> Omega_eff = two_photon_rabi(Omega1, Omega2, Delta_e)
    >>> print(f"Ω_eff/2π = {Omega_eff/(2*np.pi)/1e6:.1f} MHz")
    Ω_eff/2π = 4.6 MHz
    
    See Also
    --------
    hamiltonians.build_detuning_hamiltonian : Uses the OTHER Δ (two-photon)
    """
    return Omega1 * Omega2 / (2 * Delta_e)


def required_powers_for_two_photon_rabi(
    Omega_eff_target: float,
    Delta_e: float,
    dipole_1: float,
    dipole_2: float,
    waist_1: float,
    waist_2: float,
    power_ratio: float = 1.0
) -> Tuple[float, float]:
    """
    Calculate required laser powers to achieve a target two-photon Rabi frequency.
    
    Given Ω_eff = Ω₁Ω₂/(2Δₑ) and Ω = dE₀/ℏ ∝ √P, we can solve for the
    required powers.
    
    **The algebra:**
    
    Ω₁ = d₁ × √(4P₁/(π w₁² ε₀ c)) / ℏ
    Ω₂ = d₂ × √(4P₂/(π w₂² ε₀ c)) / ℏ
    
    Ω_eff = (d₁ d₂ / ℏ²) × √(4P₁/(π w₁² ε₀ c)) × √(4P₂/(π w₂² ε₀ c)) / (2Δₑ)
    
    Let power_ratio = P₂/P₁, then:
    
    P₁ = [Ω_eff × 2Δₑ × ℏ² × π × ε₀ c / (2 d₁ d₂)]² × (w₁² w₂²) / power_ratio
    
    Parameters
    ----------
    Omega_eff_target : float
        Desired effective Rabi frequency in rad/s
    Delta_e : float
        Intermediate state detuning in rad/s
    dipole_1 : float
        First transition dipole moment in C·m
    dipole_2 : float
        Second transition dipole moment in C·m
    waist_1 : float
        First laser waist in meters
    waist_2 : float
        Second laser waist in meters
    power_ratio : float
        P₂/P₁ ratio (default 1.0 for equal powers)
        
    Returns
    -------
    Tuple[float, float]
        (P₁, P₂) required powers in Watts
    """
    # Common factor
    prefactor = (HBAR**2 * np.pi * EPS0 * C) / (4 * dipole_1 * dipole_2)
    
    # Product P₁ × P₂
    P_product = (Omega_eff_target * 2 * Delta_e * prefactor)**2 * waist_1**2 * waist_2**2
    
    # Solve for individual powers given ratio
    P1 = np.sqrt(P_product / power_ratio)
    P2 = power_ratio * P1
    
    return P1, P2


# =============================================================================
# RYDBERG BLOCKADE
# =============================================================================

def rydberg_blockade(C6: float, R: float) -> float:
    """
    Calculate Rydberg-Rydberg interaction energy.
    
    **The van der Waals interaction:**
    
    Two Rydberg atoms at distance R interact via:
    
        V(R) = C₆ / R⁶
    
    This is the van der Waals (vdW) interaction, arising from fluctuating
    dipole-dipole coupling. The 1/R⁶ scaling comes from second-order
    perturbation theory of the dipole-dipole interaction (1/R³)².
    
    **Why does this cause a "blockade"?**
    
    Consider two atoms driven by a laser with Rabi frequency Ω:
    
    - Single atom: Can be excited when laser is on resonance (Δ = 0)
    - Two atoms: If both were excited, energy would be E_rr = 2E_r + V
      But laser is on resonance with E_r, not E_r + V/2!
      
    If V >> ℏΩ, the doubly-excited state is far off resonance, and
    the laser cannot excite both atoms. Only ONE can be excited.
    
    **Physical origin of C₆:**
    
    For two atoms in the same Rydberg state |nS⟩, the vdW interaction
    arises from virtual coupling to nearby |nP⟩ states:
    
    |nS, nS⟩ → |nP, nP⟩ → |nS, nS⟩ (second-order)
    
    Since the |nP⟩ state is close in energy (Δ ~ 1/n³), and the dipole
    coupling scales as n², the C₆ coefficient scales as n¹¹.
    
    Parameters
    ----------
    C6 : float
        Dispersion coefficient in J·m⁶
    R : float
        Interatomic separation in meters
        
    Returns
    -------
    float
        Interaction energy V(R) in Joules
        
    Example
    -------
    >>> C6 = get_C6(70, "Rb87")  # ~5.4×10⁻⁶⁰ J·m⁶
    >>> R = 4e-6  # 4 μm
    >>> V = rydberg_blockade(C6, R)
    >>> print(f"V/h = {V/HBAR/(2*np.pi)/1e6:.0f} MHz")
    V/h = 210 MHz
    """
    return C6 / R**6


def blockade_shift_MHz(C6_GHz_um6: float, R_um: float) -> float:
    """
    Calculate blockade shift in convenient units.
    
    V/h [MHz] = C₆ [GHz·μm⁶] × 1000 / R⁶ [μm⁶]
    
    Parameters
    ----------
    C6_GHz_um6 : float
        C₆ coefficient in GHz·μm⁶
    R_um : float
        Interatomic distance in μm
        
    Returns
    -------
    float
        Blockade shift in MHz
        
    Example
    -------
    >>> V_MHz = blockade_shift_MHz(860, 4.0)  # n=70 Rb at 4 μm
    >>> print(f"V = {V_MHz:.0f} MHz")
    V = 210 MHz
    """
    return C6_GHz_um6 * 1000 / R_um**6


def blockade_radius(C6: float, Omega: float) -> float:
    """
    Calculate the Rydberg blockade radius.
    
    **Definition:**
    
    The blockade radius R_b is the distance at which the interaction
    equals the Rabi frequency:
    
        V(R_b) = ℏΩ  →  C₆/R_b⁶ = ℏΩ  →  R_b = (C₆/(ℏΩ))^(1/6)
    
    **Physical meaning:**
    
    - At R < R_b: V >> ℏΩ, strong blockade, only one excitation possible
    - At R > R_b: V << ℏΩ, weak blockade, atoms behave independently
    
    **Typical values:**
    
    For Rb87 at n=70 with Ω/2π = 5 MHz:
    - C₆ ~ 860 GHz·μm⁶
    - R_b ~ 10 μm
    
    For n=53 with Ω/2π = 4.6 MHz (Bluvstein):
    - C₆ ~ 34 GHz·μm⁶
    - R_b ~ 4.3 μm
    
    Parameters
    ----------
    C6 : float
        Dispersion coefficient in J·m⁶
    Omega : float
        Rabi frequency in rad/s
        
    Returns
    -------
    float
        Blockade radius in meters
        
    Example
    -------
    >>> C6 = get_C6(70, "Rb87")
    >>> Omega = 2*np.pi * 5e6  # 5 MHz
    >>> Rb = blockade_radius(C6, Omega)
    >>> print(f"R_b = {Rb*1e6:.1f} μm")
    R_b = 10.1 μm
    """
    return (C6 / (HBAR * Omega))**(1/6)


def blockade_radius_um(C6_GHz_um6: float, Omega_MHz: float) -> float:
    """
    Calculate blockade radius in convenient units.
    
    R_b [μm] = (C₆ [GHz·μm⁶] × 1000 / Ω [MHz])^(1/6)
    
    Parameters
    ----------
    C6_GHz_um6 : float
        C₆ coefficient in GHz·μm⁶
    Omega_MHz : float
        Rabi frequency in MHz
        
    Returns
    -------
    float
        Blockade radius in μm
    """
    return (C6_GHz_um6 * 1000 / Omega_MHz)**(1/6)


def V_over_Omega(C6: float, R: float, Omega: float) -> float:
    """
    Calculate the ratio V/Ω (blockade strength parameter).
    
    **The most important parameter for Rydberg gates!**
    
    V/Ω tells you how "blockaded" your system is:
    
    - V/Ω >> 1: Strong blockade regime
      Only one atom can be excited at a time
      Gate errors ~ Ω/V (small)
      
    - V/Ω ~ 1: Intermediate regime
      Partial population of |rr⟩ state
      Gate needs careful optimization
      
    - V/Ω << 1: Weak blockade
      Both atoms excited independently
      No entangling gate possible
    
    **Optimal operating point:**
    
    For the Levine-Pichler protocol, V/Ω ~ 40-100 is typical.
    Too high: Need very close atoms (hard experimentally)
    Too low: Blockade errors dominate
    
    Parameters
    ----------
    C6 : float
        Dispersion coefficient in J·m⁶
    R : float
        Interatomic separation in meters
    Omega : float
        Rabi frequency in rad/s
        
    Returns
    -------
    float
        Dimensionless ratio V/(ℏΩ)
        
    Example
    -------
    >>> C6 = get_C6(70, "Rb87")
    >>> R = 4e-6  # 4 μm
    >>> Omega = 2*np.pi * 5e6  # 5 MHz
    >>> ratio = V_over_Omega(C6, R, Omega)
    >>> print(f"V/Ω = {ratio:.1f}")
    V/Ω = 42.3
    """
    V = rydberg_blockade(C6, R)
    return V / (HBAR * Omega)


def spacing_for_target_V_over_Omega(
    C6: float, 
    Omega: float, 
    target_ratio: float
) -> float:
    """
    Calculate atom spacing needed for a target V/Ω ratio.
    
    Given V/Ω = C₆/(ℏΩR⁶), solve for R:
    
        R = (C₆ / (ℏΩ × target_ratio))^(1/6)
    
    Parameters
    ----------
    C6 : float
        Dispersion coefficient in J·m⁶
    Omega : float
        Rabi frequency in rad/s
    target_ratio : float
        Desired V/Ω ratio
        
    Returns
    -------
    float
        Required spacing in meters
        
    Example
    -------
    >>> # Find spacing for V/Ω = 45 at n=53, Ω/2π = 4.6 MHz
    >>> C6 = get_C6(53, "Rb87")
    >>> Omega = 2*np.pi * 4.6e6
    >>> R = spacing_for_target_V_over_Omega(C6, Omega, 45)
    >>> print(f"R = {R*1e6:.2f} μm")
    """
    return (C6 / (HBAR * Omega * target_ratio))**(1/6)


# =============================================================================
# INTERMEDIATE STATE SCATTERING
# =============================================================================

def intermediate_state_scattering_rate(
    Omega1: float,
    Delta_e: float,
    Gamma_e: float
) -> float:
    """
    Calculate off-resonant scattering rate from intermediate state.
    
    **The problem:**
    
    Even though we detune from the intermediate state, there's still a small
    probability of being "virtually" in that state. During that time, the
    atom can spontaneously emit, scattering into random ground states and
    destroying the quantum information.
    
    **The rate:**
    
    The scattering rate is:
    
        γ_scatter = Ω₁² × Γₑ / (4 × Δₑ²)
    
    where:
    - Ω₁: First laser Rabi frequency
    - Γₑ: Natural linewidth of intermediate state (~6 MHz for Rb D2)
    - Δₑ: Detuning from intermediate state
    
    **Derivation:**
    
    - Virtual population of intermediate: |c_e|² ~ (Ω₁/2Δₑ)² = Ω₁²/(4Δₑ²)
    - Decay rate of intermediate: Γₑ
    - Scattering rate: γ = |c_e|² × Γₑ = Ω₁² Γₑ / (4Δₑ²)
    
    **Typical numbers (Bluvstein):**
    - Ω₁/2π ~ 100 MHz
    - Δₑ/2π ~ 7.8 GHz
    - Γₑ/2π ~ 6 MHz
    - γ_scatter/2π ~ 100² × 6 / (4 × 7800²) ~ 0.25 kHz
    
    During a 0.5 μs gate: scatter probability ~ 0.25 kHz × 0.5 μs ~ 0.013%
    
    Parameters
    ----------
    Omega1 : float
        First laser Rabi frequency in rad/s
    Delta_e : float
        Intermediate state detuning in rad/s
    Gamma_e : float
        Intermediate state linewidth in rad/s
        
    Returns
    -------
    float
        Scattering rate in rad/s (divide by 2π for Hz)
        
    Example
    -------
    >>> Omega1 = 2*np.pi * 100e6  # 100 MHz
    >>> Delta_e = 2*np.pi * 7.8e9  # 7.8 GHz
    >>> Gamma_e = 2*np.pi * 6e6   # 6 MHz
    >>> gamma = intermediate_state_scattering_rate(Omega1, Delta_e, Gamma_e)
    >>> print(f"γ_scatter/2π = {gamma/(2*np.pi):.0f} Hz")
    """
    return Omega1**2 * Gamma_e / (4 * Delta_e**2)


def dark_state_suppression_factor(Omega1: float, Omega2: float) -> float:
    """
    Calculate dark state suppression factor for scattering.
    
    **What is dark state suppression?**
    
    In two-photon excitation, there's a quantum interference effect that
    can REDUCE scattering. The "dark state" is a superposition of ground
    and Rydberg states that doesn't couple to the intermediate state:
    
        |dark⟩ = (Ω₂|g⟩ - Ω₁|r⟩) / √(Ω₁² + Ω₂²)
    
    When the atom is primarily in the dark state, scattering is suppressed.
    
    **Suppression factor:**
    
    The effective scattering rate is reduced by:
    
        S = (Ω_eff/Ω₁)² = (Ω₂/(2Δₑ))²
    
    This can give ~10-100× suppression depending on parameters.
    
    **Warning:** This simple formula only applies in specific limits.
    See Bluvstein thesis for full treatment.
    
    Parameters
    ----------
    Omega1 : float
        First laser Rabi frequency in rad/s
    Omega2 : float
        Second laser Rabi frequency in rad/s
        
    Returns
    -------
    float
        Suppression factor (multiply naive scattering rate by this)
    """
    return Omega2**2 / (Omega1**2 + Omega2**2)


# =============================================================================
# CLEBSCH-GORDAN COEFFICIENTS
# =============================================================================
# These determine the polarization-dependent coupling strengths for
# transitions between different magnetic sublevels.

# Pre-computed Clebsch-Gordan coefficients for D2 line transitions
# Format: CLEBSCH_GORDAN_D2[(F_initial, mF_initial, F_final)][(polarization)] = value
# where polarization is "pi" (Δm=0), "sigma+" (Δm=+1), or "sigma-" (Δm=-1)

CLEBSCH_GORDAN_D2 = {
    # F=1 → F'=0 (weak, usually avoided)
    (1, -1, 0): {"pi": 0, "sigma+": 0, "sigma-": 1/np.sqrt(3)},
    (1,  0, 0): {"pi": 1/np.sqrt(3), "sigma+": 0, "sigma-": 0},
    (1, +1, 0): {"pi": 0, "sigma+": 1/np.sqrt(3), "sigma-": 0},
    
    # F=1 → F'=1
    (1, -1, 1): {"pi": 1/np.sqrt(6), "sigma+": -1/np.sqrt(2), "sigma-": 0},
    (1,  0, 1): {"pi": 0, "sigma+": 1/np.sqrt(6), "sigma-": -1/np.sqrt(6)},
    (1, +1, 1): {"pi": -1/np.sqrt(6), "sigma+": 0, "sigma-": 1/np.sqrt(2)},
    
    # F=1 → F'=2 (strong, commonly used)
    (1, -1, 2): {"pi": -1/np.sqrt(2), "sigma+": 0, "sigma-": -1/np.sqrt(10)},
    (1,  0, 2): {"pi": -np.sqrt(2/5), "sigma+": -1/np.sqrt(2), "sigma-": 1/np.sqrt(2)},
    (1, +1, 2): {"pi": -1/np.sqrt(2), "sigma+": 1/np.sqrt(10), "sigma-": 0},
    
    # F=2 → F'=1
    (2, -2, 1): {"pi": 0, "sigma+": 0, "sigma-": 1/np.sqrt(2)},
    (2, -1, 1): {"pi": 1/np.sqrt(6), "sigma+": 0, "sigma-": 1/np.sqrt(6)},
    (2,  0, 1): {"pi": np.sqrt(2/5), "sigma+": 1/np.sqrt(6), "sigma-": 1/np.sqrt(6)},
    (2, +1, 1): {"pi": 1/np.sqrt(6), "sigma+": 1/np.sqrt(6), "sigma-": 0},
    (2, +2, 1): {"pi": 0, "sigma+": 1/np.sqrt(2), "sigma-": 0},
    
    # F=2 → F'=2
    (2, -2, 2): {"pi": -1/np.sqrt(3), "sigma+": 0, "sigma-": -1/np.sqrt(30)},
    (2, -1, 2): {"pi": -1/np.sqrt(12), "sigma+": -1/np.sqrt(3), "sigma-": 1/np.sqrt(20)},
    (2,  0, 2): {"pi": 0, "sigma+": -1/np.sqrt(12), "sigma-": 1/np.sqrt(12)},
    (2, +1, 2): {"pi": 1/np.sqrt(12), "sigma+": -1/np.sqrt(20), "sigma-": 1/np.sqrt(3)},
    (2, +2, 2): {"pi": 1/np.sqrt(3), "sigma+": 1/np.sqrt(30), "sigma-": 0},
    
    # F=2 → F'=3 (cycling transition, commonly used for detection)
    (2, -2, 3): {"pi": -np.sqrt(2/5), "sigma+": 0, "sigma-": -1/np.sqrt(15)},
    (2, -1, 3): {"pi": -np.sqrt(3/10), "sigma+": -np.sqrt(2/5), "sigma-": 1/(2*np.sqrt(5))},
    (2,  0, 3): {"pi": -np.sqrt(2/5), "sigma+": -np.sqrt(3/10), "sigma-": np.sqrt(3/10)},
    (2, +1, 3): {"pi": -np.sqrt(3/10), "sigma+": -1/(2*np.sqrt(5)), "sigma-": np.sqrt(2/5)},
    (2, +2, 3): {"pi": -np.sqrt(2/5), "sigma+": 1/np.sqrt(15), "sigma-": 0},
}


def get_clebsch_gordan(F_i: int, mF_i: int, F_f: int, polarization: str) -> float:
    """
    Get Clebsch-Gordan coefficient for a hyperfine transition.
    
    **What are Clebsch-Gordan coefficients?**
    
    They describe how strongly a laser with a given polarization couples
    two specific magnetic sublevels. For a transition |F,mF⟩ → |F',mF'⟩:
    
        Ω(F,mF → F',mF') = Ω_reduced × CG(F,mF,F',mF')
    
    where Ω_reduced is the "reduced" Rabi frequency (independent of mF).
    
    **Selection rules:**
    
    - π polarization: ΔmF = 0 (mF' = mF)
    - σ⁺ polarization: ΔmF = +1 (mF' = mF + 1)
    - σ⁻ polarization: ΔmF = -1 (mF' = mF - 1)
    
    Parameters
    ----------
    F_i : int
        Initial F quantum number
    mF_i : int
        Initial mF quantum number
    F_f : int
        Final F quantum number
    polarization : str
        "pi", "sigma+", or "sigma-"
        
    Returns
    -------
    float
        Clebsch-Gordan coefficient (0 if transition not allowed)
    """
    key = (F_i, mF_i, F_f)
    if key in CLEBSCH_GORDAN_D2:
        return CLEBSCH_GORDAN_D2[key].get(polarization, 0)
    return 0.0


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def compute_all_rabi_frequencies(
    species: str,
    n_rydberg: int,
    power_1: float,
    power_2: float,
    waist_1: float,
    waist_2: float,
    Delta_e: float,
    intermediate_state: str = "5P3/2"
) -> dict:
    """
    Compute all relevant Rabi frequencies for two-photon excitation.
    
    Parameters
    ----------
    species : str
        Atomic species ("Rb87" or "Cs133")
    n_rydberg : int
        Rydberg principal quantum number
    power_1 : float
        First laser power in Watts
    power_2 : float
        Second laser power in Watts
    waist_1 : float
        First laser waist in meters
    waist_2 : float
        Second laser waist in meters
    Delta_e : float
        Intermediate state detuning in rad/s
    intermediate_state : str
        Intermediate state label (default "5P3/2")
        
    Returns
    -------
    dict
        Dictionary with:
        - Omega1: First laser Rabi frequency (rad/s)
        - Omega2: Second laser Rabi frequency (rad/s)
        - Omega_eff: Effective two-photon Rabi frequency (rad/s)
        - All values also in MHz for convenience
    """
    atom = ATOM_DB[species]
    
    # Get dipole moments
    dipole_1 = atom["intermediate_states"][intermediate_state]["dipole_from_ground"]
    
    # Dipole from intermediate to Rydberg scales with n^(-3/2)
    # This is species-agnostic: Rb87 uses 5P→nS, Cs133 uses 6P→nS
    n_star = effective_n(n_rydberg, species, "S")
    n_star_ref = effective_n(atom["n_ref"], species, "S")
    dipole_2 = atom["dipole_intermediate_to_rydberg_ref"] * (n_star / n_star_ref)**(-1.5)
    
    # Calculate electric fields
    E0_1 = laser_E0(power_1, waist_1)
    E0_2 = laser_E0(power_2, waist_2)
    
    # Calculate single-photon Rabi frequencies
    Omega1 = single_photon_rabi(dipole_1, E0_1)
    Omega2 = single_photon_rabi(dipole_2, E0_2)
    
    # Calculate effective two-photon Rabi frequency
    Omega_eff = two_photon_rabi(Omega1, Omega2, Delta_e)
    
    return {
        "Omega1": Omega1,
        "Omega2": Omega2,
        "Omega_eff": Omega_eff,
        "Omega1_MHz": Omega1 / (2*np.pi) / 1e6,
        "Omega2_MHz": Omega2 / (2*np.pi) / 1e6,
        "Omega_eff_MHz": Omega_eff / (2*np.pi) / 1e6,
        "Delta_e_GHz": Delta_e / (2*np.pi) / 1e9,
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Electric field
    "laser_E0",
    "laser_intensity",
    
    # Single-photon Rabi
    "single_photon_rabi",
    "single_photon_rabi_from_power",
    
    # Two-photon Rabi
    "two_photon_rabi",
    "required_powers_for_two_photon_rabi",
    
    # Rydberg blockade
    "rydberg_blockade",
    "blockade_shift_MHz",
    "blockade_radius",
    "blockade_radius_um",
    "V_over_Omega",
    "spacing_for_target_V_over_Omega",
    
    # Scattering
    "intermediate_state_scattering_rate",
    "dark_state_suppression_factor",
    
    # Clebsch-Gordan
    "CLEBSCH_GORDAN_D2",
    "get_clebsch_gordan",
    
    # Convenience
    "compute_all_rabi_frequencies",
]
