"""
Noise Models for Rydberg Gate Simulations
==========================================

This module implements realistic noise models that limit the fidelity of
neutral atom Rydberg gates. Understanding these noise sources is crucial
for predicting gate performance and guiding experimental improvements.

WHY DO WE NEED NOISE MODELS?
----------------------------

Without noise, simulations would predict >99.99% gate fidelity.
Real experiments achieve 97-99.5%. The gap comes from many physical
decoherence mechanisms, each contributing a small but significant error.

**Bluvstein thesis (Harvard 2024) breakdown at n=53, Ω/2π=4.6 MHz:**

| Noise Source                    | Infidelity (%) |
|---------------------------------|----------------|
| Rydberg spontaneous decay       | 0.035          |
| Intermediate state scattering   | 0.043          |
| Doppler dephasing               | 0.056          |
| Laser phase noise               | 0.050          |
| Atom temperature                | ~0.1-0.3       |
| **Total (estimated)**           | ~0.5-1%        |
| **Measured**                    | 0.5%           |

THE LINDBLAD MASTER EQUATION
----------------------------

We model noise using the Lindblad master equation:

    dρ/dt = -i[H, ρ]/ℏ + Σⱼ γⱼ (Lⱼ ρ Lⱼ† - ½{Lⱼ†Lⱼ, ρ})

where:
- ρ is the density matrix (quantum state)
- H is the system Hamiltonian (coherent evolution)
- Lⱼ are "collapse operators" representing different noise channels
- γⱼ are the corresponding rates

**Physical interpretation:** Each collapse operator Lⱼ describes a process
that can occur at rate γⱼ:
- L = |1⟩⟨r| with γ_r → spontaneous decay from Rydberg to qubit
- L = |r⟩⟨r| with γ_φ → pure dephasing of Rydberg coherence

NOISE SOURCES IN THIS MODULE
----------------------------

1. **Rydberg spontaneous emission** (γ_r)
   - Rydberg atom spontaneously emits photon and decays
   - Rate: γ_r = 1/τ_rydberg ∝ n⁻³
   - Typical: ~10 kHz at n=70

2. **Blackbody radiation (BBR) decay** (γ_bbr)
   - Thermal photons from environment stimulate transitions
   - Rate: γ_bbr ∝ T⁴ × n⁻² at high temperature
   - At 300K, roughly doubles total decay rate

3. **Laser phase noise dephasing** (γ_φ_laser)
   - Finite laser linewidth causes random phase accumulation
   - Rate: γ_φ = 2π × (linewidth_Hz)
   - Typical: ~1 kHz for good lasers

4. **Thermal motion dephasing** (γ_φ_thermal)
   - Position fluctuations cause blockade interaction fluctuations
   - Rate depends on temperature and V/Ω ratio
   - Can be dominant at high temperature!

5. **Zeeman dephasing** (γ_φ_zeeman)
   - B-field fluctuations cause differential Zeeman shifts
   - Depends on qubit encoding (clock vs non-clock states)
   - Clock states: ~100 Hz at 1 mG noise

6. **Anti-trap atom loss** (γ_loss_antitrap)
   - Rydberg state repelled from optical trap → atom escapes
   - Depends on trap depth and gate duration
   - Typical: ~10-50 kHz

7. **Background gas collisions** (γ_loss_bg)
   - Collisions with residual gas eject atoms
   - Rate: ~0.1-1 Hz at 10⁻¹¹ Torr (usually negligible)

8. **Intermediate state scattering** (γ_scatter)
   - Off-resonant scattering from intermediate 5P/6P state
   - Rate: γ = Γ_e × Ω₁²/(4Δₑ²)
   - Can be suppressed by dark-state physics!

9. **Rydberg state leakage** (γ_leakage)
   - Off-resonant excitation to nearby n±1 states
   - Suppressed by pulse shaping (Gaussian, Blackman)
   - Typical: <0.01% with shaped pulses

10. **mJ state mixing** (γ_mJ)
    - Polarization impurity drives wrong Zeeman transition
    - Depends on polarization purity and B-field

References
----------
[1] de Léséleuc et al., PRA 97, 053803 (2018) - Error analysis
[2] Levine et al., PRL 123, 170503 (2019) - High-fidelity gates
[3] Bluvstein PhD Thesis (Harvard 2024) - Comprehensive error budget
[4] Saffman et al., RMP 82, 2313 (2010) - Rydberg physics review

MODULE ARCHITECTURE
===================

This module is organized into TWO parts that work together:

**PART 1: RATE CALCULATORS (lines ~115-1040)**
    Functions that compute noise RATES (γ values in rad/s) from physical parameters.
    These are pure calculations - no QuTiP objects involved.
    
    Key functions:
    - `rydberg_decay_rate()` - Spontaneous emission γ_r from lifetime
    - `bbr_decay_rate()` - Blackbody radiation contribution
    - `laser_dephasing_rate()` - Dephasing from laser linewidth
    - `zeeman_dephasing_rate()` - B-field noise contribution
    - `intermediate_state_scattering_rate()` - Off-resonant scattering
    - `leakage_rate_to_adjacent_states()` - Spectral leakage
    - `mJ_mixing_rate()` - Polarization impurity effects
    - `compute_noise_rates()` - UNIFIED: computes ALL rates, returns NoiseRates dataclass

**PART 2: COLLAPSE OPERATOR BUILDERS (lines ~1040-end)**
    Functions that build QuTiP Qobj collapse operators from rates.
    These create the Lⱼ operators for Lindblad master equation.
    
    Key functions:
    - `op_two_atom()` - Tensor product helper
    - `build_decay_operators()` - |r⟩ → |1⟩/|0⟩ decay operators
    - `build_dephasing_operators()` - Pure dephasing on Rydberg
    - `build_loss_operators()` - Atom loss channels
    - `build_scatter_operators()` - Intermediate state scattering
    - `build_all_noise_operators()` - UNIFIED: builds ALL c_ops from rates

USAGE WORKFLOW
--------------

The standard workflow for adding noise to a simulation:

    # Step 1: Get trap-dependent noise rates (from trap_physics.py)
    from .trap_physics import compute_trap_dependent_noise
    trap_noise = compute_trap_dependent_noise(
        species="Rb87", tweezer_power=30e-3, tweezer_waist=1e-6,
        temperature=2e-6, spacing=3e-6, gate_time=0.5e-6, ...
    )
    
    # Step 2: Build collapse operators (from this module)
    from .noise_models import build_all_noise_operators
    c_ops, noise_breakdown = build_all_noise_operators(
        hs=hilbert_space,
        gamma_r=trap_noise['gamma_r'],
        gamma_phi_laser=trap_noise['gamma_phi_laser'],
        gamma_phi_thermal=trap_noise['gamma_phi_thermal'],
        gamma_loss_antitrap=trap_noise['gamma_loss_antitrap'],
        ...
    )
    
    # Step 3: Use in mesolve
    result = qutip.mesolve(H, psi0, tlist, c_ops=c_ops)

WHY TWO MODULES?
----------------

- `trap_physics.py` handles TRAP-DEPENDENT noise: rates that depend on
  tweezer parameters (power, waist, wavelength), temperature, spacing.
  It uses trap_depth(), trap_frequencies(), polarizabilities, etc.
  
- `noise_models.py` handles RATE-TO-OPERATOR conversion: given noise rates,
  build the appropriate QuTiP collapse operators for the Lindblad equation.
  
This separation keeps trap physics (classical EM + thermodynamics) separate
from quantum operator construction (QuTiP Qobj manipulation).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass

from .constants import HBAR, KB, MU_B, EPS0, C


# =============================================================================
# PART 1: DATA CLASSES AND RATE CALCULATORS
# =============================================================================
#
# This section computes noise RATES (γ in rad/s) from physical parameters.
# All functions are pure calculations with no QuTiP dependencies.
# =============================================================================

@dataclass
class NoiseRates:
    """
    Container for all noise rates used in Lindblad master equation.
    
    All rates are in rad/s (angular frequency units).
    To convert to Hz: f = γ / (2π)
    
    Attributes
    ----------
    gamma_r : float
        Rydberg spontaneous emission rate (rad/s)
    gamma_bbr : float
        Blackbody radiation-induced decay rate (rad/s)
    gamma_phi_laser : float
        Laser phase noise dephasing rate (rad/s)
    gamma_phi_thermal : float
        Thermal motion dephasing rate (rad/s)
    gamma_phi_zeeman : float
        Zeeman dephasing rate from B-field noise (rad/s)
    gamma_loss_antitrap : float
        Anti-trapping atom loss rate (rad/s)
    gamma_loss_bg : float
        Background gas collision loss rate (rad/s)
    gamma_scatter : float
        Intermediate state scattering rate (rad/s)
    gamma_leakage : float
        Leakage to adjacent Rydberg states (rad/s)
    gamma_mJ : float
        mJ state mixing rate (rad/s)
    """
    gamma_r: float = 0.0
    gamma_bbr: float = 0.0
    gamma_phi_laser: float = 0.0
    gamma_phi_thermal: float = 0.0
    gamma_phi_zeeman: float = 0.0
    gamma_loss_antitrap: float = 0.0
    gamma_loss_bg: float = 0.0
    gamma_scatter: float = 0.0
    gamma_leakage: float = 0.0
    gamma_mJ: float = 0.0
    
    @property
    def total_decay_rate(self) -> float:
        """Total population decay rate from Rydberg state."""
        return (self.gamma_r + self.gamma_bbr + self.gamma_loss_antitrap + 
                self.gamma_loss_bg + self.gamma_scatter + self.gamma_leakage)
    
    @property
    def total_dephasing_rate(self) -> float:
        """Total pure dephasing rate."""
        return (self.gamma_phi_laser + self.gamma_phi_thermal + 
                self.gamma_phi_zeeman + self.gamma_mJ)
    
    @property
    def total_T2_rate(self) -> float:
        """Total T2 decoherence rate (T2 = 1 / (γ_decay/2 + γ_dephasing))."""
        return 0.5 * self.total_decay_rate + self.total_dephasing_rate
    
    def to_dict(self) -> dict:
        """Convert to dictionary for easy inspection."""
        return {
            "gamma_r": self.gamma_r,
            "gamma_bbr": self.gamma_bbr,
            "gamma_phi_laser": self.gamma_phi_laser,
            "gamma_phi_thermal": self.gamma_phi_thermal,
            "gamma_phi_zeeman": self.gamma_phi_zeeman,
            "gamma_loss_antitrap": self.gamma_loss_antitrap,
            "gamma_loss_bg": self.gamma_loss_bg,
            "gamma_scatter": self.gamma_scatter,
            "gamma_leakage": self.gamma_leakage,
            "gamma_mJ": self.gamma_mJ,
            "total_decay": self.total_decay_rate,
            "total_dephasing": self.total_dephasing_rate,
        }
    
    def summary_table(self, gate_time: float = 1e-6) -> str:
        """Generate a formatted summary table."""
        lines = [
            "=" * 60,
            "NOISE RATE SUMMARY",
            "=" * 60,
            f"{'Source':<30} {'Rate (kHz)':<15} {'Error/gate (%)'}",
            "-" * 60,
        ]
        
        rates = [
            ("Rydberg decay", self.gamma_r),
            ("BBR decay", self.gamma_bbr),
            ("Laser dephasing", self.gamma_phi_laser),
            ("Thermal dephasing", self.gamma_phi_thermal),
            ("Zeeman dephasing", self.gamma_phi_zeeman),
            ("Anti-trap loss", self.gamma_loss_antitrap),
            ("Background loss", self.gamma_loss_bg),
            ("Int. state scatter", self.gamma_scatter),
            ("Rydberg leakage", self.gamma_leakage),
            ("mJ mixing", self.gamma_mJ),
        ]
        
        for name, rate in rates:
            rate_kHz = rate / (2 * np.pi * 1e3)
            error_pct = rate * gate_time * 100
            lines.append(f"{name:<30} {rate_kHz:<15.2f} {error_pct:.3f}")
        
        lines.extend([
            "-" * 60,
            f"{'TOTAL DECAY':<30} {self.total_decay_rate/(2*np.pi*1e3):<15.2f} "
            f"{self.total_decay_rate*gate_time*100:.3f}",
            f"{'TOTAL DEPHASING':<30} {self.total_dephasing_rate/(2*np.pi*1e3):<15.2f} "
            f"{self.total_dephasing_rate*gate_time*100:.3f}",
            "=" * 60,
        ])
        
        return "\n".join(lines)


# =============================================================================
# RYDBERG SPONTANEOUS EMISSION
# =============================================================================

def rydberg_decay_rate(lifetime: float) -> float:
    """
    Calculate Rydberg spontaneous emission rate from lifetime.
    
    **What is spontaneous emission?**
    
    A Rydberg atom can spontaneously emit a photon and decay to a lower
    state. This is a fundamental quantum process that cannot be avoided -
    even in a perfect vacuum, quantum fluctuations of the electromagnetic
    field drive the decay.
    
    For Rydberg atoms, the decay is to states with lower n, typically
    through a cascade: |70S⟩ → |60P⟩ → |50S⟩ → ... → ground
    
    **Why does lifetime scale as n³?**
    
    The decay rate depends on:
    - Transition dipole moment: d ∝ n²
    - Transition frequency: ω ∝ n⁻³
    - γ ∝ ω³ × d² ∝ n⁻³ × n⁴ = n⁻³ ... wait, that gives n+!
    
    Actually, the dominant decay is to the nearest lower state, with
    ω ~ 1/n³ and d ~ n², giving γ ∝ ω³d² ∝ n⁻⁹n⁴ = n⁻⁵? 
    
    The correct answer comes from summing over ALL decay channels:
    τ ∝ n³ (empirically observed and theoretically derived).
    
    **Practical numbers:**
    - n=50: τ ~ 50 μs
    - n=70: τ ~ 140 μs (at 300K)
    - n=100: τ ~ 400 μs
    
    Parameters
    ----------
    lifetime : float
        Rydberg state lifetime (s). Get from atom_database.get_rydberg_lifetime()
        
    Returns
    -------
    float
        Decay rate γ_r = 1/τ (rad/s)
        
    Example
    -------
    >>> tau = get_rydberg_lifetime(70, "Rb87", 300)  # ~140 μs
    >>> gamma_r = rydberg_decay_rate(tau)
    >>> gamma_r / (2*np.pi) / 1e3  # Convert to kHz
    ~1.1  # ~1 kHz decay rate
    """
    if lifetime <= 0:
        return 0.0
    return 1.0 / lifetime


def bbr_decay_rate(lifetime_0K: float, lifetime_T: float) -> float:
    """
    Extract the blackbody radiation contribution to decay rate.
    
    **What is BBR decay?**
    
    At any temperature T > 0, there are thermal photons everywhere.
    These photons can stimulate transitions between Rydberg states,
    effectively causing "decay" (really redistribution) of population.
    
    At room temperature (300 K), the thermal photon density peaks at
    ~18 μm wavelength, which is resonant with transitions between
    nearby Rydberg states (ΔE ~ 50-100 GHz for n~50-100).
    
    **Temperature dependence:**
    
    The BBR rate scales as T⁴ for T >> T_rydberg (always true for us):
    γ_bbr(T) ∝ T⁴ × n⁻² (roughly)
    
    At T = 300 K: BBR roughly DOUBLES the decay rate compared to T = 0K!
    At T = 4 K (cryostat): BBR is negligible
    
    Parameters
    ----------
    lifetime_0K : float
        Lifetime at zero temperature (spontaneous only)
    lifetime_T : float
        Lifetime at finite temperature (spontaneous + BBR)
        
    Returns
    -------
    float
        BBR decay rate contribution γ_bbr (rad/s)
        
    Example
    -------
    >>> tau_0K = 280e-6  # 280 μs at T=0
    >>> tau_300K = 140e-6  # 140 μs at T=300K
    >>> gamma_bbr = bbr_decay_rate(tau_0K, tau_300K)
    >>> gamma_bbr / (2*np.pi) / 1e3  # kHz
    ~0.6  # BBR contributes ~0.6 kHz at 300K
    """
    if lifetime_0K <= 0 or lifetime_T <= 0:
        return 0.0
    
    gamma_total = 1.0 / lifetime_T
    gamma_spontaneous = 1.0 / lifetime_0K
    gamma_bbr = max(0.0, gamma_total - gamma_spontaneous)
    
    return gamma_bbr


# =============================================================================
# LASER PHASE NOISE DEPHASING
# =============================================================================

def laser_dephasing_rate(linewidth_hz: float) -> float:
    """
    Calculate dephasing rate from laser phase noise (finite linewidth).
    
    **What is laser phase noise?**
    
    Real lasers don't have perfectly stable frequency - they jitter
    randomly around a central frequency. This is characterized by the
    "linewidth" Δf, which is the FWHM of the laser spectrum.
    
    When driving a qubit transition, this phase noise transfers to
    the atomic superposition state, causing dephasing.
    
    **The formula:**
    
    For a Lorentzian lineshape (typical for lasers):
    γ_φ = π × Δf_FWHM = 2π × Δf_laser
    
    where Δf_laser is the single-sided spectral density at low frequency.
    
    In practice, for the effective two-photon excitation:
    γ_φ_total = γ_φ,laser1 + γ_φ,laser2 (linewidths add for independent lasers)
    
    **Practical numbers:**
    - Commercial diode laser: Δf ~ 1 MHz → γ_φ ~ 2π × 1 MHz (terrible!)
    - ECDL with reference cavity: Δf ~ 1-10 kHz → γ_φ ~ 2π × 1-10 kHz
    - Ultra-stable cavity locked: Δf ~ 100 Hz → γ_φ ~ 2π × 100 Hz
    
    **Two-photon benefit:**
    If both lasers are locked to the SAME reference, their phase noise
    can be common-mode rejected. Only the DIFFERENCE frequency matters.
    This can reduce effective linewidth by 10-100×!
    
    Parameters
    ----------
    linewidth_hz : float
        Effective laser linewidth in Hz (FWHM for Lorentzian)
        For two-photon: use the DIFFERENCE frequency linewidth
        
    Returns
    -------
    float
        Dephasing rate γ_φ (rad/s)
        
    Example
    -------
    >>> # Two independent lasers, each 5 kHz linewidth
    >>> gamma = laser_dephasing_rate(5e3 + 5e3)  # 10 kHz total
    >>> gamma / (2*np.pi) / 1e3  # kHz
    10.0
    
    >>> # Same lasers locked to same reference (100× common-mode rejection)
    >>> gamma_locked = laser_dephasing_rate(100)  # 100 Hz effective
    >>> gamma_locked / (2*np.pi)  # Hz
    100.0
    """
    return 2 * np.pi * linewidth_hz


# =============================================================================
# ZEEMAN DEPHASING
# =============================================================================

def zeeman_dephasing_rate(B_noise_gauss: float, qubit_type: str = "clock",
                          K_quad: float = 575.0) -> float:
    """
    Calculate dephasing rate from magnetic field fluctuations.
    
    **What is Zeeman dephasing?**
    
    In a magnetic field, atomic energy levels shift (Zeeman effect).
    If the field fluctuates, this causes time-varying energy shifts,
    which accumulate as random phase → dephasing.
    
    **Clock states vs stretched states:**
    
    CLOCK STATES: |F=1, mF=0⟩ ↔ |F=2, mF=0⟩
    - Both states have mF = 0
    - Linear Zeeman shift is ZERO (no ΔmF)
    - Only quadratic Zeeman shift matters: ΔE ∝ B²
    - First-order insensitive to B-field!
    - For Rb87: K_quad = 575 Hz/G²
    - At B = 1 G, noise δB = 1 mG: δf = 2 × 575 × 1 × 0.001 = 1.15 Hz
    
    STRETCHED STATES: e.g., |F=2, mF=-2⟩ ↔ |F=2, mF=-1⟩ (within same F)
    - Linear Zeeman shift: ΔE = g_F × μ_B × B × ΔmF
    - For Rb87 F=2: g_F = 1/2, so shift = 700 kHz/G per ΔmF
    - Much more sensitive to B-field noise!
    
    Parameters
    ----------
    B_noise_gauss : float
        RMS magnetic field fluctuation (Gauss)
        Typical lab: ~1-10 mG
        Magnetically shielded: ~0.1-1 mG
    qubit_type : str
        "clock" for mF=0 ↔ mF=0 (quadratic only)
        "stretched" for linear Zeeman sensitive states
    K_quad : float
        Quadratic Zeeman coefficient (Hz/G²)
        Rb87: 575 Hz/G²
        Cs133: 427 Hz/G²
        
    Returns
    -------
    float
        Dephasing rate γ_φ (rad/s)
        
    Example
    -------
    >>> # Clock states with 1 mG noise at 1 G bias
    >>> gamma_clock = zeeman_dephasing_rate(0.001, "clock")
    >>> gamma_clock / (2*np.pi)  # Hz
    ~1  # Very small - clock states are robust!
    
    >>> # Stretched states with same noise
    >>> gamma_stretched = zeeman_dephasing_rate(0.001, "stretched")
    >>> gamma_stretched / (2*np.pi)  # Hz
    ~700  # 700× larger - sensitive to noise!
    """
    if qubit_type == "clock":
        # Quadratic Zeeman: df = 2 × K_quad × B × δB
        # Need to know bias field for this... assume B_bias = 1 G typical
        B_bias = 1.0  # Gauss
        df_Hz = 2 * K_quad * B_bias * B_noise_gauss  # Hz
        
    elif qubit_type == "stretched":
        # Linear Zeeman: df = g_F × μ_B × δB / h ≈ 700 kHz/G × δB
        # For Rb87 F=2: g_F = 0.5, so ΔmF=1 transition
        df_Hz = 700e3 * B_noise_gauss  # Hz (for ΔmF = 1)
        
    else:
        raise ValueError(f"Unknown qubit_type: {qubit_type}")
    
    return 2 * np.pi * df_Hz


# =============================================================================
# INTERMEDIATE STATE SCATTERING
# =============================================================================

def intermediate_state_scattering_rate(Omega_1: float, Delta_e: float,
                                        Gamma_e: float) -> float:
    """
    Calculate off-resonant scattering rate from intermediate excited state.
    
    **The physics:**
    
    In two-photon Rydberg excitation:
        |g⟩ --Ω₁--> |e⟩ --Ω₂--> |r⟩
                   ↑
               detuned by Δₑ
    
    Even though we're detuned from |e⟩, there's still some probability
    of "virtually" populating it. This virtual population can scatter
    a photon (with rate Γₑ), causing:
    - Loss of coherence
    - Population in wrong state (50% chance of going to F=1 vs F=2)
    
    **The formula (Lorentzian):**
    
        γ_scatter = Γₑ × (Ω₁/2)² / (Δₑ² + (Γₑ/2)²)
        
    For large detuning (Δₑ >> Γₑ):
        γ_scatter ≈ Γₑ × Ω₁² / (4Δₑ²)
    
    **Practical numbers:**
    For Rb87 with Ω₁/2π = 100 MHz, Δₑ/2π = 1 GHz:
        γ_scatter/2π = 6 MHz × (100 MHz)² / (4 × (1 GHz)²)
                     = 6 MHz × 0.01 / 4 = 15 kHz
    
    This is comparable to Rydberg decay - a significant error source!
    
    **Mitigation:**
    1. Increase Δₑ → but need more laser power for same Ω_eff
    2. Use dark-state physics (see dark_state_suppression_factor)
    3. Shorter gate times (less time to scatter)
    
    Parameters
    ----------
    Omega_1 : float
        First-leg Rabi frequency |g⟩ → |e⟩ (rad/s)
    Delta_e : float
        Intermediate state detuning (rad/s)
    Gamma_e : float
        Intermediate state linewidth (rad/s)
        Rb87 5P₃/₂: Γₑ = 2π × 6.065 MHz
        
    Returns
    -------
    float
        Scattering rate γ_scatter (rad/s)
    """
    # Full Lorentzian formula
    Omega_1_half_sq = (Omega_1 / 2)**2
    denominator = Delta_e**2 + (Gamma_e / 2)**2
    
    return Gamma_e * Omega_1_half_sq / denominator


def dark_state_suppression_factor(Delta_e: float, delta: float,
                                   Omega_1: float, Omega_2: float) -> float:
    """
    Calculate scattering suppression from dark-state physics.
    
    **What are dark/bright states?**
    
    In a three-level system |g⟩, |e⟩, |r⟩, we can form superposition states:
    
    Dark state:   |D⟩ = cos(θ)|g⟩ - sin(θ)|r⟩  (no |e⟩ component!)
    Bright state: |B⟩ = sin(θ)|g⟩ + cos(θ)|r⟩  (couples to |e⟩)
    
    where tan(θ) = Ω₂/Ω₁
    
    **Why does this matter?**
    
    If we can arrange for the atom to evolve primarily in the dark state,
    it never populates |e⟩ → no scattering from |e⟩!
    
    **The key insight (Bluvstein thesis):**
    
    The dressed state adiabatically connected to |g⟩ has different |e⟩
    admixture depending on the SIGN of Δₑ × δ:
    
    - When sign(Δₑ) = sign(δ): "Dark configuration" → suppressed scattering
    - When sign(Δₑ) ≠ sign(δ): "Bright configuration" → enhanced scattering
    
    **Experimental choice:**
    Bluvstein uses Δₑ > 0 (blue detuned) with δ > 0 (optimal LP detuning)
    This gives ~2× suppression of scattering!
    
    Parameters
    ----------
    Delta_e : float
        Intermediate state detuning (rad/s), positive = blue-detuned
    delta : float
        Two-photon detuning (rad/s)
    Omega_1, Omega_2 : float
        Single-photon Rabi frequencies (rad/s)
        
    Returns
    -------
    float
        Suppression factor (0-1). Multiply naive rate by this.
        ~0.42 for dark configuration (matches Bluvstein thesis)
        ~1.0-1.2 for bright configuration
    """
    # Effective two-photon Rabi frequency
    Omega_eff = Omega_1 * Omega_2 / (2 * np.abs(Delta_e))
    
    # Check if in dark configuration
    is_dark = np.sign(Delta_e) == np.sign(delta) and np.abs(delta) > 1e-6
    
    if is_dark:
        # Dark state suppression
        # Empirical fit to Bluvstein thesis: 0.103% → 0.043% = 0.42× suppression
        ratio = np.abs(delta) / (np.abs(Omega_eff) + 1e-10)
        if 0.1 < ratio < 2.0:
            suppression = 0.42  # Optimal regime
        else:
            # Outside optimal, interpolate toward 1.0
            suppression = 0.42 + 0.58 * (1 - np.exp(-np.abs(ratio - 0.377) / 0.5))
    else:
        # Bright configuration - no suppression
        suppression = 1.0
    
    return min(suppression, 1.5)


def enhanced_scattering_rate(Omega_1: float, Omega_2: float, Delta_e: float,
                              delta: float, Gamma_e: float,
                              use_dark_state: bool = True) -> float:
    """
    Calculate scattering rate with dark-state physics.
    
    This combines the basic scattering rate with dark-state suppression
    when the detuning signs are chosen correctly.
    
    Parameters
    ----------
    Omega_1 : float
        First-leg Rabi frequency (rad/s)
    Omega_2 : float
        Second-leg Rabi frequency (rad/s)
    Delta_e : float
        Intermediate state detuning (rad/s)
    delta : float
        Two-photon detuning (rad/s)
    Gamma_e : float
        Intermediate state linewidth (rad/s)
    use_dark_state : bool
        Whether to apply dark-state suppression
        
    Returns
    -------
    float
        Scattering rate (rad/s)
    """
    gamma_base = intermediate_state_scattering_rate(Omega_1, Delta_e, Gamma_e)
    
    if use_dark_state:
        suppression = dark_state_suppression_factor(Delta_e, delta, 
                                                     Omega_1, Omega_2)
        return gamma_base * suppression
    else:
        return gamma_base


# =============================================================================
# RYDBERG STATE LEAKAGE
# =============================================================================

def leakage_rate_to_adjacent_states(
    Omega: float, 
    Delta_leak: float,
    pulse_shape: str = "square",
    tau: float = 1e-6,
    gamma_rydberg: float = 7143.0,  # Rydberg decay rate (Hz) for n=70
) -> float:
    """
    Calculate INCOHERENT leakage rate to nearby Rydberg states (n±1, nP, nD).
    
    Physics Background
    ------------------
    When driving |1⟩ → |nS⟩, we might accidentally off-resonantly excite:
    - |(n±1)S⟩ (adjacent principal quantum numbers, Δ ~ 20 GHz)
    - |nP⟩ or |nD⟩ (fine structure, Δ ~ 50 MHz)
    
    **Coherent vs Incoherent Leakage:**
    
    The OFF-RESONANT Rabi oscillation rate to a leakage state is:
        Ω_leak = Ω² / (2Δ_leak)  [coherent oscillation]
    
    This is NOT suitable for a Lindblad collapse operator because it's 
    a coherent process (population oscillates back and forth).
    
    The INCOHERENT leakage rate comes from population that:
    1. Coherently leaks to wrong state (fraction ~ (Ω/Δ_leak)²)
    2. Then decays/dephases there before returning to the target state
    
    The effective irreversible loss rate is:
        γ_leak_incoherent = (Ω/Δ_leak)² × γ_rydberg × S(pulse_shape)
    
    where:
    - (Ω/Δ_leak)² = time-averaged population in leakage state
    - γ_rydberg = decay rate of leakage state (~7 kHz for n~70)
    - S = spectral leakage factor from pulse bandwidth
    
    Example Calculation
    -------------------
    For Ω = 2π × 10 MHz, Δ_leak = 2π × 50 MHz (fine structure):
    - (Ω/Δ_leak)² = 0.04
    - γ_rydberg = 7 kHz
    - S ~ 0.3 (square pulse)
    - γ_leak = 0.04 × 7000 × 0.3 = 84 Hz
    
    This is much smaller than other noise sources (kHz range) but can
    accumulate over many gates.
    
    Parameters
    ----------
    Omega : float
        Rabi frequency (rad/s)
    Delta_leak : float
        Detuning to nearest leakage state (rad/s)
        - Fine structure (nS → nP): ~2π × 50 MHz
        - Adjacent n: ~2π × 20 GHz (usually negligible)
    pulse_shape : str
        "square", "gaussian", "cosine", "blackman", "drag"
        Smooth pulses have narrower spectra → less leakage
    tau : float
        Pulse duration (s)
    gamma_rydberg : float
        Rydberg state decay rate (Hz). Default 7143 Hz for n=70.
        Scales as n⁻³.
        
    Returns
    -------
    float
        Incoherent leakage rate (Hz)
        
    References
    ----------
    - Levine et al., PRL 123, 170503 (2019) - High-fidelity Rydberg gates
    - de Léséleuc et al., PRX 8, 021070 (2018) - Rydberg excitation dynamics
    """
    # Handle edge cases
    if np.abs(Delta_leak) < 1e-6 or np.abs(Omega) < 1e-6:
        return 0.0
    
    # Spectral leakage factor depends on pulse shape
    # This accounts for the pulse bandwidth overlap with leakage transitions
    x = Delta_leak * tau / (2 * np.pi)  # Dimensionless: Δ_leak × τ / 2π
    
    if np.abs(x) < 1e-10:
        # Very short pulse or near-resonant: maximum spectral overlap
        spectral_factor = 1.0
    elif pulse_shape == "square":
        # sinc² envelope for square pulse
        spectral_factor = (np.sin(np.pi * x) / (np.pi * x))**2
    elif pulse_shape == "gaussian":
        # Gaussian spectrum for Gaussian pulse (much narrower)
        spectral_factor = np.exp(-(Delta_leak * tau / 8)**2)
    elif pulse_shape == "cosine":
        # Raised cosine has narrower bandwidth than square
        if np.abs(np.abs(x) - 0.5) < 1e-10:
            spectral_factor = 0.25
        else:
            spectral_factor = (np.sin(np.pi * x) / (np.pi * x * (1 - x**2)))**2
    elif pulse_shape == "blackman":
        # Blackman window: very narrow spectrum, strong sidelobe suppression
        spectral_factor = np.exp(-3 * np.abs(x)) * 0.1
    elif pulse_shape == "drag":
        # DRAG pulses suppress leakage by design
        spectral_factor = np.exp(-(Delta_leak * tau / 8)**2) * 0.1
    else:
        # Default to square pulse spectrum
        spectral_factor = (np.sin(np.pi * x) / (np.pi * x + 1e-10))**2
    
    spectral_factor = np.clip(spectral_factor, 0, 1)
    
    # Off-resonant population fraction in leakage state
    # From perturbation theory: P_leak ~ (Ω/2Δ)² for far-detuned driving
    population_fraction = (Omega / Delta_leak)**2
    
    # Incoherent leakage rate = population × decay rate × spectral factor
    # This is the rate at which population irreversibly leaves via leakage states
    gamma_leak_incoherent = population_fraction * gamma_rydberg * spectral_factor
    
    return gamma_leak_incoherent


# =============================================================================
# mJ STATE MIXING
# =============================================================================

def mJ_mixing_rate(Omega_eff: float, polarization_purity: float,
                   Delta_zeeman: float) -> float:
    """
    Calculate mJ leakage rate from polarization impurity.
    
    **The problem:**
    
    When we drive |1⟩ → |r, mJ=+1/2⟩ with σ+ light, any σ- impurity
    can drive |1⟩ → |r, mJ=-1/2⟩ (the wrong Rydberg Zeeman sublevel).
    
    This is off-resonant by the Zeeman splitting Δ_Zeeman, but at high
    Rabi frequency and/or impure polarization, it can be significant.
    
    **The formula:**
    
    The σ- component drives with effective Rabi:
        Ω_σ- = ε_pol × Ω_σ+
    
    where ε_pol = √(1 - purity) is the polarization impurity.
    
    The off-resonant population transfer rate is:
        γ_mJ = ε_pol² × Ω² / Δ_Zeeman
    
    **Typical numbers:**
    - Polarization purity: 99% → ε_pol = 0.1 → ε²_pol = 0.01
    - Ω/2π = 5 MHz
    - B = 0.5 G → Δ_Zeeman/2π ≈ 0.7 MHz
    - γ_mJ = 0.01 × (5 MHz)² / 0.7 MHz ≈ 360 kHz (significant!)
    
    **Mitigation:**
    1. Better polarization (99.9% → 0.001 error, 100× better)
    2. Larger B-field (larger Δ_Zeeman)
    3. Lower Rabi frequency (but slower gate)
    
    Parameters
    ----------
    Omega_eff : float
        Two-photon Rabi frequency (rad/s)
    polarization_purity : float
        Polarization purity (0-1). 0.99 = 99% = 1% impurity
    Delta_zeeman : float
        Zeeman splitting between mJ states (rad/s)
        
    Returns
    -------
    float
        mJ mixing rate (rad/s)
    """
    epsilon_pol = 1.0 - polarization_purity
    
    if np.abs(Delta_zeeman) < 1e-10:
        # No Zeeman splitting - catastrophic leakage
        return epsilon_pol**2 * np.abs(Omega_eff)
    
    return epsilon_pol**2 * Omega_eff**2 / np.abs(Delta_zeeman)


def rydberg_zeeman_splitting(B_field: float, L: int = 0, J: float = 0.5) -> float:
    """
    Calculate Zeeman splitting for a Rydberg state.
    
    **The Zeeman effect:**
    
    In a magnetic field, atomic states with different mJ split:
        ΔE_Zeeman = g_J × μ_B × B × Δm_J
    
    For S₁/₂ states (L=0, J=1/2):
        g_J ≈ 2.002 (free electron g-factor)
        ΔE/h = 1.4 MHz/G × Δm_J × B(G)
    
    **Why does this matter?**
    
    The Zeeman splitting sets the energy scale for:
    1. mJ leakage (σ- polarization impurity)
    2. State-selective addressing
    3. Magnetic field noise sensitivity
    
    Parameters
    ----------
    B_field : float
        Magnetic field (Tesla)
    L : int
        Orbital angular momentum (0=S, 1=P, 2=D)
    J : float
        Total angular momentum (0.5 for S₁/₂)
        
    Returns
    -------
    float
        Zeeman splitting between adjacent mJ states (rad/s)
    """
    S = 0.5  # Electron spin
    
    # Landé g-factor
    if J == 0:
        g_J = 0
    else:
        g_J = 1 + (J*(J+1) + S*(S+1) - L*(L+1)) / (2*J*(J+1))
    
    # Add QED correction for S states
    if L == 0:
        g_J += 0.002
    
    return g_J * MU_B * B_field / HBAR


# =============================================================================
# COMBINED NOISE RATE CALCULATION
# =============================================================================

def compute_noise_rates(
    # Atomic properties
    species: str = "Rb87",
    n_rydberg: int = 70,
    rydberg_lifetime: float = 140e-6,
    rydberg_lifetime_0K: float = 280e-6,
    
    # Laser parameters
    Omega_eff: float = 2*np.pi*5e6,
    Omega_1: float = None,
    Omega_2: float = None,
    Delta_e: float = 2*np.pi*1e9,
    delta: float = 0,
    Gamma_e: float = 2*np.pi*6.065e6,
    laser_linewidth_hz: float = 1e3,
    
    # Trap/temperature
    temperature: float = 20e-6,
    omega_trap: float = 2*np.pi*100e3,
    
    # Blockade
    V: float = 2*np.pi*200e6,
    R: float = 3e-6,
    
    # Magnetic field
    B_field: float = 0.5e-4,
    B_noise_gauss: float = 0.001,
    qubit_type: str = "clock",
    
    # Polarization
    polarization_purity: float = 0.99,
    
    # Trap physics (for anti-trapping)
    U0: float = None,
    alpha_ratio: float = 300,
    mass: float = 1.44e-25,
    waist: float = 1e-6,
    gate_time: float = 1e-6,
    rydberg_fraction: float = 0.3,
    
    # Options
    pulse_shape: str = "square",
    use_dark_state: bool = True,
) -> NoiseRates:
    """
    Compute all noise rates for Rydberg gate simulation.
    
    This is the main entry point for noise modeling. It computes all
    relevant noise rates based on experimental parameters.
    
    Parameters
    ----------
    species : str
        Atomic species ("Rb87" or "Cs133")
    n_rydberg : int
        Principal quantum number
    rydberg_lifetime : float
        Rydberg state lifetime at operating temperature (s)
    rydberg_lifetime_0K : float
        Rydberg state lifetime at T=0 (s)
    Omega_eff : float
        Two-photon Rabi frequency (rad/s)
    Omega_1, Omega_2 : float
        Single-photon Rabi frequencies. If None, estimated from Omega_eff
    Delta_e : float
        Intermediate state detuning (rad/s)
    delta : float
        Two-photon detuning (rad/s)
    Gamma_e : float
        Intermediate state linewidth (rad/s)
    laser_linewidth_hz : float
        Effective laser linewidth (Hz)
    temperature : float
        Atom temperature (K)
    omega_trap : float
        Trap frequency (rad/s)
    V : float
        Blockade interaction (rad/s)
    R : float
        Atom spacing (m)
    B_field : float
        Magnetic field (T)
    B_noise_gauss : float
        B-field noise (Gauss)
    qubit_type : str
        "clock" or "stretched"
    polarization_purity : float
        Polarization purity (0-1)
    U0 : float
        Trap depth (J). If None, anti-trap loss not computed
    alpha_ratio : float
        |α_rydberg / α_ground|
    mass : float
        Atomic mass (kg)
    waist : float
        Trap beam waist (m)
    gate_time : float
        Gate duration (s)
    rydberg_fraction : float
        Average time fraction in Rydberg state
    pulse_shape : str
        Pulse shape for leakage calculation
    use_dark_state : bool
        Whether to apply dark-state suppression to scattering
        
    Returns
    -------
    NoiseRates
        Container with all computed noise rates
    """
    # Estimate single-photon Rabi frequencies if not provided
    if Omega_1 is None:
        Omega_1 = np.sqrt(2 * np.abs(Delta_e) * np.abs(Omega_eff))
    if Omega_2 is None:
        Omega_2 = Omega_1
    
    # 1. Rydberg spontaneous emission
    gamma_r = rydberg_decay_rate(rydberg_lifetime_0K)  # Use 0K lifetime
    
    # 2. BBR decay
    gamma_bbr = bbr_decay_rate(rydberg_lifetime_0K, rydberg_lifetime)
    
    # 3. Laser dephasing
    gamma_phi_laser = laser_dephasing_rate(laser_linewidth_hz)
    
    # 4. Thermal dephasing (from blockade fluctuation)
    # Position uncertainty
    sigma_r = np.sqrt(KB * temperature / (mass * omega_trap**2))
    delta_R = np.sqrt(2) * sigma_r  # Two atoms
    delta_V_over_V = 6 * delta_R / R  # Relative blockade fluctuation
    
    # Thermal dephasing depends on blockade regime
    V_over_Omega = np.abs(V) / np.abs(Omega_eff)
    if V_over_Omega < 3:
        # Weak blockade
        infidelity = delta_V_over_V**2 * V_over_Omega**2
    elif V_over_Omega > 10:
        # Strong blockade - suppressed
        infidelity = delta_V_over_V**2 * (Omega_eff / V)**2
    else:
        # Intermediate
        infidelity = delta_V_over_V**2  # Conservative
    gamma_phi_thermal = infidelity * np.abs(Omega_eff) / (2 * np.pi)
    
    # 5. Zeeman dephasing
    gamma_phi_zeeman = zeeman_dephasing_rate(B_noise_gauss, qubit_type)
    
    # 6. Anti-trap atom loss
    gamma_loss_antitrap = 0.0
    if U0 is not None and U0 > 0:
        from .trap_physics import effective_loss_rate
        gamma_loss_antitrap = effective_loss_rate(
            gate_time, U0, alpha_ratio, mass, waist, temperature, rydberg_fraction
        )
    
    # 7. Background gas loss (typically negligible)
    gamma_loss_bg = 2 * np.pi * 0.1  # ~0.1 Hz typical at 10^-11 Torr
    
    # 8. Intermediate state scattering
    gamma_scatter = enhanced_scattering_rate(
        Omega_1, Omega_2, Delta_e, delta, Gamma_e, use_dark_state
    )
    
    # 9. Rydberg leakage
    Delta_leak_fs = 2 * np.pi * 50e6  # Fine structure ~50 MHz
    gamma_leakage = leakage_rate_to_adjacent_states(
        Omega_eff, Delta_leak_fs, pulse_shape, gate_time
    )
    
    # 10. mJ mixing
    Delta_zeeman = rydberg_zeeman_splitting(B_field, L=0, J=0.5)
    gamma_mJ = mJ_mixing_rate(Omega_eff, polarization_purity, Delta_zeeman)
    
    return NoiseRates(
        gamma_r=gamma_r,
        gamma_bbr=gamma_bbr,
        gamma_phi_laser=gamma_phi_laser,
        gamma_phi_thermal=gamma_phi_thermal,
        gamma_phi_zeeman=gamma_phi_zeeman,
        gamma_loss_antitrap=gamma_loss_antitrap,
        gamma_loss_bg=gamma_loss_bg,
        gamma_scatter=gamma_scatter,
        gamma_leakage=gamma_leakage,
        gamma_mJ=gamma_mJ,
    )


# =============================================================================
# PART 2: LINDBLAD COLLAPSE OPERATORS
# =============================================================================
# 
# This section builds the actual quantum operators used in mesolve().
# 
# DESIGN:
# -------
# - Individual rate functions (Part 1) compute γ values from physics
# - These builders convert rates → QuTiP Qobj collapse operators
# - `build_all_noise_operators()` is the UNIFIED entry point
#
# USAGE FLOW:
# -----------
# 1. Compute trap-dependent rates via `trap_physics.compute_trap_dependent_noise()`
# 2. Pass rates to `build_all_noise_operators()` to get c_ops list
# 3. Feed c_ops to `qutip.mesolve()` for Lindblad dynamics
#
# For trap-dependent noise (trap depth, position uncertainty, magic wavelength),
# use `from .trap_physics import compute_trap_dependent_noise` which combines
# tweezer parameters with the rate functions in this module.
# =============================================================================

def op_two_atom(op1, op2):
    """
    Construct two-atom operator from single-atom operators.
    
    For a two-atom system, single-atom operators act on one atom while
    the identity acts on the other. This helper creates the tensor product.
    
    Parameters
    ----------
    op1 : Qobj
        Operator acting on atom 1
    op2 : Qobj
        Operator acting on atom 2
        
    Returns
    -------
    Qobj
        Tensor product op1 ⊗ op2
    """
    from qutip import tensor
    return tensor(op1, op2)


def build_decay_operators(gamma_optical: float, hs, gamma_bbr: float = 0,
                          branching_1: float = 0.5, leakage_rate: float = 0):
    """
    Build decay collapse operators for Lindblad dynamics.
    
    Models:
    1. Optical spontaneous emission: |r⟩ → |1⟩ or |0⟩
    2. BBR-induced decay: |r⟩ → loss (modeled as decay to |0⟩)
    3. mJ leakage: |r+⟩ ↔ |r-⟩ (4-level only)
    
    Parameters
    ----------
    gamma_optical : float
        Optical decay rate (1/τ_optical) in Hz
    hs : HilbertSpace
        Hilbert space object (HS3 or HS4)
    gamma_bbr : float
        Blackbody decay rate (Hz)
    branching_1 : float
        Fraction decaying to |1⟩ vs |0⟩
    leakage_rate : float
        mJ mixing rate (4-level only)
        
    Returns
    -------
    List[Qobj]
        List of collapse operators for mesolve
    """
    from qutip import tensor
    c_ops = []
    I = hs.identity
    
    if hs.dim == 3:
        # --- 3-level: single Rydberg state ---
        if gamma_optical > 0:
            sigma_r1 = hs.transitions['r->1']  # |1⟩⟨r|
            sigma_r0 = hs.transitions['r->0']  # |0⟩⟨r|
            
            L_r_to_1_atom1 = np.sqrt(gamma_optical * branching_1) * op_two_atom(sigma_r1, I)
            L_r_to_1_atom2 = np.sqrt(gamma_optical * branching_1) * op_two_atom(I, sigma_r1)
            c_ops.extend([L_r_to_1_atom1, L_r_to_1_atom2])
            
            # |r⟩ → |0⟩
            L_r_to_0_atom1 = np.sqrt(gamma_optical * (1 - branching_1)) * op_two_atom(sigma_r0, I)
            L_r_to_0_atom2 = np.sqrt(gamma_optical * (1 - branching_1)) * op_two_atom(I, sigma_r0)
            c_ops.extend([L_r_to_0_atom1, L_r_to_0_atom2])
        
        # BBR decay
        if gamma_bbr > 0:
            sigma_r0 = hs.transitions['r->0']
            L_bbr_1 = np.sqrt(gamma_bbr) * op_two_atom(sigma_r0, I)
            L_bbr_2 = np.sqrt(gamma_bbr) * op_two_atom(I, sigma_r0)
            c_ops.extend([L_bbr_1, L_bbr_2])
            
    elif hs.dim == 4:
        # --- 4-level: separate mJ states ---
        if gamma_optical > 0:
            # Decay from |r+⟩
            sigma_rp1 = hs.transitions['r+->1']  # |1⟩⟨r+|
            sigma_rp0 = hs.transitions['r+->0']  # |0⟩⟨r+|
            
            L_rp_to_1_atom1 = np.sqrt(gamma_optical * branching_1) * op_two_atom(sigma_rp1, I)
            L_rp_to_1_atom2 = np.sqrt(gamma_optical * branching_1) * op_two_atom(I, sigma_rp1)
            L_rp_to_0_atom1 = np.sqrt(gamma_optical * (1 - branching_1)) * op_two_atom(sigma_rp0, I)
            L_rp_to_0_atom2 = np.sqrt(gamma_optical * (1 - branching_1)) * op_two_atom(I, sigma_rp0)
            c_ops.extend([L_rp_to_1_atom1, L_rp_to_1_atom2, L_rp_to_0_atom1, L_rp_to_0_atom2])
            
            # Decay from |r-⟩
            sigma_rm1 = hs.transitions['r-->1']  # |1⟩⟨r-|
            sigma_rm0 = hs.transitions['r-->0']  # |0⟩⟨r-|
            
            L_rm_to_1_atom1 = np.sqrt(gamma_optical * branching_1) * op_two_atom(sigma_rm1, I)
            L_rm_to_1_atom2 = np.sqrt(gamma_optical * branching_1) * op_two_atom(I, sigma_rm1)
            L_rm_to_0_atom1 = np.sqrt(gamma_optical * (1 - branching_1)) * op_two_atom(sigma_rm0, I)
            L_rm_to_0_atom2 = np.sqrt(gamma_optical * (1 - branching_1)) * op_two_atom(I, sigma_rm0)
            c_ops.extend([L_rm_to_1_atom1, L_rm_to_1_atom2, L_rm_to_0_atom1, L_rm_to_0_atom2])
        
        # BBR decay
        if gamma_bbr > 0:
            sigma_rp0 = hs.transitions['r+->0']
            sigma_rm0 = hs.transitions['r-->0']
            L_bbr_rp1 = np.sqrt(gamma_bbr) * op_two_atom(sigma_rp0, I)
            L_bbr_rp2 = np.sqrt(gamma_bbr) * op_two_atom(I, sigma_rp0)
            L_bbr_rm1 = np.sqrt(gamma_bbr) * op_two_atom(sigma_rm0, I)
            L_bbr_rm2 = np.sqrt(gamma_bbr) * op_two_atom(I, sigma_rm0)
            c_ops.extend([L_bbr_rp1, L_bbr_rp2, L_bbr_rm1, L_bbr_rm2])
        
        # mJ leakage: |r+⟩ ↔ |r-⟩
        if leakage_rate > 0:
            sigma_rprm = hs.transitions['r+->r-']  # |r-⟩⟨r+|
            sigma_rmrp = hs.transitions['r-->r+']  # |r+⟩⟨r-|
            
            L_p_to_m_1 = np.sqrt(leakage_rate) * op_two_atom(sigma_rprm, I)
            L_p_to_m_2 = np.sqrt(leakage_rate) * op_two_atom(I, sigma_rprm)
            L_m_to_p_1 = np.sqrt(leakage_rate) * op_two_atom(sigma_rmrp, I)
            L_m_to_p_2 = np.sqrt(leakage_rate) * op_two_atom(I, sigma_rmrp)
            c_ops.extend([L_p_to_m_1, L_p_to_m_2, L_m_to_p_1, L_m_to_p_2])
    
    return c_ops


def build_dephasing_operators(gamma_phi: float, hs, gamma_phi_minus: float = None):
    """
    Build pure dephasing operators for Rydberg states.
    
    Dephasing arises from:
    - Laser phase noise
    - Doppler shifts
    - Differential AC Stark shifts
    - Magnetic field fluctuations
    
    Parameters
    ----------
    gamma_phi : float
        Dephasing rate for primary Rydberg state (Hz)
    hs : HilbertSpace
        Hilbert space object
    gamma_phi_minus : float, optional
        For 4-level: dephasing rate for |r-⟩. If None, uses same as gamma_phi.
        
    Returns
    -------
    List[Qobj]
        List of dephasing collapse operators
    """
    if gamma_phi <= 0:
        return []
    
    if gamma_phi_minus is None:
        gamma_phi_minus = gamma_phi
    
    c_ops = []
    I = hs.identity
    
    if hs.dim == 3:
        Pr = hs.projectors['r']
        Pr1 = op_two_atom(Pr, I)
        Pr2 = op_two_atom(I, Pr)
        c_ops.append(np.sqrt(gamma_phi) * Pr1)
        c_ops.append(np.sqrt(gamma_phi) * Pr2)
        
    elif hs.dim == 4:
        Prp = hs.projectors['r+']
        Prm = hs.projectors['r-']
        
        # Dephasing on |r+⟩
        Prp1 = op_two_atom(Prp, I)
        Prp2 = op_two_atom(I, Prp)
        c_ops.append(np.sqrt(gamma_phi) * Prp1)
        c_ops.append(np.sqrt(gamma_phi) * Prp2)
        
        # Dephasing on |r-⟩
        Prm1 = op_two_atom(Prm, I)
        Prm2 = op_two_atom(I, Prm)
        c_ops.append(np.sqrt(gamma_phi_minus) * Prm1)
        c_ops.append(np.sqrt(gamma_phi_minus) * Prm2)
    
    return c_ops


def build_loss_operators(gamma_loss: float, hs, loss_source: str = 'rydberg'):
    """
    Build loss collapse operators (anti-trap, background, leakage).
    
    Loss transfers population from a state to |0⟩ (outside computational basis).
    This models atom loss from the trap or leakage to non-computational states.
    
    Parameters
    ----------
    gamma_loss : float
        Loss rate (Hz)
    hs : HilbertSpace
        Hilbert space object
    loss_source : str
        Which state loses population: 'rydberg' (|r⟩) or 'qubit' (|1⟩)
        
    Returns
    -------
    List[Qobj]
        Loss collapse operators
    """
    if gamma_loss <= 0:
        return []
    
    c_ops = []
    I = hs.identity
    
    if hs.dim == 3:
        if loss_source == 'rydberg':
            # |r⟩ → |0⟩ (atom lost)
            sigma_r0 = hs.transitions['r->0']  # |0⟩⟨r|
            c_ops.append(np.sqrt(gamma_loss) * op_two_atom(sigma_r0, I))
            c_ops.append(np.sqrt(gamma_loss) * op_two_atom(I, sigma_r0))
        elif loss_source == 'qubit':
            # |1⟩ → |0⟩ (qubit decay, not physical loss)
            sigma_10 = hs.transitions.get('1->0', hs.basis['0'] * hs.basis['1'].dag())
            c_ops.append(np.sqrt(gamma_loss) * op_two_atom(sigma_10, I))
            c_ops.append(np.sqrt(gamma_loss) * op_two_atom(I, sigma_10))
            
    elif hs.dim == 4:
        if loss_source == 'rydberg':
            # Loss from both mJ states
            sigma_rp0 = hs.transitions['r+->0']
            sigma_rm0 = hs.transitions['r-->0']
            c_ops.append(np.sqrt(gamma_loss) * op_two_atom(sigma_rp0, I))
            c_ops.append(np.sqrt(gamma_loss) * op_two_atom(I, sigma_rp0))
            c_ops.append(np.sqrt(gamma_loss) * op_two_atom(sigma_rm0, I))
            c_ops.append(np.sqrt(gamma_loss) * op_two_atom(I, sigma_rm0))
        elif loss_source == 'qubit':
            sigma_10 = hs.transitions.get('1->0', hs.basis['0'] * hs.basis['1'].dag())
            c_ops.append(np.sqrt(gamma_loss) * op_two_atom(sigma_10, I))
            c_ops.append(np.sqrt(gamma_loss) * op_two_atom(I, sigma_10))
    
    return c_ops


def build_scatter_operators(gamma_scatter: float, hs):
    """
    Build intermediate state scattering operators.
    
    Models off-resonant scattering from the intermediate |e⟩ state during
    the two-photon Rydberg excitation. In the effective 3-level model,
    this appears as dephasing of |1⟩.
    
    Parameters
    ----------
    gamma_scatter : float
        Scattering rate (Hz)
    hs : HilbertSpace
        Hilbert space object
        
    Returns
    -------
    List[Qobj]
        Scatter collapse operators
    """
    if gamma_scatter <= 0:
        return []
    
    c_ops = []
    I = hs.identity
    
    # Scattering appears as dephasing on |1⟩
    P1 = hs.projectors['1']
    c_ops.append(np.sqrt(gamma_scatter) * op_two_atom(P1, I))
    c_ops.append(np.sqrt(gamma_scatter) * op_two_atom(I, P1))
    
    return c_ops


def build_all_noise_operators(
    params: dict = None,
    hs = None,
    dim: int = 3,
    # Comprehensive noise parameters (alternative to params dict)
    gamma_r: float = None,
    gamma_bbr: float = None,
    gamma_phi_laser: float = None,
    gamma_phi_thermal: float = None,
    gamma_phi_zeeman: float = None,
    gamma_loss_antitrap: float = None,
    gamma_loss_background: float = None,
    gamma_scatter_intermediate: float = None,
    gamma_leakage: float = None,
    branching_1: float = 0.5,
    mJ_leakage_rate: float = 0,
):
    """
    Build COMPLETE collapse operator set from comprehensive noise parameters.
    
    This is the UNIFIED noise model that consolidates ALL noise sources from
    the Bluvstein thesis error budget. It supports both dictionary-based and
    explicit parameter interfaces, and works with both 3-level and 4-level
    Hilbert spaces.
    
    Noise Sources (Bluvstein Error Budget):
    --------------------------------------
    1. Rydberg decay (γ_r): Spontaneous emission |r⟩ → |1⟩
    2. BBR decay (γ_bbr): Blackbody-induced redistribution
    3. Laser dephasing (γ_φ,laser): Phase noise from laser linewidth
    4. Thermal dephasing (γ_φ,thermal): From atomic motion
    5. Zeeman dephasing (γ_φ,Zeeman): Magnetic field fluctuations
    6. Anti-trap loss (γ_loss,antitrap): From repulsive Rydberg potential
    7. Background loss (γ_loss,background): Gas collisions
    8. Intermediate scattering (γ_scatter): Off-resonant |e⟩ scattering
    9. Spectral leakage (γ_leakage): Population loss to |n±1⟩
    10. mJ leakage (mJ_leakage_rate): |r+⟩ ↔ |r-⟩ mixing (4-level)
    
    Parameters
    ----------
    params : dict, optional
        Dictionary with noise rates. If provided, overrides explicit parameters.
    hs : HilbertSpace, optional
        Hilbert space object. If None, built from dim.
    dim : int
        Dimension if hs not provided (3 or 4)
    gamma_r : float
        Rydberg decay rate (Hz)
    gamma_bbr : float
        Blackbody decay rate (Hz)
    gamma_phi_laser : float
        Laser dephasing rate (Hz)
    gamma_phi_thermal : float
        Thermal/motional dephasing rate (Hz)
    gamma_phi_zeeman : float
        Zeeman dephasing rate (Hz)
    gamma_loss_antitrap : float
        Anti-trap loss rate (Hz)
    gamma_loss_background : float
        Background loss rate (Hz)
    gamma_scatter_intermediate : float
        Intermediate state scattering rate (Hz)
    gamma_leakage : float
        Spectral leakage rate to |n±1⟩ (Hz)
    branching_1 : float
        Fraction of decay going to |1⟩ vs |0⟩
    mJ_leakage_rate : float
        mJ mixing rate (4-level only)
        
    Returns
    -------
    c_ops : List[Qobj]
        Complete list of collapse operators
    noise_breakdown : dict
        Detailed breakdown of all noise contributions
    """
    # Import HilbertSpace if needed
    if hs is None:
        from .hamiltonians import build_hilbert_space
        hs = build_hilbert_space(dim)
    
    # Parse parameters from dict or explicit values
    if params is not None:
        # Extract from dictionary with legacy support
        gamma_r = params.get('gamma_r', params.get('gamma_optical', 0))
        if gamma_r == 0 and 'T1' in params and params['T1'] > 0:
            gamma_r = 1.0 / params['T1']
        
        gamma_bbr = params.get('gamma_bbr', 0)
        
        # Dephasing: support multiple keys
        gamma_phi_laser = params.get('gamma_phi_laser', 0)
        gamma_phi_thermal = params.get('gamma_phi_thermal', 0)
        gamma_phi_zeeman = params.get('gamma_phi_zeeman', 0)
        
        # Legacy dephasing support
        if gamma_phi_laser == 0 and gamma_phi_thermal == 0 and gamma_phi_zeeman == 0:
            gamma_deph = params.get('gamma_dephasing', 0)
            if gamma_deph == 0 and 'T2_star' in params and params['T2_star'] > 0:
                gamma_deph = 1.0 / params['T2_star']
            gamma_phi_laser = gamma_deph
        
        # Loss rates
        gamma_loss_antitrap = params.get('gamma_loss_antitrap', 0)
        gamma_loss_background = params.get('gamma_loss_background', 0)
        gamma_scatter_intermediate = params.get('gamma_scatter_intermediate', 0)
        gamma_leakage = params.get('gamma_leakage', 0)
        
        # Other params
        branching_1 = params.get('branching_1', 0.5)
        mJ_leakage_rate = params.get('mJ_leakage_rate', 0)
    
    # Convert None to 0
    gamma_r = gamma_r or 0
    gamma_bbr = gamma_bbr or 0
    gamma_phi_laser = gamma_phi_laser or 0
    gamma_phi_thermal = gamma_phi_thermal or 0
    gamma_phi_zeeman = gamma_phi_zeeman or 0
    gamma_loss_antitrap = gamma_loss_antitrap or 0
    gamma_loss_background = gamma_loss_background or 0
    gamma_scatter_intermediate = gamma_scatter_intermediate or 0
    gamma_leakage = gamma_leakage or 0
    
    # Build all collapse operators
    c_ops = []
    
    # 1. Decay operators (γ_r + γ_bbr)
    c_ops.extend(build_decay_operators(gamma_r, hs, gamma_bbr, branching_1, mJ_leakage_rate))
    
    # 2. Total dephasing (laser + thermal + Zeeman)
    gamma_phi_total = gamma_phi_laser + gamma_phi_thermal + gamma_phi_zeeman
    c_ops.extend(build_dephasing_operators(gamma_phi_total, hs))
    
    # 3. Anti-trap loss (|r⟩ → loss)
    c_ops.extend(build_loss_operators(gamma_loss_antitrap, hs, 'rydberg'))
    
    # 4. Background loss
    c_ops.extend(build_loss_operators(gamma_loss_background, hs, 'rydberg'))
    
    # 5. Intermediate state scattering
    c_ops.extend(build_scatter_operators(gamma_scatter_intermediate, hs))
    
    # 6. Spectral leakage to |n±1⟩
    c_ops.extend(build_loss_operators(gamma_leakage, hs, 'rydberg'))
    
    # Create detailed breakdown
    noise_breakdown = {
        # Individual rates
        'gamma_r': gamma_r,
        'gamma_bbr': gamma_bbr,
        'gamma_phi_laser': gamma_phi_laser,
        'gamma_phi_thermal': gamma_phi_thermal,
        'gamma_phi_zeeman': gamma_phi_zeeman,
        'gamma_loss_antitrap': gamma_loss_antitrap,
        'gamma_loss_background': gamma_loss_background,
        'gamma_scatter_intermediate': gamma_scatter_intermediate,
        'gamma_leakage': gamma_leakage,
        'mJ_leakage_rate': mJ_leakage_rate,
        'branching_1': branching_1,
        
        # Summary totals
        'gamma_phi_total': gamma_phi_total,
        'total_decay_rate': gamma_r + gamma_bbr,
        'total_dephasing_rate': gamma_phi_total,
        'total_loss_rate': gamma_loss_antitrap + gamma_loss_background + gamma_leakage,
        
        # Metadata
        'dim': hs.dim,
        'n_collapse_ops': len(c_ops),
    }
    
    return c_ops, noise_breakdown


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Data classes
    "NoiseRates",
    
    # Individual noise rates (Part 1)
    "rydberg_decay_rate",
    "bbr_decay_rate",
    "laser_dephasing_rate",
    "zeeman_dephasing_rate",
    "intermediate_state_scattering_rate",
    "dark_state_suppression_factor",
    "enhanced_scattering_rate",
    "leakage_rate_to_adjacent_states",
    "mJ_mixing_rate",
    "rydberg_zeeman_splitting",
    
    # Combined rate calculation (Part 1)
    "compute_noise_rates",
    
    # NOTE: For trap-dependent noise (trap depth, position uncertainty, 
    # magic wavelength), use:
    #   from .trap_physics import compute_trap_dependent_noise
    
    # Collapse operator builders (Part 2)
    "op_two_atom",
    "build_decay_operators",
    "build_dephasing_operators",
    "build_loss_operators",
    "build_scatter_operators",
    "build_all_noise_operators",
]
