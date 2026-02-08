"""
Optical Tweezer Trap Physics for Rydberg Gate Simulations
==========================================================

This module implements the physics of optical tweezers - the tiny laser traps
that hold individual atoms in place during quantum gate operations.

WHAT IS AN OPTICAL TWEEZER?
---------------------------

An optical tweezer is a tightly focused laser beam that traps neutral atoms:

    1. A high-NA lens focuses a laser to a ~1 μm spot
    2. The focused light creates an intensity gradient
    3. The atom's electron cloud gets polarized by the electric field
    4. This induced dipole interacts with the field gradient
    5. Result: atoms are pulled toward the intensity maximum!

**The math**: The potential energy is U = -α × |E|² / 2, where α is the
atomic polarizability. For positive α (red-detuned trap), U is negative
at high intensity → attractive potential → trapped atom.

WHY DOES TRAP PHYSICS MATTER FOR GATES?
---------------------------------------

1. **Position uncertainty causes interaction fluctuations**
   - Atoms aren't perfectly still - they jiggle with thermal energy
   - The blockade interaction V = C₆/R⁶ depends sensitively on distance
   - A 3% position error → 18% interaction error (since 6 × 3% = 18%)
   - This creates random phase errors in the CZ gate!

2. **Anti-trapping of Rydberg states**
   - Ground state atoms: α > 0 → attracted to trap center ✓
   - Rydberg state atoms: α < 0 → REPELLED from trap center ✗
   - During the gate, when atoms go to |r⟩, they get pushed away!
   - If they move too far, they escape → atom loss → gate failure

3. **Trap depth sets motional state**
   - Deeper trap → higher trap frequency → smaller position uncertainty
   - But deeper trap also means stronger anti-trapping!
   - There's a sweet spot that balances these effects

THE KEY CHAIN OF CALCULATIONS
-----------------------------

    Laser power → Trap depth → Trap frequency → Position uncertainty → Noise
    
    1. trap_depth(P, w, α) gives U₀ [energy]
    2. trap_frequencies(U₀, m, w, λ) gives ω_r, ω_z [angular frequency]  
    3. position_uncertainty(T, m, ω) gives σ [length]
    4. blockade_fluctuation(R, σ) gives δV/V [dimensionless]

TYPICAL VALUES
--------------

For Rb87 with P = 10 mW, w = 1 μm, λ = 852 nm:
- Trap depth: U₀/k_B ≈ 0.5-2 mK
- Radial frequency: ω_r/2π ≈ 100-200 kHz
- Axial frequency: ω_z/2π ≈ 10-30 kHz
- Position uncertainty at 20 μK: σ ≈ 50-100 nm

For comparison:
- Atom spacing: R ≈ 3-5 μm
- So σ/R ≈ 1-3%, giving δV/V ≈ 6-18%

References
----------
[1] Grimm et al., Adv. At. Mol. Opt. Phys. 42, 95 (2000) - Optical dipole traps
[2] Schlosser et al., Nature 411, 1024 (2001) - Single atom tweezers
[3] Kaufman et al., PRA 86, 043409 (2012) - Tweezer heating rates
[4] de Léséleuc et al., PRA 97, 053803 (2018) - Gate fidelity limits
[5] Levine et al., PRL 123, 170503 (2019) - High-fidelity Rydberg gates
"""

import numpy as np
from typing import Tuple, Optional

from .constants import HBAR, EPS0, C, KB, A0


# =============================================================================
# MAGIC WAVELENGTH AND POLARIZABILITY FUNCTIONS
# =============================================================================

def get_polarizability_at_wavelength(species: str, state: str, wavelength_nm: float,
                                      n_rydberg: int = 70, L_rydberg: int = 0,
                                      F: int = None) -> float:
    """
    Calculate dynamic polarizability at a given wavelength.
    
    Uses a two-level model with corrections for the nearest resonances.
    For precise values, use full multi-level calculations or literature tables.
    
    Parameters
    ----------
    species : str
        Atomic species ("Rb87" or "Cs133")
    state : str
        State type: "ground", "rydberg", or specific like "5S" or "70S"
    wavelength_nm : float
        Trap wavelength (nm)
    n_rydberg : int
        Principal quantum number for Rydberg states
    L_rydberg : int
        Orbital angular momentum (0=S, 1=P, 2=D)
    F : int, optional
        Hyperfine F quantum number for ground state
        
    Returns
    -------
    float
        Polarizability in SI units (C²·m²/J = F·m²)
        
    Notes
    -----
    The polarizability is computed using:
    
    α(ω) = α_static × (ω₀² - ω_magic²) / (ω₀² - ω²)
    
    where ω₀ is the nearest resonance and ω_magic is the magic wavelength.
    
    This is a simplified model. For precision work, use tables from:
    - Arora et al., PRA 76, 052509 (2007)
    - Safronova & Safronova, PRA 83, 052508 (2011)
    
    References
    ----------
    [1] Zhang et al., PRA 84, 043408 (2011) - Rydberg polarizabilities
    [2] Topcu & Derevianko, PRA 88, 053406 (2013) - Wavelength dependence
    """
    from .atom_database import ATOM_DB
    
    atom = ATOM_DB[species]
    wavelength_m = wavelength_nm * 1e-9
    omega = 2 * np.pi * C / wavelength_m  # Trap angular frequency
    
    if state.lower() in ["ground", "5s", "6s"]:
        # Ground state polarizability
        alpha_static = atom["alpha_ground"]
        
        # Nearest resonance (D2 line)
        if species == "Rb87":
            omega_D2 = 2 * np.pi * 384.230e12  # D2 line
            omega_D1 = 2 * np.pi * 377.107e12  # D1 line
        else:  # Cs133
            omega_D2 = 2 * np.pi * 351.726e12
            omega_D1 = 2 * np.pi * 335.116e12
        
        # Simple two-resonance model
        # α(ω) ≈ α_static × (1 + ω²/(ω_D² - ω²)) correction
        if omega < omega_D1:  # Red-detuned from both lines
            # Both D1 and D2 contribute positive polarizability
            correction = 1.0 + 0.3 * omega**2 / (omega_D1**2 - omega**2)
            alpha = alpha_static * correction
        else:
            # Between lines or blue-detuned - more complex
            alpha = alpha_static  # Use static value as approximation
        
        # Hyperfine correction if F specified
        if F is not None and "alpha_hyperfine" in atom:
            if F in atom["alpha_hyperfine"]:
                alpha = atom["alpha_hyperfine"][F]
        
        return alpha
    
    elif state.lower() in ["rydberg"] or state[0].isdigit():
        # Rydberg state polarizability with ponderomotive (free electron) correction
        # Extract n from state string if provided (e.g., "70S")
        if state[0].isdigit():
            n_rydberg = int(''.join(filter(str.isdigit, state)))
        
        # Get reference polarizability and scale with n
        alpha_ref = atom["alpha_rydberg_ref"]
        n_ref = atom["n_ref"]
        
        # Get quantum defect for scaling
        L_label = {0: "S", 1: "P", 2: "D", 3: "F"}.get(L_rydberg, "S")
        delta_qd = atom["quantum_defects"].get(L_label, 3.0)
        
        n_star = n_rydberg - delta_qd
        n_star_ref = n_ref - delta_qd
        
        # Scale: α_rydberg ∝ n*^7
        scaling_exp = atom["scaling_exponents"]["polarizability"]
        alpha_rydberg_static = alpha_ref * (n_star / n_star_ref)**scaling_exp
        
        # Wavelength dependence for Rydberg states using ponderomotive model
        # For highly excited states, the electron responds as nearly free:
        # α_ponderomotive = -e²/(m_e × ω²) (negative = anti-trapping)
        # 
        # The reference value is measured at a specific wavelength (typically ~1 μm)
        # We scale with ω² to get wavelength dependence:
        # α(λ) = α_ref × (λ/λ_ref)²
        #
        # Reference: Topcu & Derevianko, PRA 88, 053406 (2013)
        # Reference: Goldschmidt et al., PRA 91, 032518 (2015)
        
        lambda_ref = 1064e-9  # Reference wavelength (1064 nm, common trap)
        # Ponderomotive scaling: α ∝ λ² ∝ 1/ω²
        wavelength_factor = (wavelength_m / lambda_ref)**2
        
        alpha_rydberg = alpha_rydberg_static * wavelength_factor
        
        return alpha_rydberg
    
    else:
        raise ValueError(f"Unknown state: {state}. Use 'ground', 'rydberg', or 'nS/nP'")


def magic_trap_enhancement(species: str, wavelength_nm: float, n_rydberg: int = 70) -> float:
    """
    Calculate the "magic enhancement factor" for a trap wavelength.
    
    This factor indicates how close to magic the wavelength is.
    Enhancement = 1 at magic wavelength, <1 for non-magic.
    
    **What is a magic wavelength?**
    
    At a magic wavelength, the ground and excited (Rydberg) states have
    the same polarizability: α_ground = α_rydberg. This means:
    - No differential light shift between states
    - No dephasing from trap intensity fluctuations
    - Much better coherence during gate operations
    
    Parameters
    ----------
    species : str
        Atomic species ("Rb87" or "Cs133")
    wavelength_nm : float
        Trap wavelength (nm)
    n_rydberg : int
        Principal quantum number
        
    Returns
    -------
    float
        Enhancement factor (0-1, higher is better for coherence)
        
    Notes
    -----
    For typical (non-magic) traps at 1064 nm:
    - α_rydberg is negative and large → enhancement ≈ 0.01
    
    For magic wavelengths (~880 nm for Rb):
    - α_rydberg ≈ α_ground → enhancement ≈ 1.0
    """
    alpha_ground = get_polarizability_at_wavelength(species, "ground", wavelength_nm)
    alpha_rydberg = get_polarizability_at_wavelength(species, "rydberg", wavelength_nm,
                                                      n_rydberg=n_rydberg)
    
    # Enhancement factor: 1 when polarizabilities match
    ratio = alpha_rydberg / alpha_ground if np.abs(alpha_ground) > 1e-50 else 0
    
    # For typical traps, α_rydberg is negative and large
    # Enhancement = 1 / (1 + |1 - ratio|) gives 1 at magic, <1 otherwise
    enhancement = 1.0 / (1.0 + np.abs(1.0 - ratio))
    
    return enhancement


# =============================================================================
# TWEEZER SPACING FROM OPTICS
# =============================================================================

def tweezer_spacing(wavelength: float, NA: float, factor: float = 1.0) -> float:
    """
    Calculate the minimum spacing between optical tweezer traps.
    
    **What sets the spacing?**
    
    Optical tweezers are focused laser beams, and diffraction limits how
    close you can pack them. The minimum resolvable spot size (Rayleigh 
    criterion) is:
    
        d_min = λ / (2 × NA)
    
    where NA is the numerical aperture of the focusing lens.
    
    In practice, tweezers can be placed at multiples of this spacing
    using acousto-optic deflectors or spatial light modulators.
    
    **Why does spacing matter?**
    
    Closer atoms → stronger blockade (V ∝ R⁻⁶)
    But too close → optical crosstalk between tweezers
    
    Typical choice: R = 1.5-3 × d_min ≈ 2-5 μm
    
    Parameters
    ----------
    wavelength : float
        Trapping laser wavelength (m). Typical: 850-1064 nm
    NA : float
        Numerical aperture of focusing objective. Typical: 0.4-0.7
    factor : float
        Multiplicative factor for larger spacings (default 1.0).
        Use factor=2 to double the minimum spacing, etc.
        
    Returns
    -------
    float
        Tweezer spacing (m)
        
    Example
    -------
    >>> spacing = tweezer_spacing(852e-9, 0.5, factor=1.5)
    >>> spacing * 1e6  # Convert to μm
    1.28  # μm
    
    **Experimental context:**
    - Bluvstein (Harvard): λ = 850 nm, NA ≈ 0.5, R ≈ 2-4 μm
    - Lukin group: switchable spacing via AOD arrays
    """
    d_min = wavelength / (2 * NA)
    return factor * d_min


def diffraction_limited_spot(wavelength: float, NA: float) -> float:
    """
    Calculate the diffraction-limited spot size (1/e² intensity radius).
    
    The focused beam waist for a Gaussian beam with aperture-filling is:
        w₀ ≈ 0.64 × λ / NA  (for uniform illumination)
        w₀ ≈ 0.82 × λ / NA  (for Gaussian input beam)
    
    We use the Gaussian input approximation, which is more common.
    
    Parameters
    ----------
    wavelength : float
        Laser wavelength (m)
    NA : float
        Numerical aperture
        
    Returns
    -------
    float
        Beam waist 1/e² radius (m)
    """
    return 0.82 * wavelength / NA


# =============================================================================
# TRAP DEPTH
# =============================================================================

def trap_depth(power: float, waist: float, alpha: float) -> float:
    """
    Calculate optical trap depth from laser power and beam waist.
    
    **What is trap depth?**
    
    The trap depth U₀ is the energy difference between the trap center
    (intensity maximum) and infinitely far away. Think of it as "how
    deep is the bowl that holds the atom."
    
    An atom with kinetic energy less than U₀ cannot escape the trap.
    Typical trap depths: U₀/k_B = 0.5-2 mK (milliKelvin scale!)
    
    **The physics:**
    
    A Gaussian laser beam has intensity profile:
        I(r, z) = I₀ × exp(-2r²/w²) / (1 + z²/z_R²)
    
    where I₀ = 2P/(πw²) is the peak intensity at the focus.
    
    The atom experiences an AC Stark shift (light shift):
        U(r) = -α × |E(r)|² / 2 = -α × I(r) / (ε₀c)
    
    At the focus (r=0, z=0), this gives:
        U₀ = α × I₀ / (2ε₀c) = α × P / (πε₀c × w²)
    
    **Sign conventions:**
    
    For RED-detuned traps (most common):
    - Laser frequency below atomic resonance
    - Polarizability α > 0
    - U(r) < 0 at high intensity → attractive → trapping!
    
    For BLUE-detuned traps:
    - Laser frequency above atomic resonance
    - Polarizability α < 0
    - U(r) > 0 at high intensity → repulsive
    - Can still trap using dark regions (hollow beams)
    
    Parameters
    ----------
    power : float
        Laser power (W). Typical: 1-100 mW per tweezer
    waist : float
        Beam waist 1/e² intensity radius (m). Typical: 0.5-2 μm
    alpha : float
        Ground state polarizability (C²·m²/J in SI units).
        For Rb87 at 852 nm: α ≈ 710 × 4πε₀a₀³ ≈ 5.3×10⁻³⁹ C²m²/J
        
    Returns
    -------
    float
        Trap depth |U₀| in Joules. Always positive (depth).
        Convert to temperature: T = U₀/k_B
        Convert to frequency: f = U₀/(2πℏ)
        
    Example
    -------
    >>> from qpu_simulator.micro_physics.neutral_atoms.rydberg_gates import ATOM_DB
    >>> alpha = ATOM_DB["Rb87"]["alpha_ground"]
    >>> U0 = trap_depth(10e-3, 1e-6, alpha)  # 10 mW, 1 μm waist
    >>> U0 / KB * 1e3  # Convert to mK
    ~0.5-1.0  # Typical trap depth in mK
    
    **Physical intuition:**
    - More power → deeper trap (U₀ ∝ P)
    - Tighter focus → deeper trap (U₀ ∝ 1/w²)
    - Higher polarizability → deeper trap (U₀ ∝ α)
    """
    # Peak intensity at focus
    I0 = 2 * power / (np.pi * waist**2)  # W/m²
    
    # Trap depth: U = |α| × I / (2ε₀c)
    # We use |α| to return positive trap depth
    U0 = np.abs(alpha) * I0 / (2 * EPS0 * C)
    
    return U0


def trap_depth_from_temperature(temperature_K: float) -> float:
    """
    Convert a temperature to equivalent trap depth energy.
    
    Useful for quick estimates: "I need a trap depth of at least 1 mK"
    
    Parameters
    ----------
    temperature_K : float
        Temperature (K). Typical: 0.1-2 mK = 100-2000 μK
        
    Returns
    -------
    float
        Energy (J)
    """
    return KB * temperature_K


def trap_temperature(U0: float) -> float:
    """
    Convert trap depth to equivalent temperature.
    
    Parameters
    ----------
    U0 : float
        Trap depth (J)
        
    Returns
    -------
    float
        Temperature (K) where k_B T = U0
    """
    return U0 / KB


# =============================================================================
# TRAP FREQUENCIES
# =============================================================================

def trap_frequencies(U0: float, mass: float, waist: float,
                     wavelength: float) -> Tuple[float, float]:
    """
    Calculate radial and axial trap frequencies from trap depth.
    
    **What are trap frequencies?**
    
    Near the bottom of the trap, the potential is approximately harmonic
    (parabolic). The atom oscillates like a mass on a spring:
    
        U(r, z) ≈ U₀ - ½mω_r²r² - ½mω_z²z²
    
    where ω_r and ω_z are the radial and axial trap frequencies.
    
    **Why are radial and axial different?**
    
    The focused Gaussian beam has different curvature in different directions:
    
    - RADIALLY (perpendicular to beam): Sharp Gaussian profile
      Curvature ∝ 1/w², so ω_r = √(4U₀/mw²)
      
    - AXIALLY (along beam): Gradual Rayleigh range falloff
      Curvature ∝ 1/z_R², where z_R = πw²/λ is the Rayleigh range
      So ω_z = √(2U₀/mz_R²)
    
    Result: ω_r/ω_z ≈ √2 × z_R/w = √2 × πw/λ ~ 3-10
    
    The trap is "cigar-shaped" or "pancake-shaped" depending on geometry.
    
    **Why do trap frequencies matter?**
    
    1. Higher frequency → tighter confinement → smaller position uncertainty
       σ = √(k_B T / mω²), so σ ∝ 1/ω
       
    2. Trap frequencies set motional heating timescales
       - Heating rate in quanta/s compared to ω/2π tells you coherence time
       
    3. Ground state cooling requires resolved sidebands: γ_cooling < ω
    
    Parameters
    ----------
    U0 : float
        Trap depth (J) from trap_depth()
    mass : float
        Atomic mass (kg). Rb87: 1.443×10⁻²⁵ kg
    waist : float
        Beam waist (m). Typical: 0.5-2 μm
    wavelength : float
        Trap laser wavelength (m). Typical: 800-1100 nm
        
    Returns
    -------
    omega_r : float
        Radial trap frequency (rad/s). Typical: 2π × 100-200 kHz
    omega_z : float
        Axial trap frequency (rad/s). Typical: 2π × 10-30 kHz
        
    Example
    -------
    >>> omega_r, omega_z = trap_frequencies(U0, mass, 1e-6, 852e-9)
    >>> omega_r / (2*np.pi) / 1e3  # Convert to kHz
    ~100-200  # kHz
    >>> omega_r / omega_z  # Aspect ratio
    ~5-10  # Radial is much tighter
    
    **Physical intuition:**
    - Deeper trap → higher frequencies (ω ∝ √U₀)
    - Tighter focus → higher radial frequency (ω_r ∝ 1/w)
    - Heavier atom → lower frequencies (ω ∝ 1/√m)
    """
    # Rayleigh range: z_R = π w² / λ
    z_R = np.pi * waist**2 / wavelength
    
    # Radial frequency from Gaussian transverse curvature
    # U(r) ≈ U₀ × (1 - 2r²/w²) → ω_r² = 4U₀/(mw²)
    omega_r = np.sqrt(4 * U0 / (mass * waist**2))
    
    # Axial frequency from Lorentzian axial curvature
    # U(z) ≈ U₀ × (1 - z²/z_R²) → ω_z² = 2U₀/(mz_R²)
    omega_z = np.sqrt(2 * U0 / (mass * z_R**2))
    
    return omega_r, omega_z


def trap_frequency_from_depth_and_waist(U0: float, mass: float, 
                                         waist: float) -> float:
    """
    Calculate just the radial trap frequency (simplified version).
    
    Use this when you only care about radial confinement (most common case
    for in-plane atomic arrays).
    
    Parameters
    ----------
    U0 : float
        Trap depth (J)
    mass : float
        Atomic mass (kg)
    waist : float
        Beam waist (m)
        
    Returns
    -------
    float
        Radial trap frequency ω_r (rad/s)
    """
    return np.sqrt(4 * U0 / (mass * waist**2))


# =============================================================================
# THERMAL POSITION UNCERTAINTY
# =============================================================================

def position_uncertainty(temperature: float, mass: float, omega: float) -> float:
    """
    Calculate RMS position uncertainty of a trapped atom at temperature T.
    
    **What is position uncertainty?**
    
    Atoms in a trap aren't perfectly still - they have thermal kinetic energy
    that makes them oscillate. The position fluctuates around the trap center
    with an RMS spread:
    
        σ = √⟨x²⟩ = √(k_B T / mω²)
    
    **Physical derivation (equipartition theorem):**
    
    For a harmonic oscillator at thermal equilibrium:
    - Average potential energy: ⟨½mω²x²⟩ = ½k_B T
    - Solving: ⟨x²⟩ = k_B T / (mω²)
    - RMS position: σ = √⟨x²⟩ = √(k_B T / mω²)
    
    **Why does this matter for gates?**
    
    The Rydberg blockade interaction V = C₆/R⁶ depends on atom separation R.
    If atoms jiggle by σ, the interaction fluctuates:
    
        δR ≈ √2 × σ  (two atoms, independent motion)
        δV/V ≈ 6 × δR/R  (Taylor expansion of R⁻⁶)
    
    For σ = 50 nm, R = 3 μm: δV/V ≈ 6 × √2 × 0.05/3 ≈ 14%
    This is a BIG effect that limits gate fidelity!
    
    Parameters
    ----------
    temperature : float
        Atom temperature (K). Typical: 10-50 μK = 10e-6 to 50e-6 K
    mass : float
        Atomic mass (kg)
    omega : float
        Trap frequency (rad/s). Use radial frequency for in-plane motion.
        
    Returns
    -------
    float
        RMS position spread σ (m). Typical: 30-100 nm
        
    Example
    -------
    >>> sigma = position_uncertainty(20e-6, mass_Rb87, omega_r)
    >>> sigma * 1e9  # Convert to nm
    ~50  # nm
    
    **Temperature scaling:**
    - σ ∝ √T, so 4× lower temperature → 2× smaller position uncertainty
    - This is why cooling matters so much for high-fidelity gates!
    
    **Quantum limit:**
    At very low T, quantum zero-point motion dominates:
    σ_quantum = √(ℏ/2mω) ≈ 10-30 nm for typical traps
    
    The thermal formula is valid when k_B T >> ℏω, which is usually true
    for T > 1 μK and ω/2π < 1 MHz.
    """
    return np.sqrt(KB * temperature / (mass * omega**2))


def quantum_ground_state_size(mass: float, omega: float) -> float:
    """
    Calculate zero-point motion (ground state wavefunction size).
    
    **What is zero-point motion?**
    
    Even at absolute zero temperature (T = 0), quantum mechanics says the
    atom still "jiggles" due to the Heisenberg uncertainty principle:
    
        Δx × Δp ≥ ℏ/2
    
    For the quantum ground state of a harmonic oscillator:
    
        σ₀ = √(ℏ / 2mω)
    
    This is the FUNDAMENTAL LIMIT on how well you can localize an atom
    in a harmonic trap, no matter how cold you make it.
    
    **Numerical values:**
    For Rb87 at ω = 2π × 100 kHz:
    σ₀ = √(1.05e-34 / (2 × 1.44e-25 × 2π × 1e5)) ≈ 24 nm
    
    **When does it matter?**
    When k_B T < ℏω, the thermal motion is "frozen out" and you're limited
    by quantum fluctuations. This happens at:
    T < ℏω/k_B ≈ 4.8 μK × (ω/2π / 100 kHz)
    
    Most Rydberg gate experiments operate in the thermal regime (T ~ 10-50 μK),
    so quantum zero-point motion is typically a small correction.
    
    Parameters
    ----------
    mass : float
        Atomic mass (kg)
    omega : float
        Trap frequency (rad/s)
        
    Returns
    -------
    float
        Ground state wavefunction width σ₀ (m)
    """
    return np.sqrt(HBAR / (2 * mass * omega))


def thermal_de_broglie_wavelength(temperature: float, mass: float) -> float:
    """
    Calculate the thermal de Broglie wavelength.
    
    λ_dB = h / √(2π m k_B T)
    
    This tells you when quantum effects become important:
    - When λ_dB > inter-atomic spacing: quantum degeneracy (BEC regime)
    - When λ_dB > σ: quantum localization matters
    
    For Rb87 at 20 μK: λ_dB ≈ 0.4 μm
    
    Parameters
    ----------
    temperature : float
        Temperature (K)
    mass : float
        Atomic mass (kg)
        
    Returns
    -------
    float
        de Broglie wavelength (m)
    """
    h = 2 * np.pi * HBAR
    return h / np.sqrt(2 * np.pi * mass * KB * temperature)


# =============================================================================
# RYDBERG ANTI-TRAPPING
# =============================================================================
# THIS IS A CRITICAL ISSUE FOR RYDBERG GATES!

def anti_trap_potential(U0_ground: float, alpha_ratio: float) -> float:
    """
    Calculate the anti-trapping potential seen by Rydberg state atoms.
    
    **THE ANTI-TRAPPING PROBLEM**
    
    This is one of the most important issues in Rydberg gate design!
    
    **Ground state** atoms have polarizability α_g > 0 at typical trap wavelengths
    (like 852 nm or 1064 nm). This means they're attracted to intensity maxima.
    Good - they stay trapped!
    
    **Rydberg state** atoms have polarizability α_r < 0 (opposite sign!).
    This means they're REPELLED from intensity maxima.
    Bad - they get pushed out of the trap!
    
    **What happens during a gate:**
    
    1. Atom starts in ground state |1⟩, sitting happily at trap center
    2. Laser pulse excites it to Rydberg state |r⟩
    3. The trap potential FLIPS from attractive to repulsive
    4. Atom starts accelerating outward
    5. If it moves too far before being de-excited, it escapes!
    
    The anti-trap potential magnitude is:
    
        U_anti = |α_r / α_g| × U_ground
    
    **How big is this effect?**
    
    For n=70 Rb87: |α_r/α_g| ≈ 300
    So if U_ground = 1 mK, then U_anti = 300 mK ≈ 0.3 K!
    
    This is HUGE - the atom gets violently ejected from the trap
    if it stays in the Rydberg state too long.
    
    **Solutions in experiments:**
    
    1. **Fast gates**: Complete gate before atom moves significantly
    2. **Trap blanking**: Turn off trap during Rydberg excitation
    3. **Magic wavelength traps**: Special wavelengths where α_r ≈ 0
    
    Parameters
    ----------
    U0_ground : float
        Ground state trap depth (J) from trap_depth()
    alpha_ratio : float
        |α_Rydberg / α_ground| - from atomic database
        Typical: 100-1000 for n = 50-100
        
    Returns
    -------
    float
        Anti-trap potential magnitude (J). Positive = repulsive.
        
    Example
    -------
    >>> U0 = trap_depth(10e-3, 1e-6, alpha_ground)  # ~0.5 mK ground trap
    >>> alpha_ratio = 300  # for n=70
    >>> U_anti = anti_trap_potential(U0, alpha_ratio)
    >>> U_anti / KB * 1e3  # mK
    ~150  # mK - HUGE repulsive potential!
    """
    return U0_ground * alpha_ratio


def anti_trap_frequency(U0_ground: float, alpha_ratio: float,
                        mass: float, waist: float) -> float:
    """
    Calculate the anti-trapping "frequency" (exponential growth rate).
    
    The anti-trapping potential is an inverted parabola:
    
        U_anti(r) = +|α_r/α_g| × U₀ × (1 - 2r²/w²)
    
    This leads to EXPONENTIAL motion away from center:
    
        r(t) = r₀ × cosh(ω_anti × t) + (v₀/ω_anti) × sinh(ω_anti × t)
    
    where ω_anti = √(4 × |α_r/α_g| × U₀ / mw²)
    
    **Physical meaning:**
    
    ω_anti is NOT an oscillation frequency - it's an exponential growth rate.
    After time t = 1/ω_anti, the atom's distance from center has grown by e ≈ 2.7×.
    
    **Numerical example:**
    For typical parameters (U₀ = 0.5 mK, w = 1 μm, m_Rb, α_ratio = 300):
    ω_anti ≈ 2π × 3 MHz
    
    So in 50 ns, the atom moves by a factor of cosh(2π × 3 MHz × 50 ns) ≈ 2×
    
    This is why gate times must be < 1 μs!
    
    Parameters
    ----------
    U0_ground : float
        Ground state trap depth (J)
    alpha_ratio : float
        |α_Rydberg / α_ground|
    mass : float
        Atomic mass (kg)
    waist : float
        Trap beam waist (m)
        
    Returns
    -------
    float
        Anti-trap exponential growth rate ω_anti (rad/s)
    """
    U_anti = alpha_ratio * U0_ground
    return np.sqrt(4 * U_anti / (mass * waist**2))


def thermal_velocity(temperature: float, mass: float) -> float:
    """
    Calculate thermal velocity of atoms at temperature T.
    
    v_thermal = √(k_B T / m)
    
    This is the characteristic speed at which atoms "wander" in the trap.
    It's the 1D RMS velocity from the Maxwell-Boltzmann distribution.
    
    **Numerical values:**
    For Rb87 at T = 20 μK:
    v_thermal = √(1.38e-23 × 20e-6 / 1.44e-25) ≈ 4.4 cm/s = 44 μm/ms
    
    During a 1 μs gate: drift ~ 44 nm (comparable to position uncertainty)
    
    Parameters
    ----------
    temperature : float
        Temperature (K)
    mass : float
        Atomic mass (kg)
        
    Returns
    -------
    float
        Thermal velocity (m/s)
    """
    return np.sqrt(KB * temperature / mass)


def atom_loss_probability(gate_time: float, U0: float, alpha_ratio: float,
                          mass: float, waist: float, temperature: float,
                          rydberg_fraction: float = 0.3,
                          trap_on_during_rydberg: bool = True) -> float:
    """
    Estimate atom loss probability due to Rydberg anti-trapping.
    
    **THE PROBLEM:**
    
    When an atom is excited to the Rydberg state, the optical trap becomes
    repulsive (anti-trapping). The atom accelerates outward, and if it
    moves too far, it can't be recaptured when de-excited.
    
    **TWO OPERATING MODES:**
    
    1. **Trap ON during gate** (trap_on_during_rydberg=True, DEFAULT):
       - Modern approach used with fast gates
       - Anti-trapping causes exponential position growth
       - Position: r(t) ≈ r₀ × cosh(ω_anti × t)
       - Loss if atom exceeds ~2 beam waists
       
    2. **Trap OFF during gate** (trap_on_during_rydberg=False):
       - Traditional "trap blanking" approach
       - Atom undergoes ballistic flight at thermal velocity
       - Simpler but limits gate speed (must wait for trap to turn back on)
    
    **TYPICAL LOSS RATES:**
    
    From de Léséleuc et al., PRA 97, 053803 (2018):
    - ~1-5% loss per gate at ~1 μs gate times
    - Loss increases with trap power (deeper trap → stronger anti-trapping)
    
    **MITIGATION STRATEGIES:**
    
    1. Fast gates (< 500 ns): Less time for atom to move
    2. Shallow traps: Less anti-trapping, but also less confinement
    3. Magic wavelength traps: Special λ where α_r ≈ 0 (challenging!)
    4. Lower n states: α_r ∝ n⁷, so lower n has less anti-trapping
    
    Parameters
    ----------
    gate_time : float
        Total gate duration (s). Typical: 0.2-2 μs
    U0 : float
        Ground state trap depth (J)
    alpha_ratio : float
        |α_Rydberg / α_ground|. Typical: 100-1000
    mass : float
        Atomic mass (kg)
    waist : float
        Trap beam waist (m)
    temperature : float
        Atom temperature (K)
    rydberg_fraction : float
        Average fraction of gate time spent in Rydberg state.
        For CZ gate: ~0.3-0.5 (atoms spend ~1/3 of time in |r⟩)
    trap_on_during_rydberg : bool
        If True (default), use anti-trapping model.
        If False, use ballistic flight model.
        
    Returns
    -------
    float
        Probability of atom loss per gate (0 to 1)
        
    Example
    -------
    >>> P_loss = atom_loss_probability(
    ...     gate_time=1e-6,
    ...     U0=trap_depth(10e-3, 1e-6, alpha_ground),
    ...     alpha_ratio=300,
    ...     mass=mass_Rb87,
    ...     waist=1e-6,
    ...     temperature=20e-6
    ... )
    >>> P_loss
    ~0.01-0.05  # 1-5% loss probability
    """
    # Time spent in Rydberg state
    t_rydberg = rydberg_fraction * gate_time
    
    # Ground state trap frequency
    omega_trap = np.sqrt(4 * U0 / (mass * waist**2))
    
    # Thermal velocity
    v_thermal = np.sqrt(KB * temperature / mass)
    
    # Capture range: atom can be recaptured if within ~2 waists
    capture_range = 2.0 * waist
    
    if trap_on_during_rydberg:
        # ANTI-TRAPPING MODEL
        # Position grows exponentially: r(t) ≈ r₀ × cosh(ω_anti × t)
        
        omega_anti = np.sqrt(4 * alpha_ratio * U0 / (mass * waist**2))
        
        if omega_anti > 0 and t_rydberg > 0:
            # Initial thermal position spread
            sigma_initial = np.sqrt(KB * temperature / (mass * omega_trap**2))
            
            # Growth factors
            cosh_factor = np.cosh(omega_anti * t_rydberg)
            sinh_factor = np.sinh(omega_anti * t_rydberg)
            
            # Final position spread
            # From initial position: σ_pos × cosh(ω_anti × t)
            # From initial velocity: (v/ω_anti) × sinh(ω_anti × t)
            final_sigma_pos = sigma_initial * cosh_factor
            final_sigma_vel = (v_thermal / omega_anti) * sinh_factor
            final_sigma = np.sqrt(final_sigma_pos**2 + final_sigma_vel**2)
            
            # Loss probability from Gaussian tail
            if final_sigma > 0:
                P_loss = 1.0 - np.exp(-(capture_range / final_sigma)**2 / 2)
            else:
                P_loss = 0.0
        else:
            P_loss = 0.0
            
    else:
        # BALLISTIC FLIGHT MODEL (trap blanking)
        # Atom drifts at thermal velocity
        
        drift_distance = v_thermal * t_rydberg
        
        if drift_distance > 0:
            P_loss = 1.0 - np.exp(-(drift_distance / capture_range)**2 / 2)
        else:
            P_loss = 0.0
    
    return np.clip(P_loss, 0.0, 1.0)


def effective_loss_rate(gate_time: float, U0: float, alpha_ratio: float,
                        mass: float, waist: float, temperature: float,
                        rydberg_fraction: float = 0.3) -> float:
    """
    Convert atom loss probability to effective Lindblad decay rate.
    
    The master equation uses continuous decay rates, not discrete probabilities.
    If P_loss is the loss probability over time τ_gate:
    
        P_loss = 1 - exp(-γ_loss × τ_gate)
        γ_loss = -ln(1 - P_loss) / τ_gate
    
    For small P_loss: γ_loss ≈ P_loss / τ_gate
    
    Parameters
    ----------
    gate_time : float
        Gate duration (s)
    U0 : float
        Ground state trap depth (J)
    alpha_ratio : float
        |α_Rydberg / α_ground|
    mass : float
        Atomic mass (kg)
    waist : float
        Trap beam waist (m)
    temperature : float
        Atom temperature (K)
    rydberg_fraction : float
        Fraction of time in Rydberg state
        
    Returns
    -------
    float
        Effective loss rate γ_loss (Hz) for Lindblad equation
        
    Example
    -------
    >>> gamma_loss = effective_loss_rate(1e-6, U0, 300, mass, waist, 20e-6)
    >>> gamma_loss / 1e3  # kHz
    ~10-50  # Typical loss rate in kHz
    """
    P_loss = atom_loss_probability(gate_time, U0, alpha_ratio, mass,
                                   waist, temperature, rydberg_fraction)
    
    # Convert probability to rate
    if P_loss >= 0.99:
        # Saturate at ~5 decays per gate time for numerical stability
        gamma_loss = 5.0 / gate_time
    elif P_loss > 0:
        gamma_loss = -np.log(1 - P_loss) / gate_time
    else:
        gamma_loss = 0.0
    
    # Cap at reasonable maximum for Lindblad validity
    # Allow up to ~1 decay per gate time
    max_rate = 1.0 / gate_time if gate_time > 0 else 1e6
    gamma_loss = min(gamma_loss, max_rate)
    
    return gamma_loss


# =============================================================================
# BLOCKADE FLUCTUATIONS FROM THERMAL MOTION
# =============================================================================

def blockade_fluctuation(R0: float, sigma_r: float) -> float:
    """
    Calculate relative blockade fluctuation due to position uncertainty.
    
    **THE PROBLEM:**
    
    The CZ gate works by accumulating a precise π phase from the Rydberg
    blockade interaction V = C₆/R⁶. But if the atoms jiggle around, the
    distance R fluctuates, and so does V.
    
    **THE MATH:**
    
    If both atoms have independent position uncertainty σ_r, the distance
    uncertainty is:
        δR = √(σ₁² + σ₂²) = √2 × σ_r  (uncorrelated motion)
    
    Since V ∝ R⁻⁶, a Taylor expansion gives:
        δV/V ≈ -6 × δR/R = -6√2 × σ_r/R
    
    We return the magnitude |δV/V| = 6√2 × σ_r/R ≈ 8.5 × σ_r/R
    
    **EXAMPLE:**
    For σ_r = 50 nm, R = 3 μm:
    δV/V = 6 × √2 × 0.05/3 ≈ 14%
    
    A 14% interaction fluctuation causes ~14% phase error, which is HUGE!
    
    Parameters
    ----------
    R0 : float
        Nominal atom separation (m). Typical: 2-5 μm
    sigma_r : float
        RMS position uncertainty per atom (m). Typical: 30-100 nm
        
    Returns
    -------
    float
        Relative interaction fluctuation |δV/V| (dimensionless)
        
    Example
    -------
    >>> delta_V_V = blockade_fluctuation(3e-6, 50e-9)
    >>> delta_V_V
    0.14  # 14% fluctuation
    """
    # Distance uncertainty from two atoms
    delta_R = np.sqrt(2) * sigma_r
    
    # Relative blockade fluctuation
    delta_V_over_V = 6 * delta_R / R0
    
    return delta_V_over_V


def thermal_dephasing_rate(delta_V_over_V: float, V0: float,
                           Omega: float = None) -> float:
    """
    Estimate dephasing rate from blockade fluctuations.
    
    **PHYSICS:**
    
    The interaction fluctuation δV causes random phase accumulation during
    the gate. This manifests as dephasing in the Lindblad master equation.
    
    **BLOCKADE REGIME DEPENDENCE:**
    
    The effect depends on how strong the blockade is:
    
    1. **WEAK BLOCKADE** (V/Ω < 3):
       - The state |rr⟩ gets significantly populated
       - Phase accumulates during |rr⟩ occupation
       - Dephasing rate: γ ~ (δV/V)² × (V/Ω)² × Ω
       
    2. **STRONG BLOCKADE** (V/Ω > 10):
       - |rr⟩ population is suppressed by (Ω/V)²
       - Much smaller dephasing effect
       - γ ~ (δV/V)² × (Ω/V)² × Ω
       
    3. **INTERMEDIATE** (3 < V/Ω < 10):
       - Smooth interpolation between regimes
    
    Parameters
    ----------
    delta_V_over_V : float
        Relative blockade fluctuation from blockade_fluctuation()
    V0 : float
        Nominal blockade strength (rad/s)
    Omega : float, optional
        Rabi frequency (rad/s). If None, uses 2π × 5 MHz.
        
    Returns
    -------
    float
        Effective dephasing rate γ_thermal (Hz)
        
    Example
    -------
    >>> delta_V_V = blockade_fluctuation(3e-6, 50e-9)  # ~14%
    >>> V = 2*np.pi * 200e6  # 200 MHz blockade
    >>> Omega = 2*np.pi * 5e6  # 5 MHz Rabi
    >>> gamma = thermal_dephasing_rate(delta_V_V, V, Omega)
    >>> gamma / 1e3  # kHz
    ~5-20  # Depends on blockade regime
    """
    # Default Rabi frequency if not provided
    if Omega is None or Omega <= 0:
        Omega = 2 * np.pi * 5e6  # 5 MHz typical
    
    V_over_Omega = np.abs(V0) / np.abs(Omega)
    
    if V_over_Omega < 3:
        # WEAK BLOCKADE: |rr⟩ gets populated
        infidelity = (delta_V_over_V**2) * (V_over_Omega**2)
        gamma_thermal = infidelity * np.abs(Omega) / (2 * np.pi)
        
    elif V_over_Omega > 10:
        # STRONG BLOCKADE: |rr⟩ suppressed by blockade
        suppression = (np.abs(Omega) / np.abs(V0))**2
        infidelity = (delta_V_over_V**2) * suppression
        gamma_thermal = infidelity * np.abs(Omega) / (2 * np.pi)
        
    else:
        # INTERMEDIATE: Smooth interpolation
        x = (V_over_Omega - 3) / 7  # 0 at V/Ω=3, 1 at V/Ω=10
        x = np.clip(x, 0, 1)
        
        # Weak blockade rate
        infidelity_weak = (delta_V_over_V**2) * (V_over_Omega**2)
        gamma_weak = infidelity_weak * np.abs(Omega) / (2 * np.pi)
        
        # Strong blockade rate
        suppression = (np.abs(Omega) / np.abs(V0))**2
        infidelity_strong = (delta_V_over_V**2) * suppression
        gamma_strong = infidelity_strong * np.abs(Omega) / (2 * np.pi)
        
        # Cubic smoothstep interpolation
        smooth = 3*x**2 - 2*x**3
        gamma_thermal = gamma_weak * (1 - smooth) + gamma_strong * smooth
    
    return min(gamma_thermal, 10e6)  # Cap at 10 MHz sanity check


def doppler_dephasing_rate(
    temperature: float,
    mass: float,
    k_eff: float,
    gate_time: float
) -> float:
    """
    Calculate dephasing rate from Doppler shifts due to thermal atomic motion.
    
    **PHYSICS:**
    
    Atoms at finite temperature have thermal velocity v ~ sqrt(k_B*T/m).
    When driven by a laser, they see a Doppler-shifted frequency:
    
        δω_Doppler = k_eff × v
    
    where k_eff is the effective wavevector of the two-photon transition.
    For counter-propagating beams: k_eff ≈ |k_1 - k_2| (can be small)
    For co-propagating beams: k_eff ≈ k_1 + k_2 (larger Doppler)
    
    This Doppler shift varies from shot-to-shot, causing dephasing.
    The RMS frequency deviation is:
    
        δω_rms = k_eff × v_rms = k_eff × sqrt(k_B*T/m)
    
    The accumulated phase error over the gate time:
    
        δφ = δω_rms × t_gate
    
    This manifests as a dephasing rate:
    
        γ_Doppler ≈ (δω_rms)² × t_gate = k_eff² × (k_B*T/m) × t_gate
    
    Parameters
    ----------
    temperature : float
        Atom temperature (K). Typical: 10-50 μK
    mass : float
        Atomic mass (kg)
    k_eff : float
        Effective two-photon wavevector (rad/m).
        For Rb87 with 780nm + 480nm counter-propagating: 
        k_eff ≈ 2π(1/780nm - 1/480nm) ≈ 5.4e6 rad/m
    gate_time : float
        Gate duration (s). Typical: 100-500 ns
        
    Returns
    -------
    float
        Doppler dephasing rate (Hz)
        
    Example
    -------
    >>> # Rb87 at 20 μK with counter-propagating beams, 200 ns gate
    >>> k_eff = 2*np.pi * (1/780e-9 - 1/480e-9)  # ~5.4e6 rad/m
    >>> gamma = doppler_dephasing_rate(20e-6, 1.44e-25, k_eff, 200e-9)
    >>> print(f"γ_Doppler = {gamma:.0f} Hz")  # ~few kHz
    
    Notes
    -----
    This is often a DOMINANT error source at higher temperatures!
    Experiments mitigate this by:
    1. Using co-propagating beams (smaller k_eff)
    2. Sideband cooling to motional ground state
    3. Operating at very low temperature (< 10 μK)
    
    References
    ----------
    [1] de Léséleuc et al., PRA 97, 053803 (2018)
    [2] Bluvstein PhD Thesis, Harvard (2024) - Section 2.5
    """
    # RMS thermal velocity
    v_rms = np.sqrt(KB * temperature / mass)
    
    # RMS Doppler shift
    delta_omega_rms = k_eff * v_rms
    
    # Phase variance accumulates as (δω)² × t
    # Dephasing rate ≈ variance / t = (δω)² × t / t = (δω)²
    # But we want rate in Hz, so: γ = (δω_rms)² × t_gate
    gamma_doppler = (delta_omega_rms ** 2) * gate_time
    
    return gamma_doppler


def intensity_noise_dephasing_rate(
    trap_depth_J: float,
    intensity_noise_frac: float,
    gate_time: float = 200e-9,
    differential_stark_fraction: float = 0.01,
) -> float:
    """
    Calculate dephasing from trap intensity fluctuations.
    
    **PHYSICS:**
    
    The trap depth U₀ depends on laser intensity: U₀ = α×I/(2ε₀c)
    If intensity fluctuates by δI/I, the trap depth fluctuates by δU₀/U₀ = δI/I.
    
    For a magic wavelength trap, ground state differential Stark shift is minimized.
    However, during the gate, atoms spend time in the Rydberg state which has 
    different (often opposite sign) polarizability, causing differential shift.
    
    The effective differential Stark shift is:
    
        δω_diff ≈ f_diff × U₀/ℏ × (δI/I)
    
    where f_diff ~ 0.01-0.1 is the fractional differential polarizability.
    
    Dephasing from accumulated phase noise over gate time t_gate:
    
        γ_intensity ≈ δω_diff² × t_gate / 2
    
    This is the "quasi-static" limit where noise is slow compared to gate.
    
    Parameters
    ----------
    trap_depth_J : float
        Trap depth in Joules. Typical: 1-3 mK × k_B
    intensity_noise_frac : float
        RMS fractional intensity noise δI/I. Typical: 0.001-0.01 (0.1-1%)
    gate_time : float
        Gate duration (s). Typical: 100-500 ns
    differential_stark_fraction : float
        Fraction of Stark shift that is differential between qubit states.
        For magic wavelength: ~0.01. For non-magic: ~0.1-1
        
    Returns
    -------
    float
        Intensity noise dephasing rate (Hz)
        
    Example
    -------
    >>> U0 = 1e-3 * KB  # 1 mK trap depth
    >>> gamma = intensity_noise_dephasing_rate(U0, 0.01, 200e-9)
    >>> print(f"γ_intensity = {gamma:.1f} Hz")
    
    Notes
    -----
    Modern experiments achieve δI/I < 0.1% with intensity stabilization.
    Combined with magic wavelength operation, this is typically a subdominant
    error source (~1-10 Hz dephasing rate), much smaller than laser dephasing.
    """
    # Total Stark shift frequency from trap depth
    omega_trap = trap_depth_J / HBAR
    
    # Differential Stark shift fluctuation (magic wavelength suppression)
    delta_omega_diff = omega_trap * intensity_noise_frac * differential_stark_fraction
    
    # Dephasing rate from accumulated phase noise
    # For quasi-static noise: T2* ~ 1/(δω × √t), so γ ~ δω² × t / 2
    # But we want a rate, so we divide by gate time to get effective Hz
    # Simpler model: γ ≈ δω_diff (direct frequency noise → dephasing)
    gamma_intensity = delta_omega_diff
    
    return gamma_intensity


def thermal_infidelity_estimate(R0: float, sigma_r: float, V0: float,
                                Omega: float, gate_time: float) -> float:
    """
    Quick estimate of gate infidelity from thermal position fluctuations.
    
    **FORMULA:**
    
    Phase error: δφ ≈ δV × t_gate
    Infidelity: ε ≈ (δφ)² = (δV/V × V × t_gate)²
    
    With δV/V = 6σ/R and t_gate ~ 2π/Ω:
    ε ≈ (6σ/R)² × (V/Ω)²
    
    **TYPICAL VALUES:**
    
    For σ = 50 nm, R = 3 μm, V/Ω = 40:
    ε ≈ (6 × 0.05/3)² × 40² ≈ 0.01 × 1600 = 16%... 
    
    But wait! In strong blockade, |rr⟩ is suppressed, so the actual error
    is reduced by (Ω/V)² ≈ 0.0006. Real error: ~0.01%
    
    This function gives the NAIVE estimate without blockade suppression.
    
    Parameters
    ----------
    R0 : float
        Atom spacing (m)
    sigma_r : float
        Position uncertainty (m)
    V0 : float
        Blockade interaction (rad/s)
    Omega : float
        Rabi frequency (rad/s)
    gate_time : float
        Gate duration (s)
        
    Returns
    -------
    float
        Estimated infidelity (0 to 1)
    """
    delta_V_over_V = blockade_fluctuation(R0, sigma_r)
    delta_phi = delta_V_over_V * V0 * gate_time
    return delta_phi**2


# =============================================================================
# MOTIONAL HEATING
# =============================================================================

def photon_recoil_energy(mass: float, wavelength: float) -> float:
    """
    Calculate photon recoil energy.
    
    When an atom absorbs or emits a photon, it recoils with momentum ℏk.
    The kinetic energy gained is:
    
        E_recoil = (ℏk)² / (2m) = h² / (2m λ²)
    
    **Physical significance:**
    
    This sets the minimum temperature achievable with Doppler cooling
    (recoil limit), and contributes to motional heating during Rydberg
    excitation.
    
    **Numerical values:**
    For Rb87 at 780 nm: E_recoil/k_B ≈ 180 nK
    For Cs133 at 852 nm: E_recoil/k_B ≈ 100 nK
    
    Parameters
    ----------
    mass : float
        Atomic mass (kg)
    wavelength : float
        Photon wavelength (m)
        
    Returns
    -------
    float
        Recoil energy (J)
    """
    k = 2 * np.pi / wavelength
    return (HBAR * k)**2 / (2 * mass)


def recoil_temperature(mass: float, wavelength: float) -> float:
    """
    Calculate recoil temperature T_recoil = E_recoil / k_B.
    
    This is a fundamental temperature scale for laser cooling.
    Below T_recoil, you need techniques like sideband cooling.
    
    Parameters
    ----------
    mass : float
        Atomic mass (kg)
    wavelength : float
        Photon wavelength (m)
        
    Returns
    -------
    float
        Recoil temperature (K)
    """
    return photon_recoil_energy(mass, wavelength) / KB


def trap_heating_rate_intensity_noise(omega_trap: float,
                                       relative_intensity_noise: float = 1e-4,
                                       noise_bandwidth: float = 1e6) -> float:
    """
    Estimate motional heating rate from trap laser intensity noise.
    
    **PHYSICS:**
    
    Intensity fluctuations modulate the trap depth, causing parametric heating.
    If the noise spectrum has power at 2×trap frequency, energy is pumped
    into the atomic motion.
    
    Heating rate: dn/dt = (ω/4) × (δI/I)² × bandwidth
    
    **TYPICAL VALUES:**
    
    - State-of-art: dn/dt ~ 1-10 quanta/s at ω/2π ~ 100 kHz
    - Commercial lasers: dn/dt ~ 100-1000 quanta/s
    
    For 1 μs gates, heating is typically negligible (< 0.01 quanta).
    
    Parameters
    ----------
    omega_trap : float
        Trap frequency (rad/s)
    relative_intensity_noise : float
        RMS relative intensity noise δI/I. Typical: 10⁻⁴ to 10⁻³
    noise_bandwidth : float
        Relevant noise bandwidth (Hz). Typically ~MHz.
        
    Returns
    -------
    float
        Heating rate in quanta/second
    """
    return (omega_trap / 4) * (relative_intensity_noise**2) * noise_bandwidth


def trap_heating_rate_pointing_noise(omega_trap: float, waist: float,
                                      pointing_noise_rad: float = 1e-6) -> float:
    """
    Estimate heating rate from trap beam pointing noise.
    
    Pointing fluctuations shift the trap center, directly exciting
    the atomic motion.
    
    Parameters
    ----------
    omega_trap : float
        Trap frequency (rad/s)
    waist : float
        Beam waist (m)
    pointing_noise_rad : float
        RMS pointing noise (radians). Typical: ~1 μrad
        
    Returns
    -------
    float
        Heating rate in quanta/second
    """
    # Position noise at trap ~ pointing × distance (assume 1m)
    position_noise = pointing_noise_rad * 1.0
    
    # Fraction of trap size
    noise_fraction = position_noise / waist
    
    # Heating rate
    return omega_trap * noise_fraction**2


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def compute_trap_properties(power: float, waist: float, wavelength: float,
                            alpha: float, mass: float, 
                            temperature: float) -> dict:
    """
    Compute all trap-related properties in one call.
    
    This is a convenience function that computes everything you need
    for noise calculations.
    
    Parameters
    ----------
    power : float
        Trap laser power (W)
    waist : float
        Beam waist (m)
    wavelength : float
        Trap laser wavelength (m)
    alpha : float
        Ground state polarizability (SI units)
    mass : float
        Atomic mass (kg)
    temperature : float
        Atom temperature (K)
        
    Returns
    -------
    dict
        Dictionary containing:
        - U0: Trap depth (J)
        - omega_r: Radial trap frequency (rad/s)
        - omega_z: Axial trap frequency (rad/s)
        - sigma_r: Radial position uncertainty (m)
        - sigma_z: Axial position uncertainty (m)
        - v_thermal: Thermal velocity (m/s)
        - sigma_quantum: Quantum ground state size (m)
        
    Example
    -------
    >>> props = compute_trap_properties(
    ...     power=10e-3, waist=1e-6, wavelength=852e-9,
    ...     alpha=alpha_Rb87, mass=mass_Rb87, temperature=20e-6
    ... )
    >>> props["sigma_r"] * 1e9  # nm
    ~50
    """
    U0 = trap_depth(power, waist, alpha)
    omega_r, omega_z = trap_frequencies(U0, mass, waist, wavelength)
    sigma_r = position_uncertainty(temperature, mass, omega_r)
    sigma_z = position_uncertainty(temperature, mass, omega_z)
    v_thermal = thermal_velocity(temperature, mass)
    sigma_quantum = quantum_ground_state_size(mass, omega_r)
    
    return {
        "U0": U0,
        "omega_r": omega_r,
        "omega_z": omega_z,
        "sigma_r": sigma_r,
        "sigma_z": sigma_z,
        "v_thermal": v_thermal,
        "sigma_quantum": sigma_quantum,
        "U0_mK": U0 / KB * 1e3,
        "omega_r_kHz": omega_r / (2 * np.pi) / 1e3,
        "omega_z_kHz": omega_z / (2 * np.pi) / 1e3,
        "sigma_r_nm": sigma_r * 1e9,
        "sigma_z_nm": sigma_z * 1e9,
    }


def compute_trap_dependent_noise(
    species: str,
    tweezer_power: float,
    tweezer_waist: float,
    temperature: float,
    spacing: float,
    gate_time: float,
    n_rydberg: int = 70,
    gamma_phi_laser: float = 1e4,
    Omega_1: float = 0.0,
    Delta_e: float = 2*np.pi*5e9,
    intermediate_state: str = None,
    Omega_eff: float = None,
    tweezer_wavelength_nm: float = 1064.0,
    # New parameters for additional noise sources
    include_doppler: bool = True,
    include_intensity_noise: bool = True,
    intensity_noise_frac: float = 0.01,
    rydberg_wavelength_1_nm: float = 780.0,  # First leg wavelength
    rydberg_wavelength_2_nm: float = 480.0,  # Second leg wavelength
    counter_propagating: bool = True,  # Beam geometry
) -> dict:
    """
    Compute all trap-dependent noise rates including magic wavelength analysis.
    
    This UNIFIED function connects tweezer parameters (power, waist, wavelength)
    to all noise rates in the Lindblad equation, combining trap-dependent noise
    and magic wavelength effects into a single comprehensive computation.
    
    Parameters
    ----------
    species : str
        Atomic species ("Rb87" or "Cs133")
    tweezer_power : float
        Tweezer power (W)
    tweezer_waist : float
        Tweezer beam waist (m)
    temperature : float
        Atom temperature (K)
    spacing : float
        Inter-atom spacing (m)
    gate_time : float
        Gate duration (s)
    n_rydberg : int
        Rydberg principal quantum number (default: 70)
    gamma_phi_laser : float
        Laser-induced dephasing rate (rad/s, default: 1e4)
    Omega_1 : float
        First-leg Rabi frequency for intermediate state scattering (rad/s)
    Delta_e : float
        Detuning from intermediate state (rad/s)
    intermediate_state : str, optional
        Intermediate state label (e.g., "5P3/2"). Auto-detected if None.
    Omega_eff : float, optional
        Effective two-photon Rabi frequency (rad/s) for thermal dephasing.
    tweezer_wavelength_nm : float
        Tweezer wavelength in nm (default: 1064 nm). Affects magic wavelength
        analysis and anti-trapping rate.
    
    Returns
    -------
    dict
        Comprehensive noise dictionary containing:
        
        Trap parameters:
        - trap_depth_uK: Trap depth in μK
        - trap_freq_radial_kHz: Radial trap frequency in kHz
        - position_uncertainty_nm: Position uncertainty σ_r in nm
        - V_over_2pi_MHz: Blockade strength V/(2π) in MHz
        
        Noise rates (all in Hz):
        - gamma_r: Rydberg state decay rate
        - gamma_scatter_intermediate: Intermediate state scattering rate
        - gamma_phi_laser: Laser-induced dephasing rate
        - gamma_phi_thermal: Thermal/motional dephasing rate
        - gamma_loss_antitrap: Anti-trapping loss rate (from magic analysis)
        - gamma_loss_background: Background gas collision rate
        
        Blockade fluctuations:
        - blockade_fluctuation_percent: δV/V in percent
        - intermediate_linewidth_MHz: Intermediate state linewidth
        
        Magic wavelength analysis:
        - alpha_ratio: |α_rydberg/α_ground|
        - alpha_ground_au: Ground state polarizability (atomic units)
        - alpha_rydberg_au: Rydberg state polarizability (atomic units)
        - differential_shift_Hz: Differential light shift (Hz)
        - magic_enhancement: Magic wavelength enhancement factor (0-1)
        - wavelength_nm: Trap wavelength used
    """
    from .atom_database import get_atom_properties, get_C6, get_rydberg_polarizability, get_rydberg_lifetime
    from .laser_physics import rydberg_blockade
    
    atom = get_atom_properties(species)
    C6 = get_C6(n_rydberg, species)
    
    # =========================================================================
    # TRAP PARAMETERS
    # =========================================================================
    U0 = trap_depth(tweezer_power, tweezer_waist, atom["alpha_ground"])
    omega_r, omega_z = trap_frequencies(U0, atom["mass"], tweezer_waist, 
                                        atom["trap_wavelength"])
    sigma_r = position_uncertainty(temperature, atom["mass"], omega_r)
    V = rydberg_blockade(C6, spacing)
    
    # =========================================================================
    # BLOCKADE FLUCTUATION AND THERMAL DEPHASING
    # =========================================================================
    delta_V_over_V = blockade_fluctuation(spacing, sigma_r)
    gamma_phi_thermal = thermal_dephasing_rate(delta_V_over_V, V, Omega_eff)
    
    # =========================================================================
    # INTERMEDIATE STATE SCATTERING
    # =========================================================================
    from .atom_database import ATOM_DB
    
    if intermediate_state is None:
        intermediate_state = "5P3/2" if species == "Rb87" else "6P3/2"
    gamma_intermediate = ATOM_DB[species]["intermediate_states"][intermediate_state]["linewidth"]
    
    if Omega_1 > 0 and Delta_e > 0:
        from .noise_models import intermediate_state_scattering_rate
        gamma_scatter = intermediate_state_scattering_rate(Omega_1, Delta_e, gamma_intermediate)
    else:
        gamma_scatter = 0.0
    
    # =========================================================================
    # MAGIC WAVELENGTH ANALYSIS
    # =========================================================================
    # Get polarizabilities at the specified trap wavelength
    alpha_ground = get_polarizability_at_wavelength(species, "ground", tweezer_wavelength_nm)
    alpha_rydberg = get_polarizability_at_wavelength(species, "rydberg", tweezer_wavelength_nm,
                                                      n_rydberg=n_rydberg)
    
    # Polarizability ratio for anti-trapping
    alpha_ratio = np.abs(alpha_rydberg / alpha_ground) if np.abs(alpha_ground) > 1e-50 else 0
    
    # Anti-trapping loss rate (using the magic wavelength analysis)
    # This is the physics-correct way: use actual alpha_ratio from wavelength
    if alpha_ratio > 0 and gate_time > 0:
        gamma_loss_antitrap = effective_loss_rate(
            gate_time, U0, alpha_ratio, 
            atom["mass"], tweezer_waist, temperature
        )
    else:
        gamma_loss_antitrap = 0.0
    
    # Differential light shift (at trap center)
    from .constants import EPS0, C, HBAR
    if np.abs(alpha_ground) > 1e-50:
        I_center = 2 * EPS0 * C * np.abs(U0) / np.abs(alpha_ground)
    else:
        I_center = 0
    delta_alpha = np.abs(alpha_rydberg - alpha_ground)
    differential_shift = delta_alpha * I_center / (2 * EPS0 * C * HBAR * 2 * np.pi)
    
    # Magic enhancement factor
    enhancement = magic_trap_enhancement(species, tweezer_wavelength_nm, n_rydberg)
    
    # =========================================================================
    # DOPPLER DEPHASING FROM THERMAL MOTION
    # =========================================================================
    # Calculate effective k-vector for two-photon transition
    if include_doppler and gate_time > 0:
        lambda_1 = rydberg_wavelength_1_nm * 1e-9
        lambda_2 = rydberg_wavelength_2_nm * 1e-9
        k_1 = 2 * np.pi / lambda_1
        k_2 = 2 * np.pi / lambda_2
        
        if counter_propagating:
            # Counter-propagating: k_eff = |k_1 - k_2| (smaller Doppler)
            k_eff = np.abs(k_1 - k_2)
        else:
            # Co-propagating: k_eff = k_1 + k_2 (larger Doppler)
            k_eff = k_1 + k_2
        
        gamma_doppler = doppler_dephasing_rate(temperature, atom["mass"], k_eff, gate_time)
    else:
        gamma_doppler = 0.0
        k_eff = 0.0
    
    # =========================================================================
    # INTENSITY NOISE DEPHASING
    # =========================================================================
    if include_intensity_noise and intensity_noise_frac > 0:
        # Use the magic wavelength enhancement factor as the differential
        # Stark shift fraction - this accounts for how well the trap
        # is tuned to the magic wavelength. 'enhancement' is computed above.
        diff_stark_frac = min(enhancement, 0.1)  # Cap at 10%
        gamma_intensity = intensity_noise_dephasing_rate(
            U0, intensity_noise_frac, gate_time, diff_stark_frac
        )
    else:
        gamma_intensity = 0.0
    
    # =========================================================================
    # RETURN UNIFIED NOISE DICTIONARY
    # =========================================================================
    # Compute Rydberg decay rate from lifetime (room temp BBR included)
    tau_ryd = get_rydberg_lifetime(n_rydberg, species, temperature=300.0)
    gamma_r = 1.0 / tau_ryd
    
    return {
        # Trap parameters
        'trap_depth_uK': U0 / KB * 1e6,
        'trap_freq_radial_kHz': omega_r / (2*np.pi) / 1e3,
        'position_uncertainty_nm': sigma_r * 1e9,
        'V_over_2pi_MHz': V / (2*np.pi) / 1e6,
        
        # Core noise rates (Hz)
        'gamma_r': gamma_r,
        'gamma_scatter_intermediate': gamma_scatter,
        'gamma_phi_laser': gamma_phi_laser,
        'gamma_phi_thermal': gamma_phi_thermal,
        'gamma_phi_doppler': gamma_doppler,      # NEW: Doppler dephasing
        'gamma_phi_intensity': gamma_intensity,  # NEW: Intensity noise dephasing
        'gamma_loss_antitrap': gamma_loss_antitrap,  # From magic analysis
        'gamma_loss_background': 1e3,  # Default UHV assumption
        
        # Blockade fluctuations
        'blockade_fluctuation_percent': delta_V_over_V * 100,
        'intermediate_linewidth_MHz': gamma_intermediate / (2*np.pi) / 1e6,
        
        # Doppler parameters (for diagnostics)
        'k_eff_rad_per_m': k_eff,
        'v_thermal_m_per_s': np.sqrt(KB * temperature / atom["mass"]),
        
        # Magic wavelength analysis
        'alpha_ratio': alpha_ratio,
        'alpha_ground_au': alpha_ground / (4 * np.pi * EPS0 * (0.529e-10)**3),  # Convert to a.u.
        'alpha_rydberg_au': alpha_rydberg / (4 * np.pi * EPS0 * (0.529e-10)**3),
        'differential_shift_Hz': differential_shift,
        'magic_enhancement': enhancement,
        'wavelength_nm': tweezer_wavelength_nm,
    }


def calculate_zeeman_shift(
    B_field: float,
    qubit_0: Tuple[int, int],
    qubit_1: Tuple[int, int],
    species: str
) -> float:
    """
    Calculate differential Zeeman shift between qubit states.
    
    For alkali atoms, the Zeeman shift depends on whether we're using
    clock states (mF=0) or stretched states (mF=±F).
    
    **Clock states** (mF=0 for both states):
    - First-order (linear) Zeeman CANCELS for both |F, mF=0⟩ states
    - Only quadratic Zeeman shift remains: δ_Z ∝ B²
    - Typical: ~100 Hz/(Gauss)² for Rb87
    
    **Non-clock states** (mF≠0):
    - Linear Zeeman shift: δ_Z = g_F μ_B B ΔmF / ℏ
    - Much larger than clock states!
    - Typical: ~700 kHz/Gauss for Rb87
    
    Parameters
    ----------
    B_field : float
        Magnetic field strength (Tesla)
    qubit_0 : tuple (F, mF)
        Quantum numbers for |0⟩ state
    qubit_1 : tuple (F, mF)
        Quantum numbers for |1⟩ state
    species : str
        Atomic species ("Rb87" or "Cs133")
        
    Returns
    -------
    float
        Differential Zeeman shift δ_Z (rad/s)
        This is the coherent shift to add to detuning.
    
    Notes
    -----
    This function returns the COHERENT shift that modifies the effective
    detuning in the Hamiltonian. The INCOHERENT dephasing from B-field
    fluctuations is handled separately by gamma_phi_zeeman in the noise model.
    
    Examples
    --------
    >>> # Clock states: very small shift
    >>> delta_z = calculate_zeeman_shift(1e-4, (1,0), (2,0), "Rb87")
    >>> print(f"Quadratic Zeeman: {delta_z/(2*np.pi)/1e3:.2f} kHz")
    
    >>> # Stretched states: large linear shift
    >>> delta_z = calculate_zeeman_shift(1e-4, (1,1), (2,2), "Rb87")
    >>> print(f"Linear Zeeman: {delta_z/(2*np.pi)/1e6:.2f} MHz")
    """
    from .constants import MU_B, HBAR
    
    F0, mF0 = qubit_0
    F1, mF1 = qubit_1
    Delta_mF = mF1 - mF0
    
    # Check if clock states (both mF=0)
    is_clock_state = (mF0 == 0 and mF1 == 0)
    
    if is_clock_state:
        # Clock states: only quadratic Zeeman shift
        # δ_Z = K_quad × B² where K_quad is species-dependent
        # For Rb87: K_quad ≈ 575 Hz/G² for |F=1⟩→|F=2⟩ transition
        # For Cs133: K_quad ≈ 2 kHz/G² 
        
        B_gauss = B_field * 1e4  # Convert Tesla → Gauss
        
        if species == "Rb87":
            K_quad = 575.0  # Hz/G² (differential between F=1 and F=2)
        elif species == "Cs133":
            K_quad = 2000.0  # Hz/G²
        else:
            # Default approximation
            K_quad = 1000.0  # Hz/G²
        
        delta_z_Hz = K_quad * B_gauss**2
        delta_z = delta_z_Hz * 2 * np.pi  # Convert to rad/s
        
    else:
        # Non-clock states: linear Zeeman shift
        # δ_Z = g_F μ_B B ΔmF / ℏ
        
        # Landé g-factor (approximate)
        if species == "Rb87":
            if F0 == 1:
                g_F0 = -0.5  # F=1 ground state
            else:
                g_F0 = 0.5   # F=2 ground state
            if F1 == 1:
                g_F1 = -0.5
            else:
                g_F1 = 0.5
        elif species == "Cs133":
            if F0 == 3:
                g_F0 = -0.25  # F=3 ground state
            else:
                g_F0 = 0.25   # F=4 ground state
            if F1 == 3:
                g_F1 = -0.25
            else:
                g_F1 = 0.25
        else:
            # Default approximation
            g_F0 = 0.5 if F0 == 2 else -0.5
            g_F1 = 0.5 if F1 == 2 else -0.5
        
        # Differential shift
        delta_z = (g_F1 * mF1 - g_F0 * mF0) * MU_B * B_field / HBAR
    
    return delta_z


def calculate_stark_shift(
    tweezer_power: float,
    tweezer_waist: float,
    tweezer_wavelength: float,
    alpha_ground: float,
    alpha_excited: float
) -> float:
    """
    Calculate differential AC Stark shift from trap laser.
    
    The optical tweezer creates an AC Stark shift (light shift) that
    depends on the atomic polarizability α:
    
        U(r) = -(α/2) × |E(r)|²
    
    For a Gaussian beam:
        I(r) = I₀ exp(-2r²/w²)
        
    At trap center (r=0):
        δ_stark = Δα × I₀ / (2ε₀ c ℏ × 2π)
        
    where Δα = |α_excited - α_ground|.
    
    **Magic wavelength**: If Δα ≈ 0, then δ_stark ≈ 0!
    This eliminates differential light shifts, improving gate fidelity.
    
    Parameters
    ----------
    tweezer_power : float
        Tweezer laser power (W)
    tweezer_waist : float
        Gaussian beam waist (m)
    tweezer_wavelength : float
        Trap laser wavelength (m)
    alpha_ground : float
        Ground state polarizability (SI units: C²m²/J)
    alpha_excited : float
        Excited (Rydberg) state polarizability (SI units)
        
    Returns
    -------
    float
        Differential AC Stark shift δ_stark (rad/s)
        This is the coherent shift to add to detuning.
        
    Notes
    -----
    - This is the COHERENT shift that modifies effective detuning
    - The INCOHERENT dephasing from trap intensity noise is separate
    - For magic wavelength traps, this should be ~0
    
    Examples
    --------
    >>> # Standard 1064 nm trap (NOT magic)
    >>> P, w = 10e-3, 1e-6  # 10 mW, 1 μm waist
    >>> alpha_g = 5.3e-39   # Rb87 ground state (SI)
    >>> alpha_r = -1e-37    # Rb87 70S Rydberg state (negative!)
    >>> delta_s = calculate_stark_shift(P, w, 1064e-9, alpha_g, alpha_r)
    >>> print(f"Stark shift: {delta_s/(2*np.pi)/1e3:.1f} kHz")
    
    >>> # Magic wavelength ~880 nm: Δα ≈ 0
    >>> alpha_r_magic = 5.3e-39  # Matches ground state
    >>> delta_s = calculate_stark_shift(P, w, 880e-9, alpha_g, alpha_r_magic)
    >>> print(f"Magic trap shift: {delta_s/(2*np.pi):.1f} Hz")  # Should be ~0
    """
    from .constants import EPS0, C, HBAR
    
    # Intensity at beam center for Gaussian beam
    # I₀ = 2P / (π w²)
    I_center = 2 * tweezer_power / (np.pi * tweezer_waist**2)
    
    # Differential polarizability
    delta_alpha = np.abs(alpha_excited - alpha_ground)
    
    # Differential AC Stark shift (in Hz, then convert to rad/s)
    # δ_stark = Δα × I / (4πε₀ c ℏ)
    delta_stark_Hz = delta_alpha * I_center / (4 * np.pi * EPS0 * C * HBAR)
    delta_stark = delta_stark_Hz * 2 * np.pi  # Convert to rad/s
    
    return delta_stark


def calculate_qubit_stark_shift(
    tweezer_power: float,
    tweezer_waist: float,
    species: str = "Rb87",
    trap_depth_mK: float = None,
) -> float:
    """
    Calculate differential AC Stark shift between qubit hyperfine states.
    
    For neutral atom qubits encoded in hyperfine ground states, the relevant
    Stark shift is the DIFFERENTIAL shift between the two qubit levels
    (e.g., F=1 vs F=2 in Rb87), NOT the ground-to-Rydberg shift.
    
    Physics
    -------
    The hyperfine states |F=1⟩ and |F=2⟩ have slightly different polarizabilities
    due to the hyperfine interaction modifying the optical response:
    
        Δα_hyperfine = α(F=2) - α(F=1) ≈ 2.4 a.u. for Rb87 at 1064 nm
    
    This is only ~0.35% of the total ground state polarizability (~687 a.u.),
    but it causes a differential shift between the qubit states.
    
    At 1 mK trap depth, this gives:
        δf_qubit ≈ 70 kHz for Rb87
    
    This shift is typically much smaller than the qubit frequency (~6.8 GHz)
    and can be calibrated out, but it contributes to dephasing if the trap
    intensity fluctuates.
    
    Parameters
    ----------
    tweezer_power : float
        Tweezer laser power (W)
    tweezer_waist : float
        Gaussian beam waist (m)
    species : str
        Atomic species ("Rb87" or "Cs133")
    trap_depth_mK : float, optional
        If provided, use this trap depth instead of computing from power/waist.
        Useful when trap depth is known from other calibration.
        
    Returns
    -------
    float
        Qubit state differential Stark shift (rad/s)
        
    Notes
    -----
    Typical values for Rb87 at 1064 nm:
    - α(F=1) ≈ 686.1 a.u.
    - α(F=2) ≈ 688.5 a.u.  
    - Δα ≈ 2.4 a.u. (~0.35% differential)
    - At 1 mK trap depth: δf ≈ 70 kHz
    
    For Cs133 the differential is larger (~1%) due to stronger hyperfine coupling.
    
    References
    ----------
    - Arora et al., PRA 76, 052509 (2007) - Hyperfine polarizabilities
    - Safronova et al., PRA 73, 022505 (2006) - Magic wavelengths for Rb
    """
    from .constants import EPS0, C, HBAR, KB
    
    # Hyperfine differential polarizabilities (in atomic units)
    # These come from detailed atomic structure calculations
    # Reference: Arora et al., PRA 76, 052509 (2007)
    HYPERFINE_DIFFERENTIAL_AU = {
        "Rb87": 2.4,    # α(F=2) - α(F=1) at 1064 nm
        "Cs133": 7.0,   # Larger due to stronger hyperfine coupling
    }
    
    # Conversion factor: atomic units to SI (C²m²/J)
    AU_TO_SI = 1.6488e-41
    
    delta_alpha_au = HYPERFINE_DIFFERENTIAL_AU.get(species, 2.4)
    delta_alpha_si = delta_alpha_au * AU_TO_SI
    
    if trap_depth_mK is not None:
        # Use empirical relation: δf ≈ 70 kHz × (U₀/1mK) for Rb87
        # This is calibrated against detailed calculations
        scaling_factor = {"Rb87": 70e3, "Cs133": 200e3}  # Hz per mK
        delta_stark_Hz = scaling_factor.get(species, 70e3) * trap_depth_mK
    else:
        # Calculate from first principles
        # Intensity at beam center: I₀ = 2P / (π w²)
        I_center = 2 * tweezer_power / (np.pi * tweezer_waist**2)
        
        # Differential AC Stark shift
        # δf = Δα × I / (4πε₀ c ℏ)
        delta_stark_Hz = delta_alpha_si * I_center / (4 * np.pi * EPS0 * C * HBAR)
    
    return delta_stark_Hz * 2 * np.pi  # Convert to rad/s


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Tweezer geometry
    "tweezer_spacing",
    "diffraction_limited_spot",
    
    # Trap depth
    "trap_depth",
    "trap_depth_from_temperature",
    "trap_temperature",
    
    # Trap frequencies
    "trap_frequencies",
    "trap_frequency_from_depth_and_waist",
    
    # Position uncertainty
    "position_uncertainty",
    "quantum_ground_state_size",
    "thermal_de_broglie_wavelength",
    "thermal_velocity",
    
    # Anti-trapping
    "anti_trap_potential",
    "anti_trap_frequency",
    "atom_loss_probability",
    "effective_loss_rate",
    
    # Blockade fluctuations
    "blockade_fluctuation",
    "thermal_dephasing_rate",
    "thermal_infidelity_estimate",
    
    # Motional heating
    "photon_recoil_energy",
    "recoil_temperature",
    "trap_heating_rate_intensity_noise",
    "trap_heating_rate_pointing_noise",
    
    # Convenience
    "compute_trap_properties",
    "compute_trap_dependent_noise",
    "calculate_zeeman_shift",
    "calculate_stark_shift",
    "calculate_qubit_stark_shift",
]
