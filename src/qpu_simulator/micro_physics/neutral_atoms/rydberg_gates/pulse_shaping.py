"""
Pulse Shaping for Rydberg Gate Simulation
=========================================

This module provides pulse envelope functions and spectral analysis tools
for CZ gate simulations. Proper pulse shaping is crucial for reducing
errors from spectral leakage to unwanted states.

Why Pulse Shaping Matters
-------------------------
When we turn a laser on/off to drive Rydberg transitions, the temporal
shape of the pulse determines its spectral content. The Fourier transform
of the pulse envelope gives the frequency spread:

    Square pulse:  Sinc spectrum → slow decay → leakage to nearby states
    Gaussian:      Gaussian spectrum → exponential decay → less leakage
    Blackman:      Very narrow spectrum → minimal leakage

Spectral Leakage
----------------
The laser driving |1⟩ → |r⟩ can also off-resonantly excite nearby states:

    |1⟩ → |r±1⟩  (adjacent Rydberg states, ~10 GHz away)
    |1⟩ → |r,mJ⟩ (Zeeman sublevels, ~MHz away)
    
The leakage rate scales as:

    γ_leak ∝ Ω² × S(Δ_leak)
    
where S(Δ) is the spectral power at detuning Δ from the target transition.

Pulse Shape Comparison
----------------------
For a 1 μs pulse with Δ_leak = 2π × 50 MHz (fine structure):

    Shape       Spectral Power S(Δ)    Relative Leakage
    ─────────────────────────────────────────────────────
    Square      ~10⁻²                  1.0× (worst)
    Gaussian    ~10⁻⁴                  0.01× 
    Cosine      ~10⁻⁵                  0.001×
    Blackman    ~10⁻⁶                  0.0001× (best)
    DRAG        ~10⁻⁷                  0.00001× (with tuning)

DRAG (Derivative Removal by Adiabatic Gate) Pulses
--------------------------------------------------
DRAG adds a quadrature component to cancel leakage:

    Ω(t) = Ω_base(t) + i × λ × dΩ_base/dt / Δ_leak

The imaginary component creates destructive interference that suppresses
off-resonant excitation. Originally developed for superconducting qubits
(Motzoi et al., PRL 2009), it's also effective for Rydberg gates.

Area Preservation
-----------------
For shaped pulses, the integrated pulse area must match the square pulse
to achieve the same rotation angle:

    ∫ Ω(t) dt = Ω_eff × τ   (where Ω_eff τ = π for π-pulse)

Shaped pulses are automatically normalized in this module to preserve area.

References
----------
- Motzoi et al., PRL 103, 110501 (2009) - Original DRAG
- Gambetta et al., PRA 83, 012308 (2011) - DRAG theory
- de Léséleuc thesis (2019) - Rydberg pulse shaping
- Bluvstein thesis (2024) - High-fidelity Rydberg gates

Author: Quantum Simulation Team
"""

from typing import Callable, Dict, Union, Tuple, Optional
import numpy as np

from .constants import HBAR, MU_B, A0, RY_JOULES


# =============================================================================
# PULSE ENVELOPE FUNCTIONS
# =============================================================================

def pulse_envelope_square(
    t: np.ndarray,
    tau: float,
    **kwargs,
) -> np.ndarray:
    """
    Square (rectangular) pulse envelope: constant amplitude.
    
    Ω(t) = Ω₀  for 0 ≤ t ≤ τ
    
    Parameters
    ----------
    t : np.ndarray
        Time array (seconds)
    tau : float
        Total pulse duration (seconds)
    **kwargs
        Ignored (for API consistency)
        
    Returns
    -------
    np.ndarray
        Envelope values (all ones)
        
    Physics Notes
    -------------
    **Advantages:**
    - Simple to implement
    - Maximum peak power efficiency
    - Well-understood theoretically
    
    **Disadvantages:**
    - Sharp edges create broad spectral content
    - Sinc spectrum: S(Δ) ∝ sinc²(Δτ/2)
    - Significant leakage to nearby states
    - Sensitive to finite rise/fall times
    
    **Spectral width:**
    The first null is at Δ = 2π/τ. For τ = 1 μs, this is 1 MHz.
    Power at Δ = 50 MHz is S(50) ≈ 10⁻² (only 100× suppression).
    """
    return np.ones_like(t, dtype=float)


def pulse_envelope_gaussian(
    t: np.ndarray,
    tau: float,
    sigma_factor: float = 3.0,
    **kwargs,
) -> np.ndarray:
    """
    Gaussian pulse envelope.
    
    Ω(t) = exp(-(t - τ/2)² / (2σ²))
    
    where σ = τ / sigma_factor.
    
    Parameters
    ----------
    t : np.ndarray
        Time array (seconds)
    tau : float
        Total pulse duration (seconds)
    sigma_factor : float
        Ratio τ/σ. Default 3.0 means σ = τ/3.
        Larger values = narrower Gaussian = more truncation.
    **kwargs
        Ignored
        
    Returns
    -------
    np.ndarray
        Normalized Gaussian envelope (peak = 1)
        
    Physics Notes
    -------------
    **Advantages:**
    - Smooth turn-on/off
    - Gaussian spectrum (also Gaussian)
    - Exponential suppression of leakage
    
    **Disadvantages:**
    - Lower peak power efficiency (~72% area vs square)
    - Truncation at edges adds ripples to spectrum
    - sigma_factor tradeoff: larger = narrower spectrum but more truncation
    
    **Optimal sigma_factor:**
    - σ = τ/3: Truncated at ±3σ, ~0.1% truncation error
    - σ = τ/4: Truncated at ±4σ, ~0.001% truncation error
    - σ = τ/6: Wide Gaussian, but spectral width increases
    
    **Spectral width:**
    Gaussian spectrum: S(Δ) ∝ exp(-(Δσ)²)
    For σ = 0.3 μs: S(50 MHz) ≈ exp(-900) ≈ 10⁻³⁹⁰ (essentially zero)
    Truncation dominates actual leakage.
    """
    sigma = tau / sigma_factor
    t_center = tau / 2
    envelope = np.exp(-(t - t_center)**2 / (2 * sigma**2))
    
    # Normalize to peak = 1
    max_val = envelope.max()
    if max_val > 0:
        envelope = envelope / max_val
    
    return envelope


def pulse_envelope_cosine(
    t: np.ndarray,
    tau: float,
    **kwargs,
) -> np.ndarray:
    """
    Cosine (raised cosine / Hann window) pulse envelope.
    
    Ω(t) = sin²(πt/τ) = (1 - cos(2πt/τ)) / 2
    
    Parameters
    ----------
    t : np.ndarray
        Time array (seconds)
    tau : float
        Total pulse duration (seconds)
    **kwargs
        Ignored
        
    Returns
    -------
    np.ndarray
        Cosine envelope (peak = 1)
        
    Physics Notes
    -------------
    **Advantages:**
    - Exactly zero at t=0 and t=τ (no truncation artifacts)
    - Smooth first derivative at edges
    - Narrow main lobe in spectrum
    
    **Disadvantages:**
    - Area = τ/2 (half of square pulse)
    - Must double peak Rabi frequency to maintain rotation angle
    
    **Spectral characteristics:**
    - Main lobe width: 4/τ (2× wider than square)
    - First sidelobe: -23 dB below peak
    - Sidelobes fall as 1/Δ² (faster than sinc's 1/Δ)
    
    **Why sin²?**
    The sin² shape is a "raised cosine" that smoothly interpolates
    from 0 to 1 and back. It's equivalent to a Hann window in signal
    processing and is widely used for its excellent spectral properties.
    """
    return np.sin(np.pi * t / tau)**2


def pulse_envelope_blackman(
    t: np.ndarray,
    tau: float,
    **kwargs,
) -> np.ndarray:
    """
    Blackman window pulse envelope.
    
    Ω(t) = 0.42 - 0.5×cos(2πt/τ) + 0.08×cos(4πt/τ)
    
    Parameters
    ----------
    t : np.ndarray
        Time array (seconds)
    tau : float
        Total pulse duration (seconds)
    **kwargs
        Ignored
        
    Returns
    -------
    np.ndarray
        Normalized Blackman envelope (peak = 1)
        
    Physics Notes
    -------------
    **Advantages:**
    - Best sidelobe suppression of standard windows
    - First sidelobe: -58 dB below peak
    - Sidelobes fall as 1/Δ⁶
    - Zero at both edges and their first derivatives
    
    **Disadvantages:**
    - Lower area efficiency (~42% of square)
    - Wider main lobe than Hann
    - More complex waveform generation
    
    **Why Blackman for Rydberg gates?**
    When leakage to adjacent Rydberg states (n±1, or fine structure
    components) is the dominant error, Blackman's extreme sidelobe
    suppression provides the best performance. The ~58 dB suppression
    means leakage is reduced by ~1000× compared to square pulses.
    
    **Coefficients:**
    The exact Blackman window has a₀=0.42, a₁=0.5, a₂=0.08.
    These are chosen to null the 1st and 2nd sidelobes analytically.
    """
    envelope = (0.42 
                - 0.5 * np.cos(2 * np.pi * t / tau) 
                + 0.08 * np.cos(4 * np.pi * t / tau))
    
    # Normalize to peak = 1
    max_val = envelope.max()
    if max_val > 0:
        envelope = envelope / max_val
    
    return envelope


def pulse_envelope_drag(
    t: np.ndarray,
    tau: float,
    Delta_leak: float,
    lambda_drag: float = 1.0,
    base_shape: str = "gaussian",
    sigma_factor: float = 4.0,
    **kwargs,
) -> np.ndarray:
    """
    DRAG (Derivative Removal by Adiabatic Gate) pulse envelope.
    
    Ω(t) = Ω_base(t) + i × λ × dΩ_base/dt / Δ_leak
    
    The quadrature (imaginary) component cancels leakage to nearby states.
    
    Parameters
    ----------
    t : np.ndarray
        Time array (seconds)
    tau : float
        Total pulse duration (seconds)
    Delta_leak : float
        Detuning to the leakage state (rad/s).
        Typically the fine structure splitting (~2π × 50 MHz).
    lambda_drag : float
        DRAG parameter (dimensionless). Default 1.0.
        - λ = 1: Full first-order DRAG correction
        - λ < 1: Reduced correction (more robust)
        - λ > 1: Over-correction (rarely needed)
    base_shape : str
        Base envelope shape: "gaussian", "cosine", or "blackman"
    sigma_factor : float
        For Gaussian base: ratio τ/σ
    **kwargs
        Passed to base envelope function
        
    Returns
    -------
    np.ndarray
        Complex envelope: real = base, imag = DRAG correction
        
    Physics Notes
    -------------
    **How DRAG works:**
    Consider driving |1⟩ → |r⟩ with a leakage state |r'⟩ at detuning Δ.
    The off-resonant excitation to |r'⟩ is:
    
        a_r'(t) ∝ ∫ Ω(t') e^{iΔt'} dt' = FT[Ω](Δ)
    
    DRAG adds a quadrature component that destructively interferes:
    
        Ω_DRAG = Ω_base + i × (λ/Δ) × dΩ_base/dt
    
    The derivative term has a 90° phase shift, and when combined with
    the e^{iΔt} oscillation, it cancels the leakage amplitude.
    
    **Optimal lambda:**
    First-order perturbation theory gives λ = 1. However, higher-order
    effects and finite pulse duration can modify the optimum:
    - λ = 0.8-1.2 typically optimal in practice
    - Calibrate experimentally for best results
    
    **Limitations:**
    - Only works for ONE leakage state at a time
    - Multiple leakage states need multi-tone DRAG
    - Assumes weak driving (Ω << Δ)
    - Higher-order effects at strong driving
    
    **References:**
    - Motzoi et al., PRL 103, 110501 (2009)
    - Gambetta et al., PRA 83, 012308 (2011)
    """
    # Get base envelope
    if base_shape == "gaussian":
        base = pulse_envelope_gaussian(t, tau, sigma_factor=sigma_factor)
    elif base_shape == "cosine":
        base = pulse_envelope_cosine(t, tau)
    elif base_shape == "blackman":
        base = pulse_envelope_blackman(t, tau)
    else:
        raise ValueError(f"Unknown base_shape: {base_shape}. "
                        f"Use 'gaussian', 'cosine', or 'blackman'.")
    
    # Compute derivative numerically
    dt = t[1] - t[0] if len(t) > 1 else tau / 100
    d_base = np.gradient(base, dt)
    
    # DRAG correction: imaginary quadrature
    drag_correction = lambda_drag / np.abs(Delta_leak) * d_base
    
    # Return complex envelope
    return base + 1j * drag_correction


# =============================================================================
# PULSE SHAPE REGISTRY
# =============================================================================

PULSE_SHAPES: Dict[str, Callable] = {
    "square": pulse_envelope_square,
    "gaussian": pulse_envelope_gaussian,
    "cosine": pulse_envelope_cosine,
    "blackman": pulse_envelope_blackman,
    "drag": pulse_envelope_drag,
}
"""Registry of available pulse envelope functions."""


def get_pulse_envelope(
    shape: str,
    t: np.ndarray,
    tau: float,
    **kwargs,
) -> np.ndarray:
    """
    Get pulse envelope by name.
    
    This is the main interface for pulse shaping. Use this function
    rather than calling individual envelope functions directly.
    
    Parameters
    ----------
    shape : str
        Pulse shape name: "square", "gaussian", "cosine", "blackman", "drag"
    t : np.ndarray
        Time array (seconds)
    tau : float
        Total pulse duration (seconds)
    **kwargs
        Shape-specific parameters:
        - gaussian: sigma_factor (default 3.0)
        - drag: Delta_leak, lambda_drag, base_shape
        
    Returns
    -------
    np.ndarray
        Envelope values (may be complex for DRAG)
        
    Raises
    ------
    ValueError
        If shape is not recognized
        
    Examples
    --------
    >>> t = np.linspace(0, 1e-6, 100)
    >>> tau = 1e-6
    >>> 
    >>> # Square pulse
    >>> env_sq = get_pulse_envelope("square", t, tau)
    >>> 
    >>> # Gaussian with custom width
    >>> env_gauss = get_pulse_envelope("gaussian", t, tau, sigma_factor=4.0)
    >>> 
    >>> # DRAG with leakage detuning
    >>> Delta_leak = 2 * np.pi * 50e6  # 50 MHz
    >>> env_drag = get_pulse_envelope("drag", t, tau, Delta_leak=Delta_leak)
    """
    shape_lower = shape.lower()
    
    if shape_lower not in PULSE_SHAPES:
        available = list(PULSE_SHAPES.keys())
        raise ValueError(
            f"Unknown pulse shape: {shape}. "
            f"Available shapes: {available}"
        )
    
    return PULSE_SHAPES[shape_lower](t, tau, **kwargs)


def list_available_shapes() -> list:
    """Return list of available pulse shape names."""
    return list(PULSE_SHAPES.keys())


# =============================================================================
# SPECTRAL ANALYSIS
# =============================================================================

def spectral_leakage_factor(
    pulse_shape: str,
    tau: float,
    Delta_leak: float,
) -> float:
    """
    Compute spectral leakage factor S(Δ) for a pulse shape.
    
    The leakage factor is the normalized spectral power at the leakage
    detuning, relative to the peak (DC) power.
    
    Parameters
    ----------
    pulse_shape : str
        Pulse shape name
    tau : float
        Pulse duration (seconds)
    Delta_leak : float
        Detuning to leakage state (rad/s)
        
    Returns
    -------
    float
        Spectral leakage factor S(Δ) in range [0, 1].
        Lower is better (less leakage).
        
    Physics Notes
    -------------
    The spectral power at frequency Δ is |FT[Ω](Δ)|² normalized to
    |FT[Ω](0)|². For continuous pulses:
    
    **Square:**
        S(Δ) = sinc²(Δτ/2π) = [sin(πx)/(πx)]² where x = Δτ/(2π)
    
    **Gaussian:**
        S(Δ) = exp(-(Δσ)²) where σ = τ/sigma_factor
        Truncation modifies this in practice.
    
    **Cosine (Hann):**
        S(Δ) ∝ [sinc(x) / (1-x²)]² with sidelobes at -23 dB
    
    **Blackman:**
        S(Δ) → extremely small, ~exp(-3|x|) × sinc²(x)
        First sidelobe at -58 dB
    
    **DRAG:**
        Ideally zero at the designed Δ_leak
        Residual from finite pulse, ~10% of Gaussian
    """
    # Dimensionless detuning
    x = Delta_leak * tau / (2 * np.pi)
    
    # Handle x ≈ 0 (on resonance)
    if abs(x) < 1e-10:
        return 1.0
    
    if pulse_shape == "square":
        # Sinc² spectrum
        S = (np.sin(np.pi * x) / (np.pi * x))**2
        
    elif pulse_shape == "gaussian":
        # Gaussian spectrum (approximate, ignoring truncation)
        # For sigma_factor=3, σ = τ/3
        sigma_factor = 3.0
        sigma = tau / sigma_factor
        S = np.exp(-(Delta_leak * sigma)**2)
        
    elif pulse_shape == "cosine":
        # Hann window spectrum
        if abs(x - 0.5) < 1e-10 or abs(x + 0.5) < 1e-10:
            S = 0.25  # Special case at x = ±0.5
        else:
            sinc_term = np.sin(np.pi * x) / (np.pi * x)
            S = (sinc_term / (1 - x**2))**2
            
    elif pulse_shape == "blackman":
        # Blackman has very steep sidelobe decay
        # Approximate: exponential decay × sinc²
        S_exp = np.exp(-3 * abs(x))
        S_sinc = (np.sin(np.pi * x) / (np.pi * x))**2 if abs(x) > 1e-10 else 1.0
        S = min(S_exp * S_sinc, S_sinc * 0.1)  # Cap at -10 dB below sinc
        
    elif pulse_shape == "drag":
        # DRAG ideally nulls leakage at designed Δ
        # Residual is ~10% of base Gaussian
        S_gauss = np.exp(-(Delta_leak * tau / (3 * 4))**2)  # sigma_factor=4 for DRAG
        S = S_gauss * 0.1
        
    else:
        # Unknown shape: use sinc as fallback
        S = (np.sin(np.pi * x) / (np.pi * x))**2 if abs(x) > 1e-10 else 1.0
    
    return float(np.clip(S, 0, 1))


def compute_leakage_detuning(
    species: str,
    n_rydberg: int,
    L: int = 0,
    leakage_target: str = "fine_structure",
    atom_database: dict = None,
) -> float:
    """
    Estimate the detuning to the nearest leakage Rydberg state.
    
    Different leakage mechanisms have different characteristic frequencies.
    This function returns the relevant detuning for leakage calculations.
    
    Parameters
    ----------
    species : str
        Atomic species ("Rb87" or "Cs133")
    n_rydberg : int
        Principal quantum number
    L : int
        Orbital angular momentum (0=S, 1=P, 2=D)
    leakage_target : str
        Type of leakage:
        - "fine_structure": nS → nP, nD (~50 MHz for n=70) - DOMINANT!
        - "adjacent_n": nS → (n±1)S (~22 GHz for n=70) - usually negligible
        - "zeeman": mJ splitting (~1.4 MHz/G)
    atom_database : dict, optional
        Atom properties database. If None, uses default quantum defects.
        
    Returns
    -------
    float
        Detuning to leakage state (rad/s)
        
    Physics Notes
    -------------
    **Fine structure (DOMINANT for pulse shaping):**
    The fine structure splitting between L=0 (S) and L=1 (P) states
    decreases as n⁻³ for high n. For Rb at n=70, this is ~50 MHz.
    This is the relevant frequency for spectral leakage calculations.
    
    **Adjacent n states:**
    The spacing between nS and (n+1)S is ~22 GHz for n=70, scaling as n⁻³.
    This is usually far enough that leakage is negligible.
    
    **Zeeman splitting:**
    For S-states (J=1/2), the mJ=±1/2 splitting is:
        Δ_Z = g_J × μ_B × B / ℏ ≈ 2.8 MHz/G
    For typical B ~ 1 G, this is ~3 MHz.
    """
    # Default quantum defects for Rb87
    delta_qd_S = 3.13  # S-state quantum defect
    delta_qd_P = 2.65  # P-state quantum defect
    
    # Override from database if provided
    if atom_database is not None and species in atom_database:
        atom = atom_database[species]
        delta_qd_S = atom.get("quantum_defects", {}).get("S", delta_qd_S)
        delta_qd_P = atom.get("quantum_defects", {}).get("P", delta_qd_P)
    
    if leakage_target == "fine_structure":
        # This is where pulse shaping matters most!
        # Use ~50 MHz as typical experimental value for fine structure
        # The exact value depends on n and species, but 50 MHz is representative
        Delta_leak_Hz = 50e6  # 50 MHz
        Delta_leak = 2 * np.pi * Delta_leak_Hz
        
    elif leakage_target == "adjacent_n":
        # Energy difference between nS and (n±1)S
        # E_n = -Ry / (n - δ)² → ΔE ≈ 2Ry / n*³
        n_star = n_rydberg - delta_qd_S
        Delta_leak = 2 * RY_JOULES / HBAR / n_star**3
        
    elif leakage_target == "zeeman":
        # Zeeman splitting at typical field
        B_typical = 1e-4  # 1 Gauss
        g_J = 2.002  # S-state g-factor
        Delta_leak = g_J * MU_B * B_typical / HBAR
        
    else:
        raise ValueError(
            f"Unknown leakage_target: {leakage_target}. "
            f"Use 'fine_structure', 'adjacent_n', or 'zeeman'."
        )
    
    return Delta_leak


def leakage_rate_to_adjacent_states(
    Omega: float,
    Delta_leak: float,
    pulse_shape: str,
    tau: float,
) -> float:
    """
    Compute effective leakage rate to adjacent states based on pulse spectrum.
    
    The leakage rate is the product of:
    1. Off-resonant Rabi frequency: Ω²/Δ_leak (perturbative coupling)
    2. Spectral leakage factor: S(Δ_leak) (frequency content at detuning)
    
    Parameters
    ----------
    Omega : float
        Peak Rabi frequency (rad/s)
    Delta_leak : float
        Detuning to leakage state (rad/s)
    pulse_shape : str
        Pulse shape name
    tau : float
        Pulse duration (seconds)
        
    Returns
    -------
    float
        Effective leakage rate γ_leak (rad/s)
        
    Physics Notes
    -------------
    **Off-resonant driving:**
    An off-resonant pulse at detuning Δ excites the leakage state with
    effective Rabi frequency:
        Ω_eff = Ω × S(Δ)^(1/2)
    
    The excitation probability per pulse is:
        P_leak ≈ (Ω_eff / Δ)²  (for small P)
    
    The effective leakage rate is:
        γ_leak = P_leak / τ = Ω² × S(Δ) / (Δ² τ)
    
    For a CZ gate with multiple pulses, total leakage accumulates.
    
    **Example:**
    Ω = 2π × 5 MHz, Δ = 2π × 50 MHz, τ = 1 μs
    
    Square: S ≈ 0.01 → γ_leak ≈ 2π × 500 Hz
    Blackman: S ≈ 10⁻⁶ → γ_leak ≈ 2π × 0.05 Hz
    
    Over a 1 μs gate: P_leak(square) ≈ 0.3%, P_leak(Blackman) ≈ 0.003%
    """
    # Spectral leakage factor
    S = spectral_leakage_factor(pulse_shape, tau, Delta_leak)
    
    # Off-resonant coupling strength
    Omega_over_Delta = Omega / Delta_leak
    
    # Effective leakage rate
    # Formula: γ = (Ω/Δ)² × Δ × S / 2
    # This gives the rate at which population leaks to the off-resonant state
    gamma_leak = 0.5 * Delta_leak * Omega_over_Delta**2 * S
    
    return gamma_leak


# =============================================================================
# PULSE NORMALIZATION AND AREA
# =============================================================================

def compute_pulse_area(
    envelope: np.ndarray,
    t: np.ndarray,
) -> float:
    """
    Compute the area (integral) of a pulse envelope.
    
    Area = ∫ |Ω(t)| dt
    
    Parameters
    ----------
    envelope : np.ndarray
        Envelope values (may be complex)
    t : np.ndarray
        Time array (seconds)
        
    Returns
    -------
    float
        Pulse area (rad for Rabi frequency envelope)
    """
    try:
        # NumPy 2.0+
        area = np.trapezoid(np.abs(envelope), t)
    except AttributeError:
        # NumPy < 2.0
        area = np.trapz(np.abs(envelope), t)
    
    return area


def normalize_pulse_area(
    envelope: np.ndarray,
    t: np.ndarray,
    target_area: float,
) -> np.ndarray:
    """
    Normalize pulse envelope to achieve target area.
    
    Used to ensure shaped pulses give the same rotation angle as square.
    
    Parameters
    ----------
    envelope : np.ndarray
        Original envelope values
    t : np.ndarray
        Time array (seconds)
    target_area : float
        Desired pulse area
        
    Returns
    -------
    np.ndarray
        Scaled envelope
    """
    current_area = compute_pulse_area(envelope, t)
    
    if current_area < 1e-15:
        return envelope  # Avoid division by zero
    
    scale_factor = target_area / current_area
    return envelope * scale_factor


def area_correction_factor(pulse_shape: str, tau: float = 1.0) -> float:
    """
    Get the area correction factor for a pulse shape.
    
    This is the ratio of square pulse area to shaped pulse area,
    used to scale Rabi frequency for equivalent rotation.
    
    Parameters
    ----------
    pulse_shape : str
        Pulse shape name
    tau : float
        Pulse duration (only relative values matter)
        
    Returns
    -------
    float
        Correction factor. Multiply peak Ω by this to match square pulse area.
        
    Examples
    --------
    >>> # For Gaussian with sigma_factor=3
    >>> factor = area_correction_factor("gaussian")
    >>> print(f"Scale Ω by {factor:.2f}x for same rotation")
    Scale Ω by 1.38x for same rotation
    """
    t = np.linspace(0, tau, 1000)
    
    # Square pulse area = tau (for unit amplitude)
    square_area = tau
    
    # Shaped pulse area
    if pulse_shape == "square":
        return 1.0
    
    # Get envelope and compute area
    # For DRAG, use default Delta_leak (won't affect magnitude much)
    kwargs = {}
    if pulse_shape == "drag":
        kwargs["Delta_leak"] = 2 * np.pi * 50e6
    
    envelope = get_pulse_envelope(pulse_shape, t, tau, **kwargs)
    shaped_area = compute_pulse_area(envelope, t)
    
    if shaped_area < 1e-15:
        return 1.0
    
    return square_area / shaped_area


# =============================================================================
# CONVENIENCE FUNCTIONS FOR SIMULATION
# =============================================================================

def prepare_pulse_for_evolution(
    pulse_shape: str,
    t_pulse: np.ndarray,
    tau: float,
    preserve_area: bool = True,
    **kwargs,
) -> Tuple[np.ndarray, dict]:
    """
    Prepare a pulse envelope for time-dependent Hamiltonian evolution.
    
    This handles:
    1. Getting the raw envelope
    2. Optional area normalization
    3. Adding numerical floor to avoid instabilities
    
    Parameters
    ----------
    pulse_shape : str
        Pulse shape name
    t_pulse : np.ndarray
        Time grid for the pulse (seconds)
    tau : float
        Pulse duration (seconds)
    preserve_area : bool
        If True, normalize envelope so area = tau (matches square pulse)
    **kwargs
        Additional parameters for specific shapes (e.g., Delta_leak for DRAG)
        
    Returns
    -------
    envelope : np.ndarray
        Processed envelope values
    info : dict
        Processing information (area_factor, peak_scaling, etc.)
    """
    # Get raw envelope
    envelope = get_pulse_envelope(pulse_shape, t_pulse, tau, **kwargs)
    
    # Area normalization
    if preserve_area:
        try:
            current_area = np.trapezoid(np.abs(envelope), t_pulse)
        except AttributeError:
            current_area = np.trapz(np.abs(envelope), t_pulse)
        
        area_factor = current_area / tau if tau > 0 else 1.0
        envelope_normalized = envelope / area_factor
    else:
        area_factor = 1.0
        envelope_normalized = envelope
    
    # Add small floor to avoid numerical issues at pulse edges
    # This prevents division by zero in time-dependent solvers
    envelope_floor = 1e-6
    envelope_safe = np.abs(envelope_normalized) + envelope_floor
    
    # Renormalize with floor included
    try:
        area_safe = np.trapezoid(envelope_safe, t_pulse)
    except AttributeError:
        area_safe = np.trapz(envelope_safe, t_pulse)
    
    if area_safe > 0:
        envelope_safe = envelope_safe * tau / area_safe
    
    info = {
        "pulse_shape": pulse_shape,
        "area_factor": area_factor,
        "peak_scaling": np.max(np.abs(envelope_normalized)),
        "envelope_floor": envelope_floor,
        "is_complex": np.any(np.iscomplex(envelope)),
    }
    
    return np.real(envelope_safe), info


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Envelope functions
    "pulse_envelope_square",
    "pulse_envelope_gaussian",
    "pulse_envelope_cosine",
    "pulse_envelope_blackman",
    "pulse_envelope_drag",
    
    # Main interface
    "PULSE_SHAPES",
    "get_pulse_envelope",
    "list_available_shapes",
    
    # Spectral analysis
    "spectral_leakage_factor",
    "compute_leakage_detuning",
    "leakage_rate_to_adjacent_states",
    
    # Pulse area
    "compute_pulse_area",
    "normalize_pulse_area",
    "area_correction_factor",
    
    # Simulation helpers
    "prepare_pulse_for_evolution",
]
