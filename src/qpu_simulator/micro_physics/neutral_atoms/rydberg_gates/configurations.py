"""
Configuration Dataclasses for Rydberg Gate Simulations
=======================================================

This module defines dataclasses that organize simulation parameters into
logical groups. These act like "configuration files" for different parts
of the simulation.

WHY USE DATACLASSES?
--------------------

1. **Clarity**: Instead of passing 20+ arguments to functions, we pass a
   few structured objects. Compare:
   
   BAD:
   ```python
   simulate(power, waist, wavelength, temp, B_field, F0, mF0, F1, mF1, n, ...)
   ```
   
   GOOD:
   ```python
   simulate(config, tweezer, environment)
   ```

2. **Documentation**: Each parameter has a clear name and docstring.

3. **Defaults**: Sensible defaults are provided for typical experiments.

4. **Type checking**: The type hints help catch errors early.

CONFIGURATION HIERARCHY
-----------------------

There are two levels of configuration:

1. **Component configs** (building blocks):
   - LaserParameters: power, waist, polarization for a single laser
   - TweezerParameters: trap laser power, waist, wavelength
   - EnvironmentParameters: temperature, B-field, atom spacing
   - AtomicConfiguration: species, Rydberg state, qubit encoding

2. **Protocol-specific simulation inputs** (what simulate_CZ_gate needs):
   - LPSimulationInputs: Levine-Pichler protocol parameters
   - JPSimulationInputs: Jandura-Pupillo bang-bang protocol parameters
   - SmoothJPSimulationInputs: Smooth sinusoidal JP protocol parameters

PRESET CONFIGURATIONS
---------------------

- `get_standard_rb87_config()`: Standard Rb87 with clock states
- `get_standard_cs133_config()`: Standard Cs133 with clock states

References
----------
[1] Bluvstein et al., PhD Thesis, Harvard (2024) - Table 2.14
[2] Levine et al., Phys. Rev. Lett. 123, 170503 (2019)
[3] Jandura & Pupillo, Phys. Rev. X 12, 011049 (2022)
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List
import numpy as np

from .constants import HBAR, KB, MU_B, E_CHARGE, A0, EPS0, C
from .atom_database import (
    ATOM_DB, get_atom_properties, effective_n, get_quantum_defect,
    get_rydberg_energy, get_C6, get_rydberg_lifetime, get_rydberg_polarizability,
    get_intermediate_state_linewidth, get_hyperfine_splitting
)


# =============================================================================
# LASER PARAMETERS
# =============================================================================

@dataclass
class LaserParameters:
    """
    Parameters for a single Rydberg excitation laser.
    
    In two-photon Rydberg excitation, we use TWO lasers:
    
    1. **First leg** (e.g., 780 nm for Rb): Ground state → Intermediate (5P)
       - Typically uses π polarization
       - Lower power (few mW) since ground-5P coupling is strong
       
    2. **Second leg** (e.g., 480 nm for Rb): Intermediate (5P) → Rydberg (nS)
       - Typically uses σ⁺ polarization (selects mJ = +1/2 Rydberg state)
       - Higher power (few W) because P→Rydberg coupling is weak
    
    **Why does polarization matter?**
    
    Different polarizations couple to different magnetic sublevels:
    - σ⁺ (right circular): Changes mL by +1 (drives ΔmJ = +1)
    - σ⁻ (left circular): Changes mL by -1 (drives ΔmJ = -1)
    - π (linear, z): Doesn't change mL (drives ΔmJ = 0)
    
    Impure polarization (e.g., 1% σ⁻ contamination in σ⁺ beam) can excite
    the wrong mJ state, causing "mJ leakage" errors.
    
    Attributes
    ----------
    power : float
        Laser power in Watts.
        Typical: 1 mW for first leg, 100 mW - 5 W for second leg.
        
    waist : float
        Beam waist (1/e² intensity radius) in meters.
        Typical: 10-100 μm for Rydberg lasers.
        
    polarization : str
        Laser polarization:
        - "sigma+": Right circular (σ⁺)
        - "sigma-": Left circular (σ⁻)  
        - "pi": Linear, parallel to quantization axis
        - "linear": Linear, perpendicular to quantization axis
        
    polarization_purity : float
        Fraction of light in desired polarization (0-1).
        0.99 means 1% contamination with wrong polarization.
        This affects mJ leakage error rates.
        
    linewidth_hz : float
        Laser linewidth (FWHM) in Hz.
        Typical: 100 Hz - 10 kHz for narrow lasers.
        Contributes to laser dephasing: γ_laser ≈ linewidth/2
        
    Example
    -------
    >>> # Standard 480 nm Rydberg laser
    >>> blue_laser = LaserParameters(
    ...     power=1.0,         # 1 W
    ...     waist=20e-6,       # 20 μm
    ...     polarization="sigma+",
    ...     polarization_purity=0.99,
    ...     linewidth_hz=1000  # 1 kHz
    ... )
    """
    power: float = 1e-3  # 1 mW default
    waist: float = 50e-6  # 50 μm default
    polarization: str = "sigma+"
    polarization_purity: float = 0.99  # 1% impurity
    linewidth_hz: float = 100.0  # 100 Hz typical
    
    def peak_intensity(self) -> float:
        """
        Calculate peak intensity at beam center.
        
        For a Gaussian beam: I_peak = 2P / (π w²)
        
        Returns
        -------
        float
            Peak intensity in W/m²
        """
        return 2 * self.power / (np.pi * self.waist**2)
    
    def peak_electric_field(self) -> float:
        """
        Calculate peak electric field amplitude.
        
        From I = (1/2) ε₀ c E₀², we get E₀ = √(2I / ε₀c)
        
        Returns
        -------
        float
            Peak electric field in V/m
        """
        I = self.peak_intensity()
        return np.sqrt(2 * I / (EPS0 * C))


# =============================================================================
# PROTOCOL-SPECIFIC SIMULATION INPUTS
# =============================================================================

@dataclass
class TwoPhotonExcitationConfig:
    """
    Configuration for two-photon Rydberg excitation.
    
    Both LP and JP protocols typically use two-photon excitation:
        |ground⟩ --laser1--> |intermediate⟩ --laser2--> |Rydberg⟩
    
    The effective two-photon Rabi frequency is:
        Ω_eff = Ω₁ × Ω₂ / (2 × Δ_e)
    
    where Δ_e is the detuning from the intermediate state.
    
    Attributes
    ----------
    laser_1 : LaserParameters
        First leg laser (ground → intermediate).
        Rb87: 780 nm, Cs133: 852 nm
        
    laser_2 : LaserParameters
        Second leg laser (intermediate → Rydberg).
        Rb87: ~480 nm, Cs133: ~510 nm
        
    Delta_e : float
        Intermediate state detuning in rad/s.
        Positive = blue-detuned (above intermediate state).
        Typical: 2π × 1-10 GHz
        
    counter_propagating : bool
        Whether lasers are counter-propagating (reduces Doppler sensitivity).
    """
    laser_1: LaserParameters = field(default_factory=lambda: LaserParameters(
        power=50e-6, waist=50e-6, polarization="pi", linewidth_hz=1000
    ))
    laser_2: LaserParameters = field(default_factory=lambda: LaserParameters(
        power=500e-3, waist=50e-6, polarization="sigma+", linewidth_hz=1000
    ))
    Delta_e: float = 2 * np.pi * 1e9  # 1 GHz default
    counter_propagating: bool = True


@dataclass
class NoiseSourceConfig:
    """
    Configuration for noise sources in simulation.
    
    These parameters control which noise sources are included and their
    characteristics. Works for both LP and JP protocols.
    
    Attributes
    ----------
    include_spontaneous_emission : bool
        Include Rydberg state decay (typical rate: 1-10 kHz).
        
    include_intermediate_scattering : bool
        Include off-resonant scattering via intermediate state.
        
    include_motional_dephasing : bool
        Include dephasing from thermal motion in trap.
        
    include_doppler_dephasing : bool
        Include Doppler shift from thermal atomic velocities.
        
    include_intensity_noise : bool
        Include trap intensity fluctuation dephasing.
        
    intensity_noise_frac : float
        RMS fractional intensity noise (e.g., 0.01 = 1%).
        
    include_laser_dephasing : bool
        Include laser linewidth-induced dephasing.
        
    include_magnetic_dephasing : bool
        Include B-field fluctuation dephasing.
    """
    include_spontaneous_emission: bool = True
    include_intermediate_scattering: bool = True
    include_motional_dephasing: bool = True
    include_doppler_dephasing: bool = True
    include_intensity_noise: bool = True
    intensity_noise_frac: float = 0.01
    include_laser_dephasing: bool = True
    include_magnetic_dephasing: bool = True


@dataclass
class LPSimulationInputs:
    """
    Levine-Pichler protocol-specific simulation inputs.
    
    The LP protocol uses two laser pulses with a phase jump ξ between them:
    
        Pulse 1: H = (Ω/2)(|1⟩⟨r| + h.c.) - Δ|r⟩⟨r| + V|rr⟩⟨rr|, duration τ
        Pulse 2: Same but Ω → Ω·e^{iξ}, duration τ
    
    Total gate time: 2τ
    
    **Key features:**
    - Two separate pulses
    - Static detuning Δ throughout
    - Phase jump ξ between pulses
    - Pulse shape is TUNABLE (square, gaussian, cosine, blackman, drag)
    
    Attributes
    ----------
    excitation : TwoPhotonExcitationConfig
        Two-photon laser configuration.
        
    noise : NoiseSourceConfig
        Noise source configuration.
        
    delta_over_omega : float or None
        Detuning ratio Δ/Ω. If None, uses V/Ω-adapted optimal value.
        
    omega_tau : float or None
        Pulse area Ω×τ per pulse. If None, uses V/Ω-adapted optimal value.
        
    pulse_shape : str
        Temporal envelope: "square", "gaussian", "cosine", "blackman", "drag"
        
    drag_lambda : float
        DRAG coefficient (only used if pulse_shape="drag")
    """
    excitation: TwoPhotonExcitationConfig = field(default_factory=TwoPhotonExcitationConfig)
    noise: NoiseSourceConfig = field(default_factory=NoiseSourceConfig)
    delta_over_omega: Optional[float] = None
    omega_tau: Optional[float] = None
    pulse_shape: str = "square"
    drag_lambda: float = 1.0
    
    @property
    def protocol_name(self) -> str:
        return "levine_pichler"
    
    @property
    def n_pulses(self) -> int:
        return 2


@dataclass  
class JPSimulationInputs:
    """
    Jandura-Pupillo protocol-specific simulation inputs.
    
    The JP protocol uses a single pulse with bang-bang phase modulation:
    
        Single pulse: Ω(t) = Ω·e^{iφ(t)}, NO static detuning
        φ(t) ∈ {+π/2, 0, -π/2} switches at optimized times
    
    Total gate time: τ (shorter than LP by ~18%)
    
    **Key features:**
    - Single continuous pulse
    - NO static detuning (Δ = 0)
    - Bang-bang phase control at fixed switching times
    - Pulse shape is NOT tunable (always bang-bang)
    
    **Why pulse_shape is not tunable:**
    The JP protocol's time-optimality comes from saturating the control
    constraint |φ| ≤ π/2 almost everywhere. Smooth pulse shaping would
    violate this and lose the time advantage.
    
    Attributes
    ----------
    excitation : TwoPhotonExcitationConfig
        Two-photon laser configuration (same as LP).
        
    noise : NoiseSourceConfig
        Noise source configuration.
        
    omega_tau : float or None
        Total pulse area Ω×τ. If None, uses V/Ω-adapted optimal value.
        
    switching_times : list of float or None
        Dimensionless times (Ωt) where phase switches. If None, uses
        V/Ω-adapted defaults from protocols.py.
        
    phases : list of float or None
        Phase values for each segment (radians). If None, uses defaults
        from protocols.py.
    """
    excitation: TwoPhotonExcitationConfig = field(default_factory=TwoPhotonExcitationConfig)
    noise: NoiseSourceConfig = field(default_factory=NoiseSourceConfig)
    omega_tau: Optional[float] = None
    switching_times: Optional[List[float]] = None
    phases: Optional[List[float]] = None
    
    @property
    def protocol_name(self) -> str:
        return "jandura_pupillo"
    
    @property
    def pulse_shape(self) -> str:
        """JP always uses bang-bang control - not tunable."""
        return "bangbang"
    
    @property
    def n_pulses(self) -> int:
        return 1


@dataclass
class SmoothJPSimulationInputs:
    """
    Smooth sinusoidal JP (Bluvstein-form) protocol simulation inputs.
    
    The smooth JP protocol uses continuous sinusoidal phase modulation:
    
        φ(t) = A·cos(ω_mod·t - ϕ) + δ₀·t
    
    This implements the high-fidelity CZ gate from Bluvstein's thesis that
    achieved 99.5% fidelity experimentally. The key physics is continuous
    dark state evolution that suppresses Rydberg state population.
    
    Total gate time: τ where Ωτ ≈ 10 (faster than both LP and bang-bang JP)
    
    **Key features:**
    - Single continuous pulse (like bang-bang JP)
    - NO static detuning (Δ = 0)
    - Smooth sinusoidal phase modulation (NOT bang-bang)
    - Modulation frequency ω_mod ≈ 1.24Ω (close to Rabi frequency)
    - δ₀ term provides effective two-photon detuning via phase slope
    
    **Validated performance (V/Ω from 10 to 200):**
    - Fidelity > 99.9% (noise-free)
    - Controlled phase within 3° of ±180°
    - Works across wide range of blockade strengths
    
    Attributes
    ----------
    excitation : TwoPhotonExcitationConfig
        Two-photon laser configuration.
        
    noise : NoiseSourceConfig
        Noise source configuration.
        
    omega_tau : float or None
        Pulse area Ω×τ. If None, uses validated default (10.09).
        
    A : float or None
        Phase modulation amplitude in radians. If None, uses validated
        default (0.311π ≈ 56°).
        
    omega_mod_ratio : float or None
        Ratio ω_mod/Ω. If None, uses validated default (1.242).
        
    phi_offset : float or None
        Phase offset in radians. If None, uses validated default (4.696).
        
    delta_over_omega : float or None
        Effective detuning ratio from phase slope δ₀/Ω. If None, uses
        validated default (0.0205).
        
    Reference
    ---------
    Bluvstein, D., PhD Thesis, Harvard University (2024), Section 5.3
    Evered et al., Nature 622, 268 (2023)
    """
    excitation: TwoPhotonExcitationConfig = field(default_factory=TwoPhotonExcitationConfig)
    noise: NoiseSourceConfig = field(default_factory=NoiseSourceConfig)
    omega_tau: Optional[float] = None
    A: Optional[float] = None
    omega_mod_ratio: Optional[float] = None
    phi_offset: Optional[float] = None
    delta_over_omega: Optional[float] = None
    
    @property
    def protocol_name(self) -> str:
        return "smooth_jp"
    
    @property
    def pulse_shape(self) -> str:
        """Smooth JP uses sinusoidal phase modulation."""
        return "smooth_sinusoidal"
    
    @property
    def n_pulses(self) -> int:
        return 1


@dataclass
class TweezerParameters:
    """
    Optical tweezer (trap) parameters.
    
    **What is an optical tweezer?**
    
    A tightly focused laser beam that traps neutral atoms via the AC Stark
    effect (light shift). The focused beam creates a 3D intensity gradient,
    and atoms are pulled toward the intensity maximum if the trap wavelength
    is red-detuned from atomic transitions.
    
    **Why these parameters matter:**
    
    - **power**: Determines trap depth. Higher power = deeper trap = 
      better confinement, but also more light shift and heating.
      
    - **waist**: Smaller waist = tighter trap = higher trap frequencies,
      but requires more precise alignment and has smaller capture volume.
      
    - **wavelength**: Must be far-detuned from atomic transitions to avoid
      heating. "Magic wavelengths" minimize differential light shifts 
      between qubit states.
      
    - **NA**: Numerical aperture of focusing lens. Higher NA = smaller 
      achievable waist = stronger axial confinement.
    
    **Typical values for Rb87:**
    - 1064 nm Nd:YAG: power ~ 10-50 mW, waist ~ 0.8-1.5 μm
    - 820 nm "near-magic": power ~ 10-30 mW, waist ~ 0.7-1.0 μm
    
    Attributes
    ----------
    power : float
        Tweezer power in Watts.
        
    waist : float
        Beam waist (1/e² intensity radius) in meters.
        Diffraction limit: waist ≥ 0.64 × λ / NA
        
    wavelength_nm : float
        Trap wavelength in nanometers.
        
    NA : float
        Numerical aperture of focusing optics (typically 0.4-0.7).
    """
    power: float = 10e-3  # 10 mW
    waist: float = 0.9e-6  # 0.9 μm
    wavelength_nm: float = 820.0  # Near-magic for Rb
    NA: float = 0.5
    
    def diffraction_limited_waist(self) -> float:
        """
        Calculate minimum achievable waist from diffraction limit.
        
        w₀_min ≈ 0.64 × λ / NA (for Gaussian beam, Airy criterion)
        
        Returns
        -------
        float
            Minimum waist in meters
        """
        wavelength_m = self.wavelength_nm * 1e-9
        return 0.64 * wavelength_m / self.NA
    
    def rayleigh_range(self) -> float:
        """
        Calculate Rayleigh range (depth of focus).
        
        z_R = π w₀² / λ
        
        This is the distance from focus where beam area doubles.
        It determines axial trap frequency (much lower than radial).
        
        Returns
        -------
        float
            Rayleigh range in meters
        """
        wavelength_m = self.wavelength_nm * 1e-9
        return np.pi * self.waist**2 / wavelength_m
    
    def peak_intensity(self) -> float:
        """
        Calculate peak intensity at focus.
        
        Returns
        -------
        float
            Peak intensity in W/m²
        """
        return 2 * self.power / (np.pi * self.waist**2)


@dataclass
class EnvironmentParameters:
    """
    Environmental conditions affecting the atoms.
    
    These parameters describe the experimental conditions that affect
    gate fidelity through various noise channels.
    
    **Temperature (temperature)**
    
    After laser cooling, atoms have residual thermal motion with velocity
    v_th ~ √(kT/m). At T = 10 μK for Rb87:
    - v_th ≈ 3 cm/s
    - Doppler shift: Δf = v/λ ~ 40 kHz (affects laser detuning)
    - Position uncertainty: σ_r ~ √(kT/mω²) ~ 30 nm (affects blockade)
    
    Lower temperature = less motional dephasing, more precise blockade.
    
    **Magnetic field (B_field, B_field_angle)**
    
    A small bias field (~0.5-5 Gauss) defines the quantization axis for
    atomic states. Field fluctuations cause Zeeman dephasing:
    
    - For mF ≠ 0 states: γ_φ ~ g_F × μ_B × δB / ℏ ~ 1.4 MHz/G
    - For clock states (mF = 0): Only quadratic Zeeman, much smaller
    
    The field angle affects selection rules for laser transitions.
    
    **Atom spacing (spacing_factor)**
    
    Two atoms separated by R experience blockade V = C₆/R⁶.
    We typically set R = spacing_factor × R_b where R_b = (C₆/Ω)^(1/6)
    is the "blockade radius" (distance where V = Ω).
    
    - spacing_factor ~ 0.5-1: Strong blockade regime (V >> Ω)
    - spacing_factor ~ 1-2: Intermediate regime
    - spacing_factor >> 2: Weak blockade, independent atoms
    
    Attributes
    ----------
    temperature : float
        Atom temperature in Kelvin.
        Typical: 1-50 μK after laser cooling and evaporation.
        
    B_field : float
        Magnetic field magnitude in Tesla.
        Typical: 0.5-5 Gauss = 0.5-5 × 10⁻⁴ T
        
    B_field_angle : float
        Angle between B-field and laser propagation direction (radians).
        0 = parallel (B along z), π/2 = perpendicular
        
    spacing_factor : float
        Atom spacing as multiple of blockade radius.
        R = spacing_factor × R_b
    """
    temperature: float = 20e-6  # 20 μK
    B_field: float = 0.5e-4  # 0.5 Gauss = 5×10⁻⁵ T
    B_field_angle: float = 0.0  # Parallel to quantization axis
    spacing_factor: float = 2.8  # Typical for n=53
    
    def thermal_velocity(self, mass: float) -> float:
        """
        Calculate RMS thermal velocity.
        
        v_th = √(k_B T / m)
        
        Parameters
        ----------
        mass : float
            Atomic mass in kg
            
        Returns
        -------
        float
            RMS thermal velocity in m/s
        """
        return np.sqrt(KB * self.temperature / mass)
    
    def B_field_gauss(self) -> float:
        """Return magnetic field in Gauss (1 G = 10⁻⁴ T)."""
        return self.B_field / 1e-4


# =============================================================================
# ATOMIC CONFIGURATION
# =============================================================================

@dataclass
class AtomicConfiguration:
    """
    Configuration for atomic species and quantum states.
    
    This class specifies:
    1. Which atomic species (Rb87 or Cs133)
    2. Which Rydberg state (principal quantum number n)
    3. Which hyperfine states encode the qubit |0⟩ and |1⟩
    4. Which intermediate state for two-photon excitation
    
    **Qubit encoding choices:**
    
    The two qubit states |0⟩ and |1⟩ are stored in hyperfine ground states.
    Common choices:
    
    1. "Clock states" (recommended):
       |0⟩ = |F=1, mF=0⟩, |1⟩ = |F=2, mF=0⟩
       - First-order insensitive to magnetic field fluctuations
       - Requires microwave or Raman for single-qubit gates
       
    2. "Stretched states":
       |0⟩ = |F=2, mF=-2⟩, |1⟩ = |F=2, mF=-1⟩
       - Same F: no differential hyperfine light shift
       - Sensitive to B-field → more dephasing
    
    **Rydberg state choice:**
    
    Higher n gives:
    - Stronger C₆ (∝ n¹¹) → stronger blockade
    - Longer lifetime (∝ n³) → less decay during gate
    - Larger polarizability (∝ n⁷) → worse anti-trapping
    - More sensitivity to electric fields (∝ n⁷)
    
    Sweet spot: n ~ 50-70 balances these tradeoffs.
    
    Attributes
    ----------
    species : str
        Atomic species: "Rb87" or "Cs133"
        
    n_rydberg : int
        Principal quantum number of Rydberg state.
        Typical: 50-100
        
    L_rydberg : str
        Orbital angular momentum: "S", "P", "D", or "F"
        Default: "S" (nS₁/₂ states are simplest)
        
    qubit_0 : Tuple[int, int]
        (F, mF) for |0⟩ state
        
    qubit_1 : Tuple[int, int]
        (F, mF) for |1⟩ state
        
    intermediate_state : str
        Label for intermediate state in two-photon excitation.
        "5P3/2" = D2 line (most common for Rb)
        "5P1/2" = D1 line
        
    Example
    -------
    >>> # Standard Rb87 configuration with clock states
    >>> config = AtomicConfiguration(
    ...     species="Rb87",
    ...     n_rydberg=70,
    ...     qubit_0=(1, 0),  # |F=1, mF=0⟩
    ...     qubit_1=(2, 0),  # |F=2, mF=0⟩
    ...     intermediate_state="5P3/2"
    ... )
    >>> print(f"Effective n: {config.n_star:.2f}")
    >>> print(f"C₆: {config.C6 / (2*np.pi*1e9*(1e-6)**6):.1f} GHz·μm⁶")
    """
    species: str = "Rb87"
    n_rydberg: int = 70
    L_rydberg: str = "S"
    qubit_0: Tuple[int, int] = (1, 0)  # |F=1, mF=0⟩
    qubit_1: Tuple[int, int] = (2, 0)  # |F=2, mF=0⟩
    intermediate_state: str = None  # Auto-detected from species if None
    
    def __post_init__(self):
        """Compute derived quantities after initialization."""
        if self.species not in ATOM_DB:
            raise ValueError(f"Unknown species: {self.species}. "
                           f"Available: {list(ATOM_DB.keys())}")
        
        self._atom_props = ATOM_DB[self.species]
        
        # Auto-detect intermediate state based on species
        if self.intermediate_state is None:
            # Rb87: D2 line is 5P3/2 (780 nm)
            # Cs133: D2 line is 6P3/2 (852 nm)
            default_states = {"Rb87": "5P3/2", "Cs133": "6P3/2"}
            object.__setattr__(self, 'intermediate_state', default_states[self.species])
    
    # === Rydberg state properties ===
    
    @property
    def n_star(self) -> float:
        """Effective principal quantum number n* = n - δ."""
        return effective_n(self.n_rydberg, self.species, self.L_rydberg)
    
    @property
    def quantum_defect(self) -> float:
        """Quantum defect δ for the Rydberg state orbital."""
        return get_quantum_defect(self.species, self.L_rydberg)
    
    @property
    def rydberg_energy(self) -> float:
        """Binding energy of Rydberg state (negative, in Joules)."""
        return get_rydberg_energy(self.n_rydberg, self.species, self.L_rydberg)
    
    @property
    def C6(self) -> float:
        """C₆ coefficient for Rydberg-Rydberg interaction (J·m⁶)."""
        return get_C6(self.n_rydberg, self.species)
    
    @property
    def rydberg_lifetime_300K(self) -> float:
        """Rydberg lifetime at 300K including BBR (seconds)."""
        return get_rydberg_lifetime(self.n_rydberg, self.species, 300.0)
    
    @property
    def rydberg_lifetime_0K(self) -> float:
        """Rydberg lifetime at 0K, spontaneous only (seconds)."""
        return get_rydberg_lifetime(self.n_rydberg, self.species, 0.0)
    
    @property
    def rydberg_polarizability(self) -> float:
        """Rydberg state polarizability at trap wavelength (SI units)."""
        return get_rydberg_polarizability(self.n_rydberg, self.species)
    
    # === Ground state properties ===
    
    @property
    def mass(self) -> float:
        """Atomic mass in kg."""
        return self._atom_props["mass"]
    
    @property
    def hyperfine_splitting(self) -> float:
        """Ground state hyperfine splitting in Hz."""
        return get_hyperfine_splitting(self.species)
    
    @property
    def ground_polarizability(self) -> float:
        """Ground state polarizability at trap wavelength (SI units)."""
        return self._atom_props["alpha_ground"]
    
    @property
    def is_clock_transition(self) -> bool:
        """Check if qubit uses clock states (mF=0 ↔ mF=0)."""
        return self.qubit_0[1] == 0 and self.qubit_1[1] == 0
    
    @property
    def delta_mF(self) -> int:
        """Difference in mF between qubit states."""
        return abs(self.qubit_1[1] - self.qubit_0[1])
    
    @property
    def delta_F(self) -> int:
        """Difference in F between qubit states."""
        return abs(self.qubit_1[0] - self.qubit_0[0])
    
    # === Intermediate state properties ===
    
    @property
    def intermediate_linewidth(self) -> float:
        """Natural linewidth Γ of intermediate state (rad/s)."""
        return get_intermediate_state_linewidth(self.species, self.intermediate_state)
    
    # === Excitation laser wavelengths ===
    
    @property
    def excitation_wavelength_1_nm(self) -> float:
        """
        First leg excitation wavelength (ground → intermediate) in nm.
        
        For two-photon Rydberg excitation:
        - Rb87: 780 nm (5S → 5P3/2, D2 line)
        - Cs133: 852 nm (6S → 6P3/2, D2 line)
        
        Computed from transition frequency: λ = c / f
        """
        transitions = self._atom_props.get("transitions", {})
        
        # Look up transition frequency based on intermediate state
        if self.species == "Rb87":
            if self.intermediate_state == "5P3/2":
                freq = transitions.get("ground_to_5P3/2", 384.230484468e12)
            else:  # 5P1/2
                freq = transitions.get("ground_to_5P1/2", 377.107385690e12)
        else:  # Cs133
            if self.intermediate_state == "6P3/2":
                freq = transitions.get("ground_to_6P3/2", 351.725718509e12)
            else:  # 6P1/2
                freq = transitions.get("ground_to_6P1/2", 335.116048807e12)
        
        # λ = c / f, convert to nm
        return C / freq * 1e9
    
    @property
    def excitation_wavelength_2_nm(self) -> float:
        """
        Second leg excitation wavelength (intermediate → Rydberg) in nm.
        
        For two-photon Rydberg excitation:
        - Rb87: ~480 nm (5P → nS)
        - Cs133: ~510 nm (6P → nS)
        
        Computed from energy conservation:
        E_rydberg = E_ground + h*f1 + h*f2
        
        Where E_rydberg = E_ionization + E_binding(n)
        """
        # Energy of first photon (ground → intermediate)
        transitions = self._atom_props.get("transitions", {})
        
        if self.species == "Rb87":
            if self.intermediate_state == "5P3/2":
                freq1 = transitions.get("ground_to_5P3/2", 384.230484468e12)
            else:
                freq1 = transitions.get("ground_to_5P1/2", 377.107385690e12)
        else:
            if self.intermediate_state == "6P3/2":
                freq1 = transitions.get("ground_to_6P3/2", 351.725718509e12)
            else:
                freq1 = transitions.get("ground_to_6P1/2", 335.116048807e12)
        
        # Energy to ionization threshold
        E_ion = self._atom_props["E_ionization"]
        
        # Rydberg binding energy (negative)
        E_binding = self.rydberg_energy
        
        # Total energy needed: E_ion + E_binding (binding is negative)
        # = E_ion - |E_binding|
        E_total_rydberg = E_ion + E_binding
        
        # Energy of second photon
        E_photon1 = HBAR * 2 * np.pi * freq1
        E_photon2 = E_total_rydberg - E_photon1
        
        # Convert to wavelength
        freq2 = E_photon2 / (HBAR * 2 * np.pi)
        return C / freq2 * 1e9

    # === Convenience methods ===
    
    def get_g_F(self, state: Tuple[int, int]) -> float:
        """Get Landé g-factor for a hyperfine state."""
        F = state[0]
        return self._atom_props["g_F"][F]
    
    def blockade_radius(self, Omega: float) -> float:
        """
        Calculate blockade radius for given Rabi frequency.
        
        R_b = (C₆/ℏΩ)^(1/6)
        
        This is the distance where V = ℏΩ (blockade equals Rabi).
        
        Parameters
        ----------
        Omega : float
            Rabi frequency in rad/s
            
        Returns
        -------
        float
            Blockade radius in meters
        """
        return (self.C6 / (HBAR * Omega))**(1/6)
    
    def blockade_shift(self, R: float) -> float:
        """
        Calculate blockade energy shift at distance R.
        
        V(R) = C₆/R⁶
        
        Parameters
        ----------
        R : float
            Interatomic separation in meters
            
        Returns
        -------
        float
            Blockade shift in Joules
        """
        return self.C6 / R**6
    
    def summary(self) -> str:
        """Return a formatted summary of the configuration."""
        lines = [
            f"AtomicConfiguration Summary",
            f"=" * 40,
            f"Species: {self.species}",
            f"Rydberg state: {self.n_rydberg}{self.L_rydberg}₁/₂",
            f"  n* = {self.n_star:.3f}",
            f"  C₆/2π = {self.C6 / (2*np.pi):.2e} Hz·m⁶",
            f"        = {self.C6 / (2*np.pi*1e9*(1e-6)**6):.1f} GHz·μm⁶",
            f"  τ(300K) = {self.rydberg_lifetime_300K*1e6:.1f} μs",
            f"Qubit encoding:",
            f"  |0⟩ = |F={self.qubit_0[0]}, mF={self.qubit_0[1]}⟩",
            f"  |1⟩ = |F={self.qubit_1[0]}, mF={self.qubit_1[1]}⟩",
            f"  Clock transition: {self.is_clock_transition}",
            f"Intermediate: {self.intermediate_state}",
        ]
        return "\n".join(lines)


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

# NOTE: The legacy SimulateCZParameters / LPSimulatorParameters /
# JPSimulatorParameters hierarchy has been removed. The simulation engine
# consumes the lightweight protocol-specific inputs:
#   - LPSimulationInputs
#   - JPSimulationInputs  (routes through SmoothJPSimulationInputs internally)
#   - SmoothJPSimulationInputs
# together with the component configs above (AtomicConfiguration,
# TweezerParameters, EnvironmentParameters, LaserParameters, etc.).
#
# For preset apparatus parameters, see get_standard_rb87_config() and
# get_standard_cs133_config() below, or use the optimize_cz_gate module's
# ApparatusConstraints for automated optimisation.


def get_standard_rb87_config(n_rydberg: int = 70) -> AtomicConfiguration:
    """
    Return standard Rb87 configuration with clock states.
    
    Parameters
    ----------
    n_rydberg : int
        Principal quantum number (default 70)
        
    Returns
    -------
    AtomicConfiguration
        Standard Rb87 setup
    """
    return AtomicConfiguration(
        species="Rb87",
        n_rydberg=n_rydberg,
        L_rydberg="S",
        qubit_0=(1, 0),
        qubit_1=(2, 0),
        intermediate_state="5P3/2"
    )


def get_standard_cs133_config(n_rydberg: int = 70) -> AtomicConfiguration:
    """
    Return standard Cs133 configuration with clock states.
    
    Parameters
    ----------
    n_rydberg : int
        Principal quantum number (default 70)
        
    Returns
    -------
    AtomicConfiguration
        Standard Cs133 setup
    """
    return AtomicConfiguration(
        species="Cs133",
        n_rydberg=n_rydberg,
        L_rydberg="S",
        qubit_0=(3, 0),  # Cs has F=3,4 not F=1,2
        qubit_1=(4, 0),
        intermediate_state="6P3/2"
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Component configs
    "LaserParameters",
    "TweezerParameters", 
    "EnvironmentParameters",
    "AtomicConfiguration",
    
    # Two-photon excitation
    "TwoPhotonExcitationConfig",
    
    # Noise configuration
    "NoiseSourceConfig",
    
    # Protocol-specific simulation inputs
    "LPSimulationInputs",
    "JPSimulationInputs",
    "SmoothJPSimulationInputs",
    
    # Preset factory functions
    "get_standard_rb87_config",
    "get_standard_cs133_config",
]
