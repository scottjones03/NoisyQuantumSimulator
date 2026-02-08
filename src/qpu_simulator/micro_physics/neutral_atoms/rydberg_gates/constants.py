"""
Physical Constants for Rydberg Gate Simulations
=================================================

This module defines all fundamental physical constants used throughout the
Rydberg gate simulation. All values are in SI units unless otherwise noted.

WHY THESE CONSTANTS MATTER FOR RYDBERG PHYSICS
----------------------------------------------

**ℏ (HBAR) - Reduced Planck constant**
    The quantum of action. Appears EVERYWHERE in quantum mechanics:
    - Energy-frequency relation: E = ℏω (photon energy)
    - Rabi frequency: Ω = d·E₀/ℏ (how fast we drive transitions)
    - Heisenberg uncertainty: ΔxΔp ≥ ℏ/2 (limits on measurement)
    
    For Rydberg atoms: Typical Rabi frequencies are Ω ~ 2π × 1-10 MHz,
    which corresponds to ℏΩ ~ 10⁻²⁷ J = 10⁻⁸ eV of coupling energy.

**ε₀ (EPS0) - Vacuum permittivity**
    Determines the strength of electrostatic interactions:
    - Coulomb's law: F = q₁q₂/(4πε₀r²)
    - Polarizability units: α has SI units of C²·m²/J = 4πε₀×(volume)
    
    For trapped atoms: The AC Stark shift (trap depth) is:
    U = -α·I/(2ε₀c), where I is laser intensity.

**c (C) - Speed of light**
    Connects energy and momentum of photons:
    - E = ℏω = ℏck (photon energy-wavevector relation)
    - Laser intensity: I = (1/2)ε₀c|E₀|² (power per area)
    
    For two-photon excitation: The lasers have wavelengths ~780nm and ~480nm,
    meaning k ~ 2π/(500nm) ~ 10⁷ m⁻¹, giving photon momenta ℏk ~ 10⁻²⁷ kg·m/s.

**e (E_CHARGE) - Elementary charge**
    The charge of a proton (negative of electron charge):
    - Atomic units: 1 a.u. of charge = e
    - Dipole moments: d = e·r (charge × distance)
    
    For Rydberg atoms: Transition dipole moments are d ~ n²·e·a₀,
    meaning high-n states have HUGE dipoles (hundreds of Debye).

**a₀ (A0) - Bohr radius**
    The natural length scale of atomic physics:
    - a₀ = ℏ²/(mₑe²·4πε₀) ≈ 0.529 Å = 5.29×10⁻¹¹ m
    - Hydrogen ground state size: ⟨r⟩ = 1.5 a₀
    - Rydberg orbital radius: r ~ n² a₀ (grows quadratically!)
    
    For n=70 Rydberg: r ~ 70² × a₀ ≈ 2600 Å = 0.26 μm
    This is comparable to OPTICAL WAVELENGTHS!

**kB (KB) - Boltzmann constant**
    Connects temperature to energy:
    - Thermal energy: kBT ~ 25 meV at room temperature (T=300K)
    - 1 μK ↔ kB × 1μK ~ 10⁻²⁹ J ~ 10⁻¹⁰ eV
    
    For ultracold atoms: Trap depths are measured in mK or μK.
    At T = 10 μK, thermal velocity v_th ~ √(kBT/m) ~ 1 cm/s for Rb.

**μB (MU_B) - Bohr magneton**
    The quantum of magnetic moment:
    - μB = eℏ/(2mₑ) ≈ 9.27×10⁻²⁴ J/T
    - Zeeman shift: ΔE = gμBBmJ (energy shift in magnetic field)
    
    For qubit control: Magnetic field fluctuations of 1 mG cause
    frequency shifts of ~1.4 kHz × g × ΔmF, which can dephase the qubit.

**Ry (RY_JOULES, RY_EV) - Rydberg constant**
    The binding energy of hydrogen:
    - Ry = mₑe⁴/(2(4πε₀)²ℏ²) ≈ 13.6 eV
    - Hydrogen energy levels: Eₙ = -Ry/n²
    
    For Rydberg atoms with quantum defect δ:
    Eₙ = -Ry/(n-δ)² = -Ry/n*²
    
    At n=70: Binding energy ~ Ry/70² ~ 2.8 meV ~ 0.7 THz
    This is in the microwave/terahertz range!

References
----------
CODATA 2018 recommended values:
https://physics.nist.gov/cuu/Constants/

Steck, D.A., "Rubidium 87 D Line Data" (2021):
https://steck.us/alkalidata/rubidium87numbers.pdf
"""

import numpy as np

# =============================================================================
# FUNDAMENTAL CONSTANTS (CODATA 2018)
# =============================================================================

HBAR = 1.054571817e-34  # Reduced Planck constant [J·s]
"""
The fundamental quantum of action.

**Physical meaning**: The smallest "packet" of angular momentum or action
that can exist in nature. Appears in virtually every quantum formula.

**For Rydberg gates**: Determines the relationship between laser power
and Rabi frequency through Ω = d·E₀/ℏ.

**Numerical value**: 1.054571817×10⁻³⁴ J·s

**Useful conversions**:
- ℏ × 2π × 1 MHz = 6.63×10⁻²⁸ J (energy at 1 MHz frequency)
- ℏ × 2π × 1 GHz = 6.63×10⁻²⁵ J (energy at 1 GHz frequency)
"""

EPS0 = 8.8541878128e-12  # Vacuum permittivity [F/m = C²/(N·m²)]
"""
The permittivity of free space.

**Physical meaning**: Determines how strong electric fields are for a given
charge distribution. Appears in Coulomb's law and in electromagnetic wave
propagation.

**For Rydberg gates**: Used to calculate:
1. AC Stark shifts (trap depth): U = -α·I/(2ε₀c)
2. Laser intensity from electric field: I = (1/2)ε₀c·E₀²
3. Converting between atomic units and SI units for polarizability

**Numerical value**: 8.854×10⁻¹² F/m
"""

C = 299792458.0  # Speed of light [m/s] (exact by definition)
"""
The speed of light in vacuum.

**Physical meaning**: The maximum speed at which information can travel.
Also sets the relationship between electric and magnetic fields in EM waves.

**For Rydberg gates**: Used to:
1. Convert between wavelength and frequency: λ = c/f
2. Calculate laser intensity: I = (1/2)ε₀c·E₀²
3. Determine photon recoil momentum: p = ℏk = ℏω/c

**Numerical value**: 299,792,458 m/s (exact)

**Useful conversions**:
- 780 nm light has frequency c/(780nm) = 384 THz
- 480 nm light has frequency c/(480nm) = 625 THz
"""

E_CHARGE = 1.602176634e-19  # Elementary charge [C] (exact by definition)
"""
The charge of a proton (magnitude of electron charge).

**Physical meaning**: The fundamental unit of electric charge. All observed
charges in nature are integer multiples of e.

**For Rydberg gates**: Used for:
1. Dipole moments: d = e × a₀ × (matrix element) 
2. Converting atomic units: 1 a.u. charge = e
3. Zeeman shifts: involves electron charge and mass

**Numerical value**: 1.602176634×10⁻¹⁹ C (exact since 2019 SI redefinition)

**Atomic units conversion**:
- 1 a.u. of electric field = e/(4πε₀a₀²) = 5.14×10¹¹ V/m
- 1 a.u. of dipole = e×a₀ = 8.48×10⁻³⁰ C·m = 2.54 Debye
"""

A0 = 5.29177210903e-11  # Bohr radius [m]
"""
The most probable distance of electron from nucleus in hydrogen ground state.

**Physical meaning**: The natural length scale of atomic physics. All atomic
sizes, orbitals, and distances are most naturally expressed in units of a₀.

**For Rydberg atoms**: The orbital radius scales as n²×a₀:
- n=1: r ~ 1.5 a₀ = 0.79 Å (ground state hydrogen)
- n=50: r ~ 2500 a₀ = 132 nm (visible light wavelength!)
- n=70: r ~ 4900 a₀ = 259 nm (Rydberg atom is HUGE)
- n=100: r ~ 10000 a₀ = 529 nm (larger than optical wavelength)

**Numerical value**: 5.292×10⁻¹¹ m = 0.5292 Å

**Conversions**:
- 1 μm = 18,897 a₀ (typical trap/atom separation scale)
- 1 nm = 18.9 a₀
"""

KB = 1.380649e-23  # Boltzmann constant [J/K] (exact by definition)
"""
Connects temperature to energy.

**Physical meaning**: kB×T is the characteristic thermal energy at temperature T.
Determines how thermal fluctuations affect system behavior.

**For ultracold atoms**: 
- Room temperature (300 K): kBT = 25 meV = 6 THz (way too hot for quantum!)
- Doppler limit (~140 μK for Rb): kBT = 12 neV = 2.9 MHz
- Typical experiments (1-20 μK): kBT = 0.1-2 neV = 20-400 kHz
- Ground state cooling (<1 μK): kBT < 0.1 neV

**Numerical value**: 1.380649×10⁻²³ J/K (exact since 2019)

**Temperature-to-frequency conversion**:
- kB × 1 μK / h = 20.8 kHz (useful for dephasing estimates)
- kB × 1 mK / h = 20.8 MHz (useful for trap depth)
"""

MU_B = 9.2740100783e-24  # Bohr magneton [J/T]
"""
The quantum of magnetic moment for an electron.

**Physical meaning**: The natural unit for atomic magnetic moments.
An electron has magnetic moment μ = -gₛμB×S, where gₛ ≈ 2 is the spin g-factor.

**For qubit control**: The Zeeman shift is ΔE = gFμBBmF:
- For Rb87 F=1 state (gF = -1/2): ΔE/h = -0.7 MHz/G × mF
- For Rb87 F=2 state (gF = +1/2): ΔE/h = +0.7 MHz/G × mF
- Clock states (mF=0↔mF=0): Quadratic Zeeman shift only (~575 Hz/G²)

**Numerical value**: 9.274×10⁻²⁴ J/T

**Practical numbers**:
- 1 Gauss = 10⁻⁴ T
- μB×(1 G)/h = 1.4 MHz (linear Zeeman shift per Gauss for g=1)
- For clock transition (F=1,mF=0 ↔ F=2,mF=0): shift = 575 Hz × B²(G²)
"""

# =============================================================================
# DERIVED CONSTANTS
# =============================================================================

RY_JOULES = 2.1798723611035e-18  # Rydberg constant [J]
"""
The Rydberg energy - binding energy of hydrogen ground state.

**Physical meaning**: The fundamental energy scale of atomic physics.
All atomic binding energies are expressible in terms of Ry.

**Formula**: Ry = mₑe⁴/(2(4πε₀)²ℏ²) = mₑc²α²/2

**For Rydberg atoms**: The binding energy of state n is:
    Eₙ = -Ry/(n-δₗ)² = -Ry/n*²
    
where δₗ is the quantum defect and n* = n - δₗ is the effective principal
quantum number. The quantum defect accounts for core penetration.

**At n=70 for Rb (δₛ ≈ 3.13)**:
    n* = 70 - 3.13 = 66.87
    E₇₀ = -Ry/(66.87)² = -4.88 meV = -1.18 THz = -39.4 cm⁻¹

**Numerical value**: 2.180×10⁻¹⁸ J
"""

RY_EV = RY_JOULES / E_CHARGE  # Rydberg constant [eV] ≈ 13.6 eV
"""
The Rydberg energy in electron-volts.

**Numerical value**: 13.605693 eV

**For quick estimates**:
- Binding energy at n=50: Ry/50² = 5.4 meV
- Binding energy at n=70: Ry/70² = 2.8 meV
- Binding energy at n=100: Ry/100² = 1.4 meV

These correspond to frequencies ~1 THz, in the far-infrared/terahertz range.
"""

# =============================================================================
# NUCLEAR G-FACTORS
# =============================================================================
# These determine the hyperfine structure and Zeeman shifts of atomic states.

G_I_RB87 = -0.0009951414  # Rb87 nuclear g-factor (dimensionless)
"""
Nuclear g-factor for ⁸⁷Rb.

**Physical meaning**: The nuclear magnetic moment is μI = gI × μN × I,
where μN is the nuclear magneton. The nuclear g-factor is negative for ⁸⁷Rb,
meaning the nuclear magnetic moment points opposite to the nuclear spin.

**For hyperfine structure**: The hyperfine splitting comes from the interaction
between nuclear and electronic magnetic moments. For ⁸⁷Rb 5S₁/₂:
    - F=1 (I and J anti-parallel): lower energy
    - F=2 (I and J parallel): higher energy
    - Splitting: 6.835 GHz

**Numerical value**: -9.95×10⁻⁴

Note: The nuclear magneton μN = eℏ/(2mp) ≈ μB/1836 is much smaller than
the Bohr magneton, so nuclear Zeeman shifts are ~2000× smaller than
electronic ones at the same field.
"""

G_I_CS133 = -0.00039885395  # Cs133 nuclear g-factor (dimensionless)
"""
Nuclear g-factor for ¹³³Cs.

**For hyperfine structure**: ¹³³Cs has nuclear spin I = 7/2.
Ground state hyperfine splitting: 9.193 GHz (defines the SI second!)

**Numerical value**: -3.99×10⁻⁴
"""

G_E = 2.00231930436256  # Free electron g-factor (dimensionless)
"""
The electron spin g-factor.

**Physical meaning**: In Dirac theory gₑ = 2 exactly. QED corrections give
the "anomalous" value gₑ = 2.00231930436... The deviation from 2 is one of
the most precisely measured quantities in physics.

**For Rydberg atoms**: Determines Zeeman splitting of fine structure levels.
For S₁/₂ states (L=0): gJ ≈ gₑ ≈ 2.002
For P₃/₂ states: gJ = (1 + (J(J+1) + S(S+1) - L(L+1))/(2J(J+1))) ≈ 4/3

**Numerical value**: 2.00231930436256
"""

# =============================================================================
# UNIT CONVERSION HELPERS
# =============================================================================

def frequency_to_energy(freq_hz: float) -> float:
    """
    Convert frequency (Hz) to energy (Joules).
    
    E = ℏω = h×f = 2πℏ×f
    
    Parameters
    ----------
    freq_hz : float
        Frequency in Hz
        
    Returns
    -------
    float
        Energy in Joules
        
    Examples
    --------
    >>> frequency_to_energy(1e6)  # 1 MHz
    6.626e-28  # ~6.6×10⁻²⁸ J
    
    >>> frequency_to_energy(6.835e9)  # Rb87 hyperfine splitting
    4.53e-24  # ~4.5×10⁻²⁴ J = 28 μeV
    """
    return 2 * np.pi * HBAR * freq_hz


def energy_to_frequency(energy_joules: float) -> float:
    """
    Convert energy (Joules) to frequency (Hz).
    
    f = E/(2πℏ) = E/h
    
    Parameters
    ----------
    energy_joules : float
        Energy in Joules
        
    Returns
    -------
    float
        Frequency in Hz
    """
    return energy_joules / (2 * np.pi * HBAR)


def temperature_to_energy(temp_kelvin: float) -> float:
    """
    Convert temperature (K) to thermal energy (J).
    
    E = kB × T
    
    Parameters
    ----------
    temp_kelvin : float
        Temperature in Kelvin
        
    Returns
    -------
    float
        Thermal energy in Joules
        
    Examples
    --------
    >>> temperature_to_energy(10e-6)  # 10 μK
    1.38e-28  # ~1.4×10⁻²⁸ J
    """
    return KB * temp_kelvin


def wavelength_to_frequency(wavelength_m: float) -> float:
    """
    Convert wavelength (m) to frequency (Hz).
    
    f = c / λ
    
    Parameters
    ----------
    wavelength_m : float
        Wavelength in meters
        
    Returns
    -------
    float
        Frequency in Hz
        
    Examples
    --------
    >>> wavelength_to_frequency(780e-9)  # 780 nm (Rb D2 line)
    3.84e14  # 384 THz
    """
    return C / wavelength_m


def au_to_si_polarizability(alpha_au: float) -> float:
    """
    Convert polarizability from atomic units to SI units.
    
    1 a.u. of polarizability = 4πε₀ × a₀³
    
    In atomic units, the ground state Rb polarizability is ~319 a.u.
    This converts to ~5.3×10⁻³⁹ C²·m²/J in SI.
    
    Parameters
    ----------
    alpha_au : float
        Polarizability in atomic units
        
    Returns
    -------
    float
        Polarizability in SI units (C²·m²/J = F·m²)
    """
    return alpha_au * 4 * np.pi * EPS0 * A0**3


def si_to_au_polarizability(alpha_si: float) -> float:
    """
    Convert polarizability from SI units to atomic units.
    
    Parameters
    ----------
    alpha_si : float
        Polarizability in SI units (C²·m²/J)
        
    Returns
    -------
    float
        Polarizability in atomic units
    """
    return alpha_si / (4 * np.pi * EPS0 * A0**3)


# =============================================================================
# EXPORT ALL
# =============================================================================

__all__ = [
    # Fundamental constants
    "HBAR", "EPS0", "C", "E_CHARGE", "A0", "KB", "MU_B",
    # Derived constants
    "RY_JOULES", "RY_EV",
    # Nuclear g-factors
    "G_I_RB87", "G_I_CS133", "G_E",
    # Unit conversions
    "frequency_to_energy", "energy_to_frequency",
    "temperature_to_energy", "wavelength_to_frequency",
    "au_to_si_polarizability", "si_to_au_polarizability",
]
