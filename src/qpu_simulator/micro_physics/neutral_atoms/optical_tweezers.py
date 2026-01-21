# Optical Tweezer Trapping Model
#
# Classical harmonic trap model for optical tweezer confinement.
#
# Physics included:
#   - Gaussian beam approximation for trap potential
#   - Harmonic oscillator model: V(x) = (1/2) m ω² (x - x₀)²
#   - Trap depth from laser intensity
#   - Trap frequency from curvature
#   - Photon scattering and heating
#   - Thermal distribution of motional states
#
# Inputs (Hardware Parameters):
#   - Laser wavelength, power, waist
#   - Atomic mass and polarizability
#   - Temperature / initial motional state
#
# Outputs:
#   - Trap depth U
#   - Trap frequencies (ω_x, ω_y, ω_z)
#   - Scattering-induced heating rate
#   - Atom lifetime in trap
#   - Probability of escape given temperature
