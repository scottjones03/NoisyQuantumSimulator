# Neutral Atom Hardware Configuration
#
# Parameters for Rydberg-based neutral atom quantum computers.
#
# Atomic Species:
#   - Rb-87 (most common)
#   - Cs-133
#   - Sr-88
#
# Key Parameters:
#
# Atomic properties:
#   - Qubit states (hyperfine or Zeeman)
#   - Intermediate state (5P for Rb)
#   - Rydberg state (principal quantum number n)
#
# Rydberg properties:
#   - C6 coefficient vs n
#   - Blockade radius vs n
#   - Rydberg lifetime (T1) vs n
#   - Dephasing time (T2*)
#
# Laser parameters:
#   - Wavelengths (780nm, 480nm for Rb)
#   - Rabi frequencies (Ω_ge, Ω_er)
#   - Detunings (Δ, δ)
#   - Pulse shapes (Gaussian, Blackman, etc.)
#
# Trap parameters:
#   - Tweezer wavelength (typically 850nm or 1064nm)
#   - Trap depth
#   - Beam waist
#   - Trap frequencies
#
# Motion parameters:
#   - Max AOD velocity
#   - Max acceleration
#   - Heating rate per move
#
# Measurement:
#   - Detection efficiency
#   - Atom loss per measurement
#   - Readout time
#
# Example configurations:
#   - Lukin group (Harvard): n~70, 2D arrays
#   - Browaeys group (Pasqual): n~60, 3D arrays
#   - QuEra: Large-scale 2D arrays
