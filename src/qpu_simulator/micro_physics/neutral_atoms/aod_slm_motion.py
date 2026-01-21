# AOD/SLM Atom Motion Model
#
# Classical simulation of atom reconfiguration via acousto-optic deflectors
# and spatial light modulators.
#
# Physics included:
#   - Waveform → trap center trajectory mapping
#   - Langevin dynamics in moving harmonic potential
#   - Motional heating from acceleration
#   - Velocity and acceleration limits
#
# Equations of motion:
#   m ẍ = -m ω² (x - x₀(t)) - γ ẋ + ξ(t)
#
# Where:
#   - x₀(t) from AOD/SLM waveform
#   - γ = damping from cooling
#   - ξ(t) = stochastic heating noise
#
# Inputs (System State):
#   - Current atom position, velocity
#   - Motional excitation / temperature
#
# Inputs (Hardware Parameters):
#   - AOD waveform: frequency vs time
#   - SLM intensity map
#   - Trap depth during motion
#   - Max velocity, max acceleration
#
# Inputs (Architectural):
#   - Start position, end position
#   - Timing constraints
#
# Outputs:
#   - Trajectory x(t), y(t)
#   - Move duration
#   - Final motional energy / temperature
#   - Heating increment ΔT
#   - Loss probability
