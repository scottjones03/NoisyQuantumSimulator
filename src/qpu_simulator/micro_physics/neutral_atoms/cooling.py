# Neutral Atom Cooling Model
#
# Models for resetting motional state after heating.
#
# Physics included:
#   - Optical molasses cooling
#   - Sideband cooling (resolved sideband)
#   - Doppler cooling limits
#   - Cooling time vs final temperature tradeoff
#
# Inputs (System State):
#   - Current motional temperature
#
# Inputs (Hardware Parameters):
#   - Cooling laser parameters
#   - Trap frequencies
#   - Target temperature
#
# Outputs:
#   - Cooling duration
#   - Final motional state / temperature
#   - Success probability
#   - Atom loss during cooling
