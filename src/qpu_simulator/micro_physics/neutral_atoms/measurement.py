# Neutral Atom Measurement Model
#
# Classical/stochastic model for fluorescence-based state readout.
#
# Physics included:
#   - Bright/dark state discrimination via photon scattering
#   - Poisson photon statistics
#   - Detection efficiency and threshold
#   - Atom loss during measurement
#   - State preparation and measurement (SPAM) errors
#
# Inputs (System State):
#   - Atom internal state (qubit state)
#
# Inputs (Hardware Parameters):
#   - Detection laser power, duration
#   - Scattering rate for bright state
#   - Photon collection efficiency
#   - Camera/detector parameters
#
# Outputs:
#   - Measurement confusion matrix
#   - Readout fidelity (bright→bright, dark→dark)
#   - False positive/negative rates
#   - Atom loss probability
#   - Measurement duration
