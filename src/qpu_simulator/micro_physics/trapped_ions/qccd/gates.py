# QCCD Gate Models
#
# QuTiP-based simulation of trapped ion gates in QCCD architecture.
#
# Physics included:
#   - Two-level qubit model (typically hyperfine or Zeeman)
#   - Mølmer-Sørensen gate via bichromatic laser fields
#   - Geometric phase gates
#   - Single-qubit Raman/microwave gates
#   - Motional mode coupling
#   - Spectral crowding and crosstalk
#
# Inputs (System State):
#   - Ion qubit states
#   - Motional mode occupation
#
# Inputs (Hardware Parameters):
#   - Trap frequencies (axial, radial)
#   - Rabi frequencies
#   - Gate detuning from motional sidebands
#   - Heating rates
#   - Number of ions in interaction zone
#
# Outputs:
#   - CPTP map for gate
#   - Gate duration
#   - Infidelity contributions (motional, off-resonant, heating)
#   - Crosstalk to neighboring ions
