# Single-Qubit Gate Micro-Physics for Neutral Atoms
#
# QuTiP-based simulation of single-qubit operations via Raman or microwave transitions.
#
# Physics included:
#   - 3-4 level Λ-system model
#   - Spontaneous Raman scattering
#   - Laser phase noise
#   - AC Stark shifts
#
# Inputs (Hardware Parameters):
#   - Rabi frequency Ω
#   - Detuning from intermediate state
#   - Scattering rate
#   - Pulse shape and duration
#
# Outputs:
#   - CPTP map for single-qubit gate
#   - Gate duration
#   - Error rate (depolarizing, dephasing)
#   - Heating probability (via recoil)
