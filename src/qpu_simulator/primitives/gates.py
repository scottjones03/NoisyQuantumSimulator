# Gate Primitives
#
# Abstraction for quantum gate operations.
#
# Single-Qubit Gates:
#   - Pauli gates (X, Y, Z)
#   - Rotations (Rx, Ry, Rz)
#   - Hadamard, Phase, T gates
#   - Arbitrary U3 rotations
#
# Two-Qubit Gates:
#   - CZ (Controlled-Z) - native for Rydberg
#   - MS (Mølmer-Sørensen) - native for ions
#   - CNOT, CX - derived
#   - Geometric phase gates
#
# API:
#   SingleQubitGate(qubit_id, gate_type, angle=None, axis=None)
#   TwoQubitGate(qubit_1, qubit_2, gate_type, distance=None)
#
# Returns GateResult:
#   - duration: Gate time
#   - fidelity: 1 - error
#   - error_map: CPTP channel (Pauli, depolarizing, etc.)
#   - leakage: Population outside computational basis
#   - crosstalk: Effect on neighboring qubits
#
# Platform mappings:
#   - Neutral atoms: Rydberg CZ, Raman single-qubit
#   - Ions: MS gate, microwave/Raman single-qubit
#   - Cavity: Photon-mediated entangling
