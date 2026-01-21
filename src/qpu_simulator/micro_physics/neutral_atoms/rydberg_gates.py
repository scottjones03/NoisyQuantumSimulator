# Rydberg Gate Micro-Physics
#
# QuTiP-based simulation of Rydberg-mediated entangling gates.
#
# Physics included:
#   - 4-level atomic model per atom: |0⟩, |1⟩, |e⟩, |r⟩
#   - Time-dependent Hamiltonian with laser coupling
#   - Rydberg-Rydberg interaction (van der Waals blockade)
#   - Lindblad dissipation: spontaneous emission, Rydberg decay, dephasing
#
# Inputs (System State):
#   - Initial atom state (density matrix)
#
# Inputs (Hardware Parameters):
#   - Rydberg n, Rabi frequencies Ω(t), detunings Δ/δ
#   - Laser phase φ(t), pulse shape/duration
#   - Decay rates: γ_e, γ_r, T2*
#   - Inter-atom distance r_12
#   - C6(n) coefficient
#
# Inputs (Architectural):
#   - Gate type (π-pulse, CZ)
#   - Target qubits
#
# Outputs:
#   - CPTP map for the gate
#   - Gate fidelity
#   - Leakage probability
#   - Gate duration
#   - Conditional atom loss probability
