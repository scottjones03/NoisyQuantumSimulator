# Cavity-Mediated Gate Models
#
# QuTiP-based simulation of cavity-mediated entangling operations.
#
# Physics included:
#   - Jaynes-Cummings Hamiltonian
#   - Two atoms in shared cavity mode
#   - Virtual photon exchange
#   - Cavity decay (κ) and atom decay (γ)
#
# Inputs (Hardware Parameters):
#   - Atom-cavity coupling g
#   - Cavity detuning Δ
#   - Cavity decay rate κ
#   - Atom spontaneous emission γ
#   - Laser Rabi frequency Ω
#
# Outputs:
#   - CPTP map for entangling gate
#   - Decoherence from cavity loss
#   - Photon emission statistics
#   - Gate duration and fidelity
