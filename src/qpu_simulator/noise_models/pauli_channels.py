# Pauli Error Channels
#
# Standard Pauli-basis error models.
#
# Channel types:
#
# Depolarizing:
#   ρ → (1-p)ρ + (p/3)(XρX + YρY + ZρZ)
#   - Symmetric noise
#   - p = total error probability
#
# Dephasing (phase-flip):
#   ρ → (1-p)ρ + pZρZ
#   - Models T2 decay
#   - No population change
#
# Bit-flip:
#   ρ → (1-p)ρ + pXρX
#   - Models T1 decay (simplified)
#
# Asymmetric Pauli:
#   ρ → (1-px-py-pz)ρ + px·XρX + py·YρY + pz·ZρZ
#   - General single-qubit Pauli channel
#
# Two-qubit Pauli:
#   - 15 independent error rates
#   - Correlated vs uncorrelated errors
#   - Hardware-specific patterns (ZZ, IX, etc.)
#
# Conversion utilities:
#   - From gate fidelity to depolarizing rate
#   - From T1/T2 to error rates
#   - From process tomography to Pauli rates
