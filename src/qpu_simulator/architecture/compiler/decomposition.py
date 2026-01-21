# Gate Decomposition
#
# Decompose arbitrary gates into native gate set.
#
# Decompositions:
#   - Single-qubit: Euler angles (ZYZ, ZXZ, etc.)
#   - Two-qubit: Cartan/KAK decomposition
#   - Multi-qubit: Recursive decomposition
#
# Native gate mappings:
#   - CNOT → CZ + Hadamards
#   - Arbitrary rotation → Rz + native single-qubit
#   - Toffoli → T gates + CNOTs
#
# Optimization:
#   - Minimize gate count
#   - Prefer native gates
#   - Absorb single-qubit gates into adjacent two-qubit gates
