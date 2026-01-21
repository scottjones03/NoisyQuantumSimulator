# Atom/Ion Loss Models
#
# Stochastic loss of physical qubits.
#
# Loss mechanisms:
#   - Background gas collisions
#   - Heating above trap depth
#   - Off-resonant excitation during gates
#   - Rydberg decay to untrapped states
#   - Anti-trapping during Rydberg excitation
#   - Measurement-induced loss
#
# Loss modeling:
#   - Per-operation loss probability
#   - Cumulative loss tracking
#   - Conditional loss (e.g., loss given Rydberg excitation)
#
# Impact on QEC:
#   - Erasure errors (detectable loss)
#   - Replacement strategies
#   - Code deformation around lost qubits
#   - Threshold degradation
#
# Loss-as-erasure advantage:
#   - Higher threshold for erasure errors
#   - Erasure conversion techniques
#   - Dual-rail / flag qubit encoding
