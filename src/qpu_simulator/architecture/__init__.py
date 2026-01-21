# Architecture Layer (Level 2)
#
# System-level simulation including scheduling, QEC, and compilation.
# This layer is fully shared across all hardware platforms.
#
# Submodules:
#   - scheduler: Operation scheduling with timing and parallelism
#   - topology: Physical qubit layout and connectivity
#   - qec: Quantum error correction protocols
#   - compiler: Circuit compilation and optimization
#   - simulator: Main simulation engine
#
# Key Responsibilities:
#   1. Circuit scheduling with hardware constraints
#   2. Enforcing physical adjacency and crosstalk rules
#   3. QEC syndrome extraction and decoding
#   4. Error propagation using primitive error models
#   5. Performance metric computation
#
# Integration with Stim:
#   - Clifford circuit simulation
#   - Pauli frame tracking
#   - Detector error model generation
