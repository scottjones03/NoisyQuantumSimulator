# Primitives Layer (Level 1)
#
# Hardware-abstracted operation API that provides a common interface
# across all quantum hardware platforms.
#
# This layer consumes outputs from the micro-physics layer and exposes
# a unified API for the architectural simulator.
#
# Core Primitives:
#   - Move: Qubit/atom/ion transport
#   - SingleQubitGate: Arbitrary single-qubit rotations
#   - TwoQubitGate: Entangling operations (CZ, MS, etc.)
#   - Measure: State readout
#   - Cool: Motional state reset
#   - Idle: Wait/decoherence
#
# Each primitive encapsulates:
#   - CPTP error map
#   - Timing/duration
#   - Resource costs
#   - Loss probabilities
#   - Hardware constraints
