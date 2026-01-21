# Primitive Base Classes
#
# Abstract base classes defining the primitive interface contract.
# All hardware-specific primitive implementations must inherit from these.
#
# Design Principles:
#   1. Same API across all platforms - only parameters differ
#   2. Each primitive returns structured results (timing, errors, costs)
#   3. Primitives are stateless - they operate on provided state
#   4. Error models are CPTP maps or equivalent representations
#
# Base Classes:
#   - Primitive: Root class for all operations
#   - GatePrimitive: Base for single/two-qubit gates
#   - MovePrimitive: Base for transport operations
#   - MeasurePrimitive: Base for readout
#   - CoolPrimitive: Base for cooling/reset
#
# Result Types:
#   - PrimitiveResult: Contains timing, error_map, loss_prob, etc.
#   - GateResult: Adds fidelity, leakage
#   - MoveResult: Adds heating, trajectory
#   - MeasureResult: Adds confusion_matrix, outcome
