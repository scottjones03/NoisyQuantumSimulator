# Circuit Compiler
#
# Compiles high-level circuits to hardware-executable operations.
#
# Compilation stages:
#   1. Gate decomposition (to native gate set)
#   2. Qubit mapping (logical to physical)
#   3. Routing (insert SWAP/MOVE operations)
#   4. Scheduling (temporal ordering)
#   5. Optimization (gate cancellation, etc.)
#
# Native gate sets:
#   - Neutral atoms: {Rz, Raman, Rydberg-CZ}
#   - Trapped ions: {Rz, Raman, MS}
#   - Cavity QED: {Rz, cavity-CZ}
#
# Routing strategies:
#   - SWAP-based (fixed connectivity)
#   - Move-based (reconfigurable arrays)
#   - Hybrid (partial reconfiguration)
#
# Optimization passes:
#   - Gate cancellation
#   - Commutation-based reordering
#   - Template matching
#   - Movement minimization
