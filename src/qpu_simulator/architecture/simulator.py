# Main Simulation Engine
#
# Core simulator that executes scheduled circuits with noise.
#
# Simulation modes:
#   1. Clifford (Stim-based)
#      - Efficient for stabilizer circuits
#      - Pauli frame tracking
#      - Detector error models
#   
#   2. State vector (small systems)
#      - Full quantum state
#      - For validation and debugging
#   
#   3. Density matrix (with noise)
#      - CPTP map application
#      - Exact error modeling
#   
#   4. Monte Carlo
#      - Stochastic error sampling
#      - For large-scale QEC simulation
#
# Simulation flow:
#   1. Load scheduled circuit
#   2. For each timestep:
#      a. Apply operations (with errors from primitives)
#      b. Track syndrome measurements
#      c. Accumulate loss events
#   3. Decode and compute logical outcome
#   4. Report performance metrics
#
# Integration:
#   - Uses primitives for error models
#   - Uses QEC module for decoding
#   - Uses topology for constraint checking
