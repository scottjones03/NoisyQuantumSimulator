# Qubit Routing
#
# Route operations to satisfy connectivity constraints.
#
# For fixed connectivity:
#   - SWAP insertion algorithms
#   - Heuristic mapping (SABRE, etc.)
#   - Optimal mapping (SAT/ILP based)
#
# For reconfigurable arrays (neutral atoms):
#   - Movement planning
#   - Parallel transport optimization
#   - Zone-based routing
#
# For QCCD ions:
#   - Shuttling path planning
#   - Junction scheduling
#   - Chain splitting/merging
#
# Cost models:
#   - SWAP cost (gate error)
#   - MOVE cost (heating, loss, time)
#   - Idle cost (decoherence during wait)
