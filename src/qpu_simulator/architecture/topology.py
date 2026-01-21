# Topology Manager
#
# Manages physical qubit layout and connectivity.
#
# Representations:
#   - 2D grid (surface code compatible)
#   - Arbitrary graph (NetworkX based)
#   - Zone-based (QCCD architecture)
#   - Reconfigurable (neutral atoms)
#
# Connectivity types:
#   - Fixed: Static coupling graph
#   - Distance-based: Interaction strength vs separation
#   - Reconfigurable: Dynamic topology via movement
#
# Functions:
#   - get_neighbors(qubit_id): Adjacent qubits for gates
#   - get_distance(q1, q2): Physical distance
#   - can_interact(q1, q2): Check if gate is possible
#   - get_path(q1, q2): Movement path for reconfiguration
#
# Platform specifics:
#   - Neutral atoms: Blockade radius defines connectivity
#   - QCCD: Zone-based with shuttling between
#   - Penning: 2D crystal with long-range coupling
