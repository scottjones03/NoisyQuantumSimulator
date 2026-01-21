# Idle Primitive
#
# Models decoherence and loss during wait times.
#
# Used for:
#   - Qubits waiting while others execute gates
#   - Synchronization barriers
#   - Classical computation delays
#
# API:
#   Idle(qubit_id, duration)
#
# Returns IdleResult:
#   - duration: Wait time
#   - error_map: Accumulated decoherence (T1, T2)
#   - loss_probability: Background loss during wait
#   - heating: Motional heating accumulated
#
# Decoherence channels:
#   - T1 decay (energy relaxation)
#   - T2 dephasing (phase randomization)
#   - T2* (inhomogeneous dephasing)
#   - Leakage to non-computational states
#
# Platform specifics:
#   - Neutral atoms: Rydberg decay, trap lifetime
#   - Ions: Motional heating, magnetic field noise
#   - Cavity: Cavity-induced dephasing
